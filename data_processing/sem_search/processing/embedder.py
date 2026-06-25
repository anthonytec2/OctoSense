"""
Semantic Video Search - Gemma4 Embedding Pipeline

Uses Gemma4-31B (via vLLM) for dense captioning and Qwen3-Embedding-8B for text embedding.

Pipeline:
1. Gemma4 sees the image -> Generates a text description.
2. Text encoder reads the description -> Generates a vector.

Storage:
- Per-sequence H5 files in processed directory
- Stores captions, embeddings, and metadata for each sequence
"""

import numpy as np
import json
import logging
import time
import os
import base64
import asyncio
from pathlib import Path
from typing import List, Any, Dict, Optional
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# vLLM OpenAI-compatible API
from openai import AsyncOpenAI

# Disable transformers/sentence_transformers internal progress bars
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_error()

logger = logging.getLogger(__name__)




from .prompts import SYSTEM_PROMPT, _format_multi_prompt
from .telemetry import (
    TELEM_FIELDS,
    _load_ego_traj,
    _traj_summary,
    build_ego_motion_line,
    _local_datetime_from_bag_id,
    _telem_tuple,
    _telem_from_row,
)

VLM_ID = 'google/gemma-4-31B-it'
DEFAULT_VLLM_API_BASE = 'http://localhost:8000/v1'
MAX_TOKENS = 768
TEMPERATURE = 0.03

from ..text_encoder import (
    TEXT_EMBED_ID,
    load_text_embedder_only,
    generate_embeddings_from_captions,
)

from torchcodec.decoders import VideoDecoder as _VideoDecoder

def _crop_resize(arr: np.ndarray, size: int = 896) -> Image.Image:
    image = Image.fromarray(arr, 'RGB')
    w, h = image.size
    s = min(w, h)
    image = image.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return image.resize((size, size), Image.LANCZOS)


def _tensor_to_pil(tensor) -> Image.Image:
    """Convert torchcodec frame tensor (C, H, W) uint8 to cropped PIL Image."""
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    return _crop_resize(arr)




def extract_frames_window(video_path: str, frame_ids, device: str = 'cuda') -> List[Image.Image]:
    """Extract the given frames for one window via torchcodec GPU random-access decode.

    frame_ids: explicit frame indices
    """
    dec = _VideoDecoder(video_path, device=device)
    return [_tensor_to_pil(dec[int(i)].data) for i in frame_ids]


def encode_image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _text_part(text: str) -> Dict:
    return {"type": "text", "text": text}


def _image_part(b64: str) -> Dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


async def generate_captions_only_async(
    frames: List[Any],
    async_client: AsyncOpenAI,
    ego_lines: Optional[List[str]] = None,
) -> List[str]:
    """Generate captions using vLLM API (Phase 1).

    frames: each element is a list of PIL Images (a multi-frame window) captioned with
    CAPTION_PROMPT_MULTI. All windows are sent concurrently; the caller bounds the count
    by batching.
    ego_lines: optional per-window telemetry sentence (speed/yaw/time-of-day) injected
        into the multi-frame prompt's {ego_motion_line} slot; '' or None => model infers.
    """
    # Encode each window's frames to b64 and build its prompt (telemetry injected).
    encoded = []
    prompts = []
    for fi, frame in enumerate(frames):
        ego = ego_lines[fi] if (ego_lines and fi < len(ego_lines)) else ""
        encoded.append([encode_image_to_base64(f) for f in frame])
        prompts.append(_format_multi_prompt(ego))

    async def process_one(b64s, prompt, idx):
        try:
            content = []
            for i, img in enumerate(b64s):
                content += [_text_part(f"Frame {i + 1}:"), _image_part(img)]
            content.append(_text_part(prompt))
            resp = await async_client.chat.completions.create(
                model=VLM_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            if resp.choices:
                return idx, resp.choices[0].message.content.strip(), None
            return idx, "", "No choices in response"
        except Exception as e:
            return idx, "", str(e)

    captions = [""] * len(frames)
    tasks = [process_one(b, p, i) for i, (b, p) in enumerate(zip(encoded, prompts))]
    for coro in asyncio.as_completed(tasks):
        idx, caption, error = await coro
        if error:
            logger.warning(f"Caption generation failed for frame {idx}: {error}")
        captions[idx] = caption

    return captions


# ----------------- CAPTION STORAGE -----------------
def save_captions_temporary(video_id: str, windows: List[Any], processed_dir: str, model_name: str = 'gemma4-captioner'):
    """Save captions temporarily (Phase 1 output)"""
    import h5py
    from pathlib import Path
    
    parts = video_id.split('/')
    if len(parts) == 2:
        session, bag_name = parts
        sequence_dir = Path(processed_dir) / session / bag_name
    else:
        sequence_dir = Path(processed_dir) / video_id
    
    sequence_dir.mkdir(parents=True, exist_ok=True)
    h5_path = sequence_dir / 'captions_temp.h5'
    
    # Filter windows with captions
    windows_with_captions = [w for w in windows if hasattr(w, 'caption') and w.caption]
    
    if not windows_with_captions:
        logger.warning(f"No captions for {video_id}, skipping save")
        return h5_path
    
    captions = [w.caption for w in windows_with_captions]
    metadata_dtype = [('window_id', 'i4'), ('frame_idx', 'i4'), ('timestamp', 'f8')] + TELEM_FIELDS
    metadata = np.array(
        [(w.window_id, w.frame_idx, w.timestamp) + _telem_tuple(w) for w in windows_with_captions],
        dtype=metadata_dtype
    )
    
    with h5py.File(h5_path, 'w') as h5:
        h5.create_dataset('captions', data=[c.encode('utf-8') for c in captions], dtype=h5py.string_dtype())
        h5.create_dataset('metadata', data=metadata)
        h5.attrs['model_name'] = model_name
        h5.attrs['video_id'] = video_id
        h5.attrs['num_windows'] = len(windows_with_captions)
    
    logger.info(f"✓ Saved {len(windows_with_captions)} captions to {h5_path}")
    return h5_path

def load_captions_temporary(captions_path: Path) -> tuple:
    """Load captions from temporary file"""
    import h5py
    
    with h5py.File(captions_path, 'r') as h5:
        captions_bytes = h5['captions'][:]
        metadata = h5['metadata'][:]
        captions = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in captions_bytes]
    
    return captions, metadata

# ----------------- VIDEO EMBEDDING WRAPPER -----------------
async def embed_video_phase1_captions_async(video_id, mp4_path, h5_path, async_client, window_size_sec=5.0, batch_size=16, device='cuda'):
    """Phase 1: Generate captions only using vLLM API (async version) with overlapping preloading"""
    from .ingestion import extract_semantic_windows
    import asyncio

    logger.info(f"[Phase 1] {video_id}: extracting windows (window_size={window_size_sec}s)...")
    windows = extract_semantic_windows(h5_path, mp4_path, video_id, window_size_sec, device)
    logger.info(f"[Phase 1] {video_id}: {len(windows)} windows, generating captions in batches of {batch_size}")

    # Exact per-window frame indices for the 3 sampled frames: searchsorted on the
    # real img/left/t timestamps at [start, start+W/2, start+W]
    window_fids = [None] * len(windows)
    try:
        import h5py, hdf5plugin  # noqa: F401
        with h5py.File(h5_path, "r") as hf:
            img_t = hf["img/left/t"][()].astype(float).ravel()
        nfr = len(img_t)
        for k, w in enumerate(windows):
            tgt = (w.timestamp, w.timestamp + window_size_sec / 2.0, w.timestamp + window_size_sec)
            window_fids[k] = [int(np.clip(np.searchsorted(img_t, tt), 0, nfr - 1)) for tt in tgt]
    except Exception as e:
        logger.warning(f"{video_id}: timestamp frame-spacing failed ({e}); using ±span fallback")
        for k, w in enumerate(windows):
            c = w.frame_idx
            window_fids[k] = [max(0, c - 30), c, c + 30]

   
    dt_str, tod = _local_datetime_from_bag_id(video_id)
    time_ctx = dt_str or ""
    is_night = 1 if tod in ("nighttime", "dusk", "dawn") else 0
    half = float(window_size_sec) / 2.0
    window_ego = [""] * len(windows)
    try:
        import h5py, hdf5plugin  # noqa: F401
        with h5py.File(h5_path, "r") as hf:
            poses, ptime, src = _load_ego_traj(hf)
        if poses is not None:
            logger.info(f"[Phase 1] {video_id}: ego telemetry from {src} trajectory over the {window_size_sec:.0f}s window")
        for k, w in enumerate(windows):
            # window is [start, start+W]; center the summary at start+half so it
            # covers the SAME interval the 3 sampled frames span.
            s = _traj_summary(poses, ptime, float(w.timestamp) + half, half) if poses is not None else None
            window_ego[k] = build_ego_motion_line(s, time_ctx)
            # Stash structured telemetry on the window so it's persisted for hybrid
            # (structured + semantic) search — e.g. filter night + turning + speed range.
            w.telem = {
                "speed_mps": float(s["mean_mps"]) if s else float("nan"),
                "turn_deg": float(s["turn_deg"]) if s else float("nan"),
                "dist_m": float(s["dist_m"]) if s else float("nan"),
                "is_night": is_night,
            }
    except Exception as e:
        logger.warning(f"{video_id}: telemetry injection failed ({e}); captions will infer motion from frames")

    # Decode + caption each batch
    num_batches = (len(windows) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(windows), batch_size)):
        batch_windows = windows[i:i+batch_size]
        batch_ego = window_ego[i:i+batch_size]
        batch_fids = window_fids[i:i+batch_size]

        # Decode this batch; collect successes in parallel lists (empty caption on failure).
        ok_windows, ok_frames, ok_ego = [], [], []
        for w, fids, ego in zip(batch_windows, batch_fids, batch_ego):
            try:
                frames = extract_frames_window(mp4_path, fids)
            except Exception as e:
                logger.warning(f"{video_id}: failed to extract window {fids}: {e}")
                frames = None
            if frames:
                ok_windows.append(w)
                ok_frames.append(frames)
                ok_ego.append(ego)
            else:
                w.caption = ""

        if not ok_windows:
            logger.warning(f"{video_id}: no valid frames for batch {batch_idx+1}")
            continue

        captions = await generate_captions_only_async(
            ok_frames, async_client, ego_lines=ok_ego)
        for w, c in zip(ok_windows, captions):
            w.caption = c

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            logger.info(f"[Phase 1] {video_id}: processed {batch_idx + 1}/{num_batches} batches")
    
    logger.info(f"[Phase 1] Completed {video_id}: {len([w for w in windows if w.caption])}/{len(windows)} windows with captions")
    return windows


# ----------------- PER-SEQUENCE H5 STORAGE -----------------
def save_sequence_embeddings(
    video_id: str,
    embeddings: np.ndarray,
    captions: List[str],
    metadata: np.ndarray,
    processed_dir: str,
    model_name: str = 'gemma4-captioner',
):
    """
    Save embeddings, captions, and metadata to the per-sequence H5 file, under group
    {vlm_id}/{text_embed_id}/ with datasets: embeddings (N, D) float32, captions (N,)
    vlen-str, metadata (the structured array carried straight from captions_temp.h5),
    plus a config subgroup. The group is replaced if it already exists.
    """
    import h5py
    import re

    parts = video_id.split('/')
    if len(parts) == 2:
        session, bag_name = parts
        sequence_dir = Path(processed_dir) / session / bag_name
    else:
        sequence_dir = Path(processed_dir) / video_id
    sequence_dir.mkdir(parents=True, exist_ok=True)

    # HDF5 group names can't contain '/', so sanitize the model ids.
    group_path = "{}/{}".format(
        re.sub(r'[^\w\-.]', '_', VLM_ID),
        re.sub(r'[^\w\-.]', '_', TEXT_EMBED_ID),
    )

    h5_path = sequence_dir / 'semantic_embeddings.h5'
    logger.info(f"Saving {len(embeddings)} embeddings to {h5_path} under group {group_path}")
    with h5py.File(h5_path, 'w') as h5:
        group = h5.create_group(group_path)
        group.create_dataset('embeddings', data=embeddings, compression='gzip', compression_opts=9)
        group.create_dataset('captions', data=[c.encode('utf-8') for c in captions], dtype=h5py.string_dtype())
        group.create_dataset('metadata', data=metadata)
        config_group = group.create_group('config')
        config_group.attrs['model_name'] = model_name
        config_group.attrs['vlm_id'] = VLM_ID
        config_group.attrs['text_embed_id'] = TEXT_EMBED_ID
        config_group.attrs['embedding_dim'] = embeddings.shape[1]
        config_group.attrs['num_windows'] = len(embeddings)
        config_group.attrs['video_id'] = video_id

# ----------------- TOP-LEVEL INGESTION (PHASE 1: CAPTIONS ONLY) -----------------
def embed_all_videos_phase1_captions(
    metadata_path: str,
    window_size_sec: float = 5.0,
    batch_size: int = 16,
    device: str = 'cuda',
    model_name: str = 'gemma4-captioner',
    vllm_api_base: str = None,
    processed_dir: Optional[str] = None,
    session_filter: Optional[str] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Phase 1: Generate captions using vLLM API and save to H5 files.
    
    Args:
        vllm_api_base: vLLM API base URL (defaults to config value)
        processed_dir: Base directory for per-sequence H5 files
        session_filter: Optional session name to filter videos (e.g., 'sess8')
        resume: If True, skip videos that already have captions
    """
    import h5py
    
    with open(metadata_path, 'r') as f:
        videos = json.load(f)

    valid_videos = [v for v in videos if Path(v['mp4_path']).exists()]
    
    # Filter by session if specified
    if session_filter:
        valid_videos = [
            v for v in valid_videos 
            if session_filter in v.get('video_id', '') or session_filter in v.get('mp4_path', '')
        ]
        logger.info(f"Filtered to session '{session_filter}': {len(valid_videos)} videos")
    
    # Determine processed directory
    if processed_dir is None:
        if valid_videos:
            first_mp4 = Path(valid_videos[0]['mp4_path'])
            processed_dir = str(first_mp4.parent.parent.parent)
        else:
            raise ValueError("No valid videos found and processed_dir not specified")
    
    api_base = vllm_api_base or DEFAULT_VLLM_API_BASE
    
    logger.info(f"Phase 1: Generating captions with vLLM API")
    logger.info(f"  Videos: {len(valid_videos)}")
    logger.info(f"  Window size: {window_size_sec}s")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  vLLM API Base: {api_base}")
    logger.info(f"  Per-sequence H5 storage: {processed_dir}")
    logger.info(f"  Resume mode: {resume}")
    
    # Filter out videos that already have captions if resume=True
    if resume:
        from .ingestion import extract_semantic_windows
        videos_to_process = []
        skipped_count = 0
        
        for v in valid_videos:
            # Estimate expected window count
            try:
                windows = extract_semantic_windows(v['h5_path'], v['mp4_path'], v['video_id'], window_size_sec, device)
                expected_windows = len(windows)
            except Exception as e:
                logger.warning(f"Could not extract windows for {v['video_id']} to check resume status: {e}")
                expected_windows = None
            
            if check_captions_exist(v['video_id'], processed_dir, expected_windows, model_name):
                skipped_count += 1
                logger.info(f"[Phase 1] Skipping {v['video_id']} (captions already exist)")
            else:
                videos_to_process.append(v)
        
        valid_videos = videos_to_process
        logger.info(f"Resume mode: Skipped {skipped_count} videos with existing captions, {len(valid_videos)} remaining")
    
    if not valid_videos:
        logger.info("No videos to process (all already have captions)")
        return {
            'count': 0,
            'videos_processed': 0,
            'elapsed_sec': 0,
            'fps': 0
        }
    
    start_time = time.time()
    total_captions = 0

    async def process_all_videos():
        """Process all videos in async context with proper client management."""
        nonlocal total_captions
        # Create async client within the async context
        async_client = AsyncOpenAI(base_url=api_base, api_key="none", timeout=120.0)
        try:
            for vid_idx, v in enumerate(valid_videos, 1):
                logger.info(f"[Phase 1] [{vid_idx}/{len(valid_videos)}] Processing {v['video_id']}")
                try:
                    windows = await embed_video_phase1_captions_async(
                        v['video_id'], v['mp4_path'], v['h5_path'], async_client,
                        window_size_sec, batch_size, device
                    )
                    
                    # Save captions to H5 file
                    captions_h5_path = save_captions_temporary(v['video_id'], windows, processed_dir, model_name)
                    
                    caption_count = sum(1 for w in windows if hasattr(w, 'caption') and w.caption)
                    total_captions += caption_count
                    logger.info(f"[Phase 1] [{vid_idx}/{len(valid_videos)}] Completed {v['video_id']}: {caption_count} captions")
                except Exception as e:
                    logger.error(f"[Phase 1] [{vid_idx}/{len(valid_videos)}] Failed {v['video_id']}: {e}", exc_info=True)
        finally:
            # Close async client before event loop closes
            # AsyncOpenAI.close() is synchronous, no await needed
            async_client.close()
    
    # Run async processing
    asyncio.run(process_all_videos())
    
    elapsed = time.time() - start_time
    fps = total_captions / elapsed if elapsed > 0 else 0
    
    logger.info(f"=" * 60)
    logger.info(f"PHASE 1 COMPLETE")
    logger.info(f"  Videos processed: {len(valid_videos)}")
    logger.info(f"  Total captions: {total_captions}")
    logger.info(f"  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    logger.info(f"  Throughput: {fps:.2f} captions/sec")
    logger.info(f"  Captions saved to H5 files in: {processed_dir}")
    logger.info(f"=" * 60)
    
    return {
        'count': total_captions,
        'videos_processed': len(valid_videos),
        'elapsed_sec': elapsed,
        'fps': fps
    }

# ----------------- TOP-LEVEL INGESTION (PHASE 2: EMBEDDINGS FROM CAPTIONS) -----------------

def embed_all_videos_phase2_embeddings(
    metadata_path: str,
    output_dir: str,
    window_size_sec: float = 5.0,
    batch_size: int = 64,
    device: str = 'cuda',
    model_name: str = 'gemma4-captioner',
    processed_dir: Optional[str] = None,
    session_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Phase 2: Generate embeddings from captions stored in H5 files.
    
    Reads captions from H5 files created in Phase 1, generates embeddings,
    and saves to per-sequence H5 files.
    
    Args:
        session_filter: Optional session name to filter videos (e.g., 'sess8')
    """
    import h5py
    
    with open(metadata_path, 'r') as f:
        videos = json.load(f)

    valid_videos = [v for v in videos if Path(v['mp4_path']).exists()]
    
    # Filter by session if specified
    if session_filter:
        valid_videos = [
            v for v in valid_videos 
            if session_filter in v.get('video_id', '') or session_filter in v.get('mp4_path', '')
        ]
        logger.info(f"Filtered to session '{session_filter}': {len(valid_videos)} videos")
    
    # Determine processed directory
    if processed_dir is None:
        if valid_videos:
            first_mp4 = Path(valid_videos[0]['mp4_path'])
            processed_dir = str(first_mp4.parent.parent.parent)
        else:
            raise ValueError("No valid videos found and processed_dir not specified")
    
    logger.info(f"Phase 2: Generating embeddings from captions")
    logger.info(f"  Videos: {len(valid_videos)}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Per-sequence H5 storage: {processed_dir}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    all_embeddings, all_metadata = [], []

    video_ids = [v['video_id'] for v in valid_videos]

    # Single GPU path
    text_embedder_model = load_text_embedder_only(model_name, device)
    logger.info("Text embedder loaded, starting embedding generation...")

    for vid_idx, video_id in enumerate(tqdm(video_ids, desc="[Phase 2] Generating embeddings", unit="video"), 1):
        try:
            captions_h5_path = load_captions_temporary_path(video_id, processed_dir)
            if not captions_h5_path or not captions_h5_path.exists():
                logger.warning(f"[Phase 2] No captions file for {video_id}, skipping")
                continue
            captions, metadata = load_captions_temporary(captions_h5_path)
            if not captions:
                continue
           
            embeddings = generate_embeddings_from_captions(captions, text_embedder_model)
            save_sequence_embeddings(video_id, embeddings, captions, metadata, processed_dir, model_name)
            for emb, cap, row in zip(embeddings, captions, metadata):
                all_embeddings.append(emb)
                all_metadata.append({
                    "video_id": video_id,
                    "caption": cap,
                    "window_id": int(row["window_id"]),
                    "frame_idx": int(row["frame_idx"]),
                    "timestamp": float(row["timestamp"]),
                    **(_telem_from_row(row) or {}),
                })
            logger.info(f"[Phase 2] [{vid_idx}/{len(video_ids)}] {video_id}: {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"[Phase 2] [{vid_idx}/{len(video_ids)}] Failed {video_id}: {e}", exc_info=True)

    if not all_embeddings:
        logger.error("No embeddings generated.")
        return {'count': 0, 'videos_processed': 0, 'elapsed_sec': 0, 'fps': 0}

    logger.info(f"Processing complete: {len(all_embeddings)} embeddings from {len(valid_videos)} videos")
    logger.info("Building global FAISS index...")
    
    # Stack & Normalize for global index
    arr = np.vstack(all_embeddings).astype(np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
    
    logger.info(f"Embeddings shape: {arr.shape}, normalized (mean norm: {np.linalg.norm(arr, axis=1).mean():.4f})")
    
    # Save global metadata
    logger.info(f"Saving global metadata to {out_path / 'metadata.json'}...")
    with open(out_path / 'metadata.json', 'w') as f:
        json.dump(all_metadata, f)
    
    # Build FAISS index
    try:
        import faiss
        d = arr.shape[1]
        logger.info(f"Building FAISS index (dim={d}, vectors={len(all_embeddings)})...")
        index = faiss.IndexFlatIP(d)
        index.add(arr)
        faiss.write_index(index, str(out_path / 'faiss.index'))
        logger.info(f"FAISS index saved: {index.ntotal} vectors")
        
        with open(out_path / 'index_config.json', 'w') as f:
            json.dump({
                'model_name': model_name, 
                'embedding_dim': d, 
                'count': index.ntotal,
                'window_size_sec': window_size_sec,
                'processed_dir': processed_dir
            }, f)
        logger.info(f"Index config saved")
    except ImportError:
        logger.warning("FAISS not installed, skipping index build.")

    elapsed = time.time() - start_time
    fps = len(all_embeddings) / elapsed if elapsed > 0 else 0
    
    logger.info(f"=" * 60)
    logger.info(f"PHASE 2 COMPLETE")
    logger.info(f"  Videos processed: {len(valid_videos)}")
    logger.info(f"  Total embeddings: {len(all_embeddings)}")
    logger.info(f"  Embedding dimension: {arr.shape[1]}")
    logger.info(f"  Window size: {window_size_sec}s")
    logger.info(f"  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    logger.info(f"  Throughput: {fps:.2f} embeddings/sec")
    logger.info(f"  Per-sequence H5 files saved to: {processed_dir}")
    logger.info(f"=" * 60)
    
    return {
        'count': len(all_embeddings),
        'videos_processed': len(valid_videos),
        'elapsed_sec': elapsed,
        'fps': fps
    }

def load_captions_temporary_path(video_id: str, processed_dir: str) -> Optional[Path]:
    """Get path to temporary captions H5 file."""
    parts = video_id.split('/')
    if len(parts) == 2:
        session, bag_name = parts
        sequence_dir = Path(processed_dir) / session / bag_name
    else:
        sequence_dir = Path(processed_dir) / video_id
    
    captions_h5_path = sequence_dir / 'captions_temp.h5'
    return captions_h5_path if captions_h5_path.exists() else None

def check_captions_exist(video_id: str, processed_dir: str, expected_windows: int = None, model_name: str = 'gemma4-captioner') -> bool:
    """
    Check if captions already exist for a video.
    
    Args:
        video_id: Video ID to check
        processed_dir: Base directory for per-sequence H5 files
        expected_windows: Optional expected number of windows (for validation)
        model_name: Model name to verify matches
    
    Returns:
        True if captions exist and are valid, False otherwise
    """
    captions_path = load_captions_temporary_path(video_id, processed_dir)
    if not captions_path or not captions_path.exists():
        return False
    
    try:
        import h5py
        with h5py.File(captions_path, 'r') as h5:
            # Check if file has required datasets
            if 'captions' not in h5 or 'metadata' not in h5:
                logger.warning(f"Invalid captions file for {video_id}: missing datasets")
                return False
            
            # Check model name matches
            if h5.attrs.get('model_name') != model_name:
                logger.warning(f"Captions file for {video_id} has different model ({h5.attrs.get('model_name')} vs {model_name})")
                return False
            
            # Check video_id matches
            if h5.attrs.get('video_id') != video_id:
                logger.warning(f"Captions file video_id mismatch: {h5.attrs.get('video_id')} vs {video_id}")
                return False
            
            num_windows = len(h5['captions'])
            
            # Validate expected count if provided
            if expected_windows is not None:
                if num_windows != expected_windows:
                    logger.warning(f"Captions count mismatch for {video_id}: {num_windows} vs {expected_windows}")
                    return False
            
            # Check that captions are not empty
            if num_windows == 0:
                logger.warning(f"Empty captions file for {video_id}")
                return False
            
            logger.debug(f"Found valid captions for {video_id}: {num_windows} windows")
            return True
            
    except Exception as e:
        logger.warning(f"Error checking captions for {video_id}: {e}")
        return False

