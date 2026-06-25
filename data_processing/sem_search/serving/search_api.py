"""
Semantic Video Search - FastAPI Search API

REST API for semantic video search with:
- Cross-sequence diversity prioritization
- Adjacent window merging
- FFmpeg clip extraction for remote viewing
"""
import os
import bisect
import urllib.parse
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import logging
import h5py
import hdf5plugin
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .diversity import rerank_with_diversity, calculate_diversity_score, SearchResult
from .window_merger import merge_all_by_sequence, MergedWindow
from .clip_extractor import ClipExtractor, get_clip_url

logger = logging.getLogger(__name__)

# Thread pool executor for running blocking operations
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="search_worker")

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Video Search API",
    description="Search robotics video logs using natural language queries",
    version="1.0.0"
)

clip_extractor = ClipExtractor(fps=100.0)


# ============================================================================
# Data Models
# ============================================================================
class SearchResultItem(BaseModel):
    """Single search result item."""
    video_id: str
    sequence_id: str
    start_time: float
    end_time: float
    start_frame_idx: int
    end_frame_idx: int
    duration: float
    score: float
    merged_count: int
    mp4_path: str
    h5_path: str
    clip_url: str
    caption: Optional[str] = None  # Caption for the clip (if available)


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[SearchResultItem]
    total_candidates: int
    diversity_score: float


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class VideoMetadataEntry:
    """Metadata for a single video."""
    video_id: str
    mp4_path: str
    h5_path: str
    frame_count: int
    duration_sec: float
    timestamp_start: float
    timestamp_end: float


# In-memory metadata store (populated from the index's metadata.json on load)
VIDEO_METADATA: Dict[str, VideoMetadataEntry] = {}

# Global search index (loaded on startup)
_SEARCH_INDEX = {
    'faiss_index': None,
    'window_metadata': None,
    'model_name': None,  # Model used to build the index
    'bm25': None,        # BM25Okapi over captions (for hybrid lexical+dense search)
    'loaded': False
}


def load_search_index(index_dir: str) -> bool:
    """
    Load FAISS index and window metadata from disk.
        Also pre-loads the text embedding model to avoid blocking on first search.
    """
    import json
    import numpy as np
    
    index_path = Path(index_dir)
    faiss_path = index_path / 'faiss.index'
    metadata_path = index_path / 'metadata.json'
    config_path = index_path / 'index_config.json'
    
    if not faiss_path.exists():
        logger.warning(f"FAISS index not found: {faiss_path}")
        return False
    
    if not metadata_path.exists():
        logger.warning(f"Window metadata not found: {metadata_path}")
        return False
    
    try:
        import faiss
        
        # Load FAISS index
        index = faiss.read_index(str(faiss_path))
        
        model_name = 'gemma4-captioner'  # Default fallback (matches embedder default)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if config.get('type') == 'ivf' and hasattr(index, 'nprobe'):
                nprobe = config.get('nprobe', 10)
                index.nprobe = nprobe
                logger.info(f"Set IVF nprobe to {nprobe}")
            
            # Get model name from config (for query encoding)
            model_name = config.get('model_name', 'gemma4-captioner')
            logger.info(f"Index was built with model: {model_name}")
        
        _SEARCH_INDEX['faiss_index'] = index
        _SEARCH_INDEX['model_name'] = model_name
        logger.info(f"Loaded FAISS index: {index.ntotal} vectors")
        
        # Load window metadata
        with open(metadata_path, 'r') as f:
            _SEARCH_INDEX['window_metadata'] = json.load(f)
        logger.info(f"Loaded window metadata: {len(_SEARCH_INDEX['window_metadata'])} windows")

       
        cap_by_key = {}
        frames_by_video = {}
        for w in _SEARCH_INDEX['window_metadata']:
            vid = w.get('video_id'); fi = w.get('frame_idx')
            cap = w.get('caption', '') or ''
            if vid is None or fi is None:
                continue
            cap_by_key[(vid, int(fi))] = cap
            frames_by_video.setdefault(vid, []).append(int(fi))
        for vid in frames_by_video:
            frames_by_video[vid].sort()
        _SEARCH_INDEX['caption_by_key'] = cap_by_key
        _SEARCH_INDEX['frames_by_video'] = frames_by_video
        logger.info(f"Built caption lookup over {len(cap_by_key)} windows")

        # BM25 (hybrid lexical+dense)
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [(w.get('caption', '') or '').lower().split()
                         for w in _SEARCH_INDEX['window_metadata']]
            _SEARCH_INDEX['bm25'] = BM25Okapi(tokenized)
            logger.info(f"Built BM25 over {len(tokenized)} captions (hybrid search enabled)")
        except ImportError:
            logger.info("rank_bm25 not installed; dense-only search")
            _SEARCH_INDEX['bm25'] = None
        except Exception as e:
            logger.warning(f"BM25 build failed ({e}); dense-only search")
            _SEARCH_INDEX['bm25'] = None
        
        # Per-video mp4/h5 paths are resolved on demand from DATA_BASE_PATH
        # (the HF dataset root) in load_video_metadata — not from build-time paths.
        _SEARCH_INDEX['loaded'] = True
        
        # Pre-load the text embedding model to avoid blocking on first search
        try:
            logger.info(f"Pre-loading text embedding model: {model_name}")
            from ..text_encoder import encode_text_query
            # Run a dummy encode to load and cache the model
            _ = encode_text_query("test", model_name=model_name)
            logger.info("Text embedding model pre-loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to pre-load text embedding model: {e}")
            logger.warning("Model will be loaded on first search (may cause slight delay)")
        
        return True
        
    except ImportError:
        logger.error("FAISS not installed. Install with: pip install faiss-gpu")
        return False
    except Exception as e:
        logger.error(f"Failed to load search index: {e}")
        return False


def _rrf_fuse(ranked_lists, k: int = 60):
    """Reciprocal Rank Fusion of several ranked index lists. Returns
    [(idx, rrf_score), ...] sorted by descending fused score. Rank-based, so it
    cleanly combines dense cosine and BM25 (incomparable score scales)."""
    agg = {}
    for lst in ranked_lists:
        for rank, idx in enumerate(lst):
            agg[idx] = agg.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(agg.items(), key=lambda x: -x[1])


def search_faiss(query: str, top_k: int = 100) -> List[tuple]:
    """
    Search the index with a text query — hybrid dense (FAISS) + lexical (BM25),
    fused with Reciprocal Rank Fusion so exact terms (sign text 'EXIT 30',
    'ambulance') get lexical lift while semantics still drive recall. Dense-only
    if no BM25 index is loaded.
    """
    if not _SEARCH_INDEX['loaded']:
        raise RuntimeError("Search index not loaded — run ingest-all and start the server with a valid --index-dir")

    try:
        from ..text_encoder import encode_text_query
        import numpy as np
        
        # Get model name from index config (must match the model used to build index)
        model_name = _SEARCH_INDEX.get('model_name', 'gemma4-captioner')
        
        # Encode query text using the same model that built the index
        query_embedding = encode_text_query(query, model_name=model_name)
        
        # Ensure proper shape: FAISS expects (n_queries, d)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        elif query_embedding.ndim == 2 and query_embedding.shape[0] != 1:
            logger.warning(f"Unexpected query shape: {query_embedding.shape}, reshaping to (1, {query_embedding.shape[-1]})")
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure float32 for FAISS compatibility
        if query_embedding.dtype != np.float32:
            logger.debug(f"Converting query embedding from {query_embedding.dtype} to float32")
            query_embedding = query_embedding.astype(np.float32)
        
        query_norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        mean_norm = query_norms.mean()
        
        if abs(mean_norm - 1.0) > 0.001:
            logger.warning(f"Query embedding not normalized (mean norm={mean_norm:.6f}), fixing...")
            query_embedding = query_embedding / (query_norms + 1e-8)
            new_norms = np.linalg.norm(query_embedding, axis=1)
            logger.info(f"Normalized query embedding (new mean norm: {new_norms.mean():.6f})")
        else:
            logger.debug(f"Query embedding already normalized (mean norm: {mean_norm:.6f})")
        
        # Log embedding statistics for debugging
        logger.debug(f"Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
        logger.debug(f"Query embedding stats: min={query_embedding.min():.4f}, max={query_embedding.max():.4f}, mean={query_embedding.mean():.4f}")
        
        # Search FAISS
        faiss_index = _SEARCH_INDEX['faiss_index']
        
        # Verify index dimensions match
        if hasattr(faiss_index, 'd'):
            if faiss_index.d != query_embedding.shape[1]:
                logger.error(f"Dimension mismatch! Index dim={faiss_index.d}, query dim={query_embedding.shape[1]}")
                raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} doesn't match index dimension {faiss_index.d}")
            logger.debug(f"Index dimension verified: {faiss_index.d}")
        
        # Log index configuration
        if hasattr(faiss_index, 'nprobe'):
            logger.info(f"Searching with nprobe={faiss_index.nprobe}")
        
        scores, indices = faiss_index.search(query_embedding, top_k)
        
        # Log score statistics
        valid_mask = indices[0] >= 0
        valid_scores = scores[0][valid_mask]
        valid_indices = indices[0][valid_mask]
        
        logger.info(f"FAISS search for '{query}': found {len(valid_indices)} valid candidates (out of {top_k} requested)")
        
        if len(valid_scores) > 0:
            logger.info(
                f"Score statistics: min={valid_scores.min():.4f}, "
                f"max={valid_scores.max():.4f}, mean={valid_scores.mean():.4f}, "
                f"median={np.median(valid_scores):.4f}"
            )
        else:
            logger.error("No valid results found! All indices were -1")
            logger.error("This indicates a serious index problem - rebuild the index")
            return []
        
        # Build results - only use valid indices
        window_metadata = _SEARCH_INDEX['window_metadata']
        results = []

        dense_order = [int(j) for j in valid_indices]
        bm25 = _SEARCH_INDEX.get('bm25')
        if bm25 is not None:
            try:
                toks = query.lower().split()
                bm_scores = bm25.get_scores(toks)
                bm_order = [int(j) for j in np.argsort(bm_scores)[::-1][:top_k]]
                fused = _rrf_fuse([dense_order, bm_order])
                top = max((s for _, s in fused), default=1.0) or 1.0
                ranked = [(s / top, idx) for idx, s in fused][:top_k]
                logger.info(f"Hybrid RRF fusion: {len(dense_order)} dense + {len(bm_order)} bm25 → {len(ranked)} fused")
            except Exception as e:
                logger.warning(f"BM25 fusion failed ({e}); dense-only ranking")
                ranked = list(zip([float(s) for s in valid_scores], dense_order))
        else:
            ranked = list(zip([float(s) for s in valid_scores], dense_order))

        logger.info(f"Top 5 candidates before diversity reranking:")

        for i, (score, idx) in enumerate(ranked):
            if idx < 0 or idx >= len(window_metadata):
                logger.warning(f"Invalid index {idx} (metadata length: {len(window_metadata)})")
                continue
            
            window = window_metadata[idx]
            video_id = window['video_id']
            
            # Get paths from video metadata
            if video_id in VIDEO_METADATA:
                meta = VIDEO_METADATA[video_id]
                mp4_path = meta.mp4_path
                h5_path = meta.h5_path
            else:
                mp4_path = ''
                h5_path = ''
            
            window_data = {
                'frame_idx': window['frame_idx'],
                'timestamp': window['timestamp'],
                'mp4_path': mp4_path,
                'h5_path': h5_path
            }
            
            results.append((float(score), video_id, window_data))
            
            # Log top 5
            if i < 5:
                logger.info(f"  {i+1}. Score: {score:.4f}, Video: {video_id}, Time: {window_data['timestamp']:.2f}s")
        
        logger.info(f"Returning {len(results)} valid results from FAISS search")
        return results
        
    except Exception as e:
        logger.error(f"FAISS search failed: {e}", exc_info=True)
        raise


def load_video_metadata(video_id: str) -> VideoMetadataEntry:
    """
    Resolve a video's files in the HF dataset layout:
        <DATA_BASE_PATH>/<session>/<bag>/{img_left.mp4, data.h5}
    DATA_BASE_PATH points at the dataset root (e.g. hf_staging_octosense).
    """
    if video_id in VIDEO_METADATA:
        return VIDEO_METADATA[video_id]

    base_path = os.environ.get('DATA_BASE_PATH', '/data/rosbags/hf_staging_octosense')
    parts = video_id.split('/')

    if len(parts) == 2:
        session, bag_name = parts
        seq_dir = f"{base_path}/{session}/{bag_name}"
    else:
        seq_dir = f"{base_path}/{video_id}"
    mp4_path = f"{seq_dir}/img_left.mp4"
    h5_path = f"{seq_dir}/data.h5"
    
    # Load timestamp info from H5
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as h5:
            if 'img/left/t' in h5:
                timestamps = h5['img/left/t'][:]
                frame_count = len(timestamps)
                t_start = float(timestamps[0])
                t_end = float(timestamps[-1])
                duration = t_end - t_start
            else:
                frame_count = 0
                t_start = 0.0
                t_end = 0.0
                duration = 0.0
    else:
        frame_count = 0
        t_start = 0.0
        t_end = 0.0
        duration = 0.0
    
    entry = VideoMetadataEntry(
        video_id=video_id,
        mp4_path=mp4_path,
        h5_path=h5_path,
        frame_count=frame_count,
        duration_sec=duration,
        timestamp_start=t_start,
        timestamp_end=t_end
    )
    
    VIDEO_METADATA[video_id] = entry
    return entry



# ============================================================================
# Search Endpoints
# ============================================================================

def _filter_same_sequence_by_time(
    results: List[SearchResult],
    min_time_diff_sec: float = 120.0
) -> List[SearchResult]:
    """
    Filter results from the same sequence to ensure they're at least min_time_diff_sec apart.
    
    Keeps results in order of score, but removes any result from a sequence if there's
    already a result from that sequence within min_time_diff_sec.
    
    Args:
        results: List of SearchResult objects (should be sorted by score descending)
        min_time_diff_sec: Minimum time difference in seconds between results from same sequence
    
    Returns:
        Filtered list of SearchResult objects
    """
    if not results:
        return []
    
    filtered: List[SearchResult] = []
    # Track the last timestamp for each sequence
    last_timestamp_by_sequence: Dict[str, float] = {}
    
    for result in results:
        seq_id = result.sequence_id
        timestamp = result.timestamp
        
        # Check if we have a previous result from this sequence
        if seq_id in last_timestamp_by_sequence:
            last_timestamp = last_timestamp_by_sequence[seq_id]
            time_diff = abs(timestamp - last_timestamp)
            
            # Skip if too close in time
            if time_diff < min_time_diff_sec:
                logger.debug(
                    f"Skipping result from {seq_id} at {timestamp:.2f}s "
                    f"(only {time_diff:.2f}s after previous at {last_timestamp:.2f}s)"
                )
                continue
        
        # Add this result and update the last timestamp for this sequence
        filtered.append(result)
        last_timestamp_by_sequence[seq_id] = timestamp
    
    if len(filtered) < len(results):
        logger.info(
            f"Filtered {len(results)} results to {len(filtered)} "
            f"(removed {len(results) - len(filtered)} results from same sequence within {min_time_diff_sec}s)"
        )
    
    return filtered


def _caption_from_index(video_id: str, start_frame_idx: int) -> Optional[str]:
    """Caption for a (merged) clip: the window whose frame_idx is nearest the clip
    start, from the in-memory window metadata (metadata.json). An exact start-frame
    match is just the distance-0 case. None if no caption index is loaded."""
    cap_by_key = _SEARCH_INDEX.get('caption_by_key')
    frames = _SEARCH_INDEX.get('frames_by_video', {}).get(video_id)
    if not cap_by_key or not frames:
        return None
    s = int(start_frame_idx)
    i = bisect.bisect_left(frames, s)
    cand = [frames[j] for j in (i - 1, i) if 0 <= j < len(frames)]
    best = min(cand, key=lambda f: abs(f - s)) if cand else None
    return cap_by_key.get((video_id, best)) if best is not None else None


def run_search_pipeline(q: str, limit: int = 12, diversity_weight: float = 0.3,
                        max_gap_sec: float = 10.0, include_captions: bool = False) -> SearchResponse:
    """Core semantic-search pipeline (synchronous): dense+BM25 candidates ->
    diversity rerank -> same-sequence time filter -> merge adjacent windows ->
    build result items. Shared by the /search HTTP route and the cli_search
    command line so both rank identically."""
    candidates = search_faiss(q, 100)
    total_candidates = len(candidates)

    diverse_results = rerank_with_diversity(
        candidates, max_results=limit * 2, diversity_weight=diversity_weight)
    # Keep same-sequence hits >=2 min apart so the top-N spans distinct moments.
    diverse_results = _filter_same_sequence_by_time(diverse_results, min_time_diff_sec=120.0)

    windows = [
        {'video_id': r.video_id, 'sequence_id': r.sequence_id, 'timestamp': r.timestamp,
         'frame_idx': r.frame_idx, 'score': r.score, 'mp4_path': r.mp4_path, 'h5_path': r.h5_path}
        for r in diverse_results
    ]
    merged_results = merge_all_by_sequence(windows, max_gap_sec=max_gap_sec, window_duration=5.0)
    merged_results = merged_results[:limit]
    div_score = calculate_diversity_score(diverse_results[:limit])

    result_items = []
    for m in merged_results:
        clip_url = get_clip_url(m.video_id, start_frame_idx=m.start_frame_idx, end_frame_idx=m.end_frame_idx)
        caption = _caption_from_index(m.video_id, m.start_frame_idx) if include_captions else None
        result_items.append(SearchResultItem(
            video_id=m.video_id, sequence_id=m.sequence_id, start_time=m.start_time,
            end_time=m.end_time, start_frame_idx=m.start_frame_idx, end_frame_idx=m.end_frame_idx,
            duration=m.duration, score=m.score, merged_count=m.merged_count,
            mp4_path=m.mp4_path or '', h5_path=m.h5_path or '', clip_url=clip_url, caption=caption))

    return SearchResponse(query=q, results=result_items,
                          total_candidates=total_candidates, diversity_score=div_score)


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Natural language search query"),
    limit: int = Query(12, description="Maximum results to return"),
    diversity_weight: float = Query(0.3, description="Diversity penalty weight (0-1)"),
    max_gap_sec: float = Query(10.0, description="Max gap for merging adjacent windows"),
    include_captions: bool = Query(False, description="Include captions in search results")
):
    """Semantic search over video logs (HTTP wrapper around run_search_pipeline)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: run_search_pipeline(q, limit, diversity_weight, max_gap_sec, include_captions))




# ============================================================================
# Clip Extraction Endpoint
# ============================================================================

@app.get("/api/clip/{video_id:path}")
async def get_clip(
    video_id: str,
    start_frame_idx: int = Query(..., description="Start frame index (0-based, inclusive)"),
    end_frame_idx: int = Query(..., description="End frame index (0-based, inclusive)")
):
    """
    Extract and stream a video clip using frame indices.
    
    Uses FFmpeg to extract a segment from the source video without
    loading the entire file, enabling efficient remote viewing.

    """
    # URL decode video_id
    video_id = urllib.parse.unquote(video_id)
    
    # Load metadata
    try:
        metadata = load_video_metadata(video_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
    
    # Check if files exist
    if not os.path.exists(metadata.mp4_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {metadata.mp4_path}")
    
    # Validate frame indices
    if start_frame_idx < 0:
        raise HTTPException(status_code=400, detail=f"start_frame_idx must be >= 0, got {start_frame_idx}")
    if end_frame_idx < start_frame_idx:
        raise HTTPException(status_code=400, detail=f"end_frame_idx ({end_frame_idx}) must be >= start_frame_idx ({start_frame_idx})")
    if end_frame_idx >= metadata.frame_count:
        logger.warning(
            f"end_frame_idx ({end_frame_idx}) exceeds frame_count ({metadata.frame_count}), "
            f"clamping to {metadata.frame_count - 1}"
        )
        end_frame_idx = metadata.frame_count - 1
        if end_frame_idx < start_frame_idx:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid frame range: start_frame_idx={start_frame_idx}, end_frame_idx={end_frame_idx}"
            )
    
    # Stream the clip
    try:
        video_stream = clip_extractor.extract_clip_streaming(
            metadata.mp4_path,
            start_frame_idx=start_frame_idx,
            end_frame_idx=end_frame_idx,
        )
        filename = f"clip_{video_id.replace('/', '_')}_frames_{start_frame_idx}_{end_frame_idx}.mp4"

        return StreamingResponse(
            video_stream,
            media_type="video/mp4",
            headers={
                'Content-Disposition': f'inline; filename="{filename}"',
                'Accept-Ranges': 'bytes'
            }
        )
    except Exception as e:
        logger.error(f"Clip extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clip extraction failed: {str(e)}")


# ============================================================================
# Thumbnail Endpoint
# ============================================================================

@app.get("/api/thumbnail/{video_id:path}")
async def get_thumbnail(
    video_id: str,
    frame_idx: int = Query(..., description="Frame index (0-based) for thumbnail")
):
    """
    Get a thumbnail image for a video at a specific frame index.
    
    Uses timestamp-based seeking for efficient frame extraction from videos.
    Calculates the timestamp from the frame index and seeks directly to that time.
    """
    video_id = urllib.parse.unquote(video_id)
    
    try:
        metadata = load_video_metadata(video_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
    
    if not os.path.exists(metadata.mp4_path):
        raise HTTPException(status_code=404, detail=f"Video file not found")
    
    # Validate frame index
    if frame_idx < 0:
        raise HTTPException(status_code=400, detail=f"frame_idx must be >= 0, got {frame_idx}")
    if frame_idx >= metadata.frame_count:
        raise HTTPException(
            status_code=400, 
            detail=f"frame_idx ({frame_idx}) exceeds frame_count ({metadata.frame_count})"
        )
    
    # Extract single frame using timestamp-based seeking
    import subprocess
    
    # Use ClipExtractor's FPS and first_frame_pts for consistency
    fps = clip_extractor.fps
    first_frame_pts = clip_extractor.first_frame_pts
    dt = 1.0 / fps
    
    # Calculate timestamp for this frame
    frame_time = first_frame_pts + frame_idx * dt
    
    logger.debug(
        f"Thumbnail frame {frame_idx} -> timestamp {frame_time:.6f}s "
        f"(fps={fps}, dt={dt:.6f}s)"
    )
    
    cmd = [
        'ffmpeg',
        '-loglevel', 'error',
        '-ss', f'{frame_time:.6f}',  # Seek to frame timestamp (BEFORE input)
        '-i', metadata.mp4_path,
        '-vframes', '1',              # Extract single frame
        '-f', 'image2',
        '-c:v', 'mjpeg',
        '-q:v', '2',                  # High quality JPEG
        'pipe:1'
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            check=True, 
            timeout=300  # Increased timeout for large videos
        )
        
        if len(result.stdout) == 0:
            logger.error(f"FFmpeg returned empty output for frame {frame_idx}")
            raise HTTPException(status_code=500, detail="Thumbnail extraction returned empty image")
        
        return StreamingResponse(
            iter([result.stdout]),
            media_type="image/jpeg"
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
        logger.error(f"Thumbnail extraction failed for frame {frame_idx}: {error_msg}")
        raise HTTPException(
            status_code=500, 
            detail=f"Thumbnail extraction failed: {error_msg[:200]}"
        )
    except subprocess.TimeoutExpired:
        logger.error(f"Thumbnail extraction timed out for frame {frame_idx}")
        raise HTTPException(status_code=500, detail="Thumbnail extraction timed out")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


# ============================================================================
# Static Files & WebUI
# ============================================================================

# Mount static files for WebUI
webui_path = Path(__file__).parent / "webui"
if webui_path.exists():
    app.mount("/static", StaticFiles(directory=str(webui_path)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main WebUI page."""
    index_path = webui_path / "index.html"
    if index_path.exists():
        return index_path.read_text()
    
    # Fallback minimal UI
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Semantic Video Search</title>
        <style>
            body { font-family: system-ui; max-width: 1200px; margin: 0 auto; padding: 20px; }
            .search-box { display: flex; gap: 10px; margin-bottom: 20px; }
            .search-box input { flex: 1; padding: 10px; font-size: 16px; }
            .search-box button { padding: 10px 20px; }
            .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
            .result-card { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
            .result-card video { width: 100%; }
            .result-card .metadata { padding: 10px; }
        </style>
    </head>
    <body>
        <h1>Semantic Video Search</h1>
        <div class="search-box">
            <input type="text" id="query" placeholder="e.g., ego vehicle stopped at red light">
            <button onclick="search()">Search</button>
        </div>
        <div id="results" class="results"></div>
        
        <script>
            async function search() {
                const query = document.getElementById('query').value;
                const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
                const data = await response.json();
                displayResults(data.results);
            }
            
            function displayResults(results) {
                const container = document.getElementById('results');
                container.innerHTML = results.map(r => `
                    <div class="result-card">
                        <video src="${r.clip_url}" controls preload="metadata"></video>
                        <div class="metadata">
                            <div><strong>${r.video_id}</strong></div>
                            <div>Time: ${r.start_time.toFixed(1)}s - ${r.end_time.toFixed(1)}s</div>
                            <div>Score: ${r.score.toFixed(3)}</div>
                            ${r.merged_count > 1 ? `<div>Merged ${r.merged_count} windows</div>` : ''}
                        </div>
                    </div>
                `).join('');
            }
        </script>
    </body>
    </html>
    """


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
