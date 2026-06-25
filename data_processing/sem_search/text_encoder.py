"""
Text query/document encoder (Qwen3-Embedding-8B).
"""
import os
import logging
import numpy as np
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

TEXT_EMBED_ID = 'Qwen/Qwen3-Embedding-8B'

EMBEDDING_DIM = 4096

QUERY_TASK_INSTRUCTION = (
    "Given a driving-scene search query, retrieve dashcam captions describing matching scenes"
)

_pipeline_cache = {}

def _model_cache_dir() -> str:
    """Resolve the HF model cache dir, falling back to HF's default when HF_HOME
    is unset (otherwise os.path.join(None, ...) would crash)."""
    hf_home = os.environ.get('HF_HOME') or os.path.join(
        os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'huggingface')
    return os.path.join(hf_home, 'hub')

def load_text_embedder_only(model_name: str = 'gemma4-captioner', device: str = 'cuda'):
    """Load only text embedder for embedding generation (Phase 2)"""
    global _pipeline_cache
    
    # Check cache first
    cache_key = f'text_embedder_{model_name}_{device}'
    if cache_key in _pipeline_cache:
        logger.debug(f"Using cached text embedder: {model_name}")
        return _pipeline_cache[cache_key]
    
    
    logger.info(f"Loading text embedder only: {TEXT_EMBED_ID}")
    text_embedder = SentenceTransformer(
        TEXT_EMBED_ID,
        trust_remote_code=True,
        cache_folder=_model_cache_dir(),
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
        device=device
    )
    logger.info(f"✓ Text embedder loaded on {device}")
    
    result = {
        'text_embedder': text_embedder,
        'embedding_dim': EMBEDDING_DIM
    }
    
    # Cache the result
    _pipeline_cache[cache_key] = result
    
    return result

def generate_embeddings_from_captions(captions: List[str], text_embedder_model: Dict) -> np.ndarray:
    """Generate embeddings from captions (Phase 2)"""
    text_embedder = text_embedder_model['text_embedder']
    
    embeddings = text_embedder.encode(
        captions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64
    )
    
    return embeddings.astype(np.float32)

def encode_text_query(query: str, model_name: str = 'gemma4-captioner', device: str = None) -> np.ndarray:
    """Encode text query to embedding vector (API compatible).

    Prepends the Qwen3-Embedding query instruction (`Instruct: {task}\\nQuery: {q}`)
    """
    if device is None:
        device = os.environ.get('SEMSEARCH_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
    text_model = load_text_embedder_only(model_name, device)
    q = f"Instruct: {QUERY_TASK_INSTRUCTION}\nQuery: {query}"
    vec = text_model['text_embedder'].encode([q], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)
