"""
Semantic Video Search - Diversity Reranking Module

Cross-sequence diversity prioritization: prefer results from different sequences (H5 files)
over multiple matches from the same sequence. Uses weighted scoring to penalize same-sequence
results while still allowing 2-3 results per sequence if scores are significantly higher.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with score and metadata."""
    score: float
    video_id: str
    sequence_id: str
    frame_idx: int
    timestamp: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    start_frame_idx: Optional[int] = None
    end_frame_idx: Optional[int] = None
    duration: Optional[float] = None
    merged_count: int = 1
    mp4_path: Optional[str] = None
    h5_path: Optional[str] = None
    clip_url: Optional[str] = None


def rerank_with_diversity(
    candidates: List[Tuple[float, str, Dict[str, Any]]],
    max_results: int = 20,
    diversity_weight: float = 0.3,
    max_per_sequence: int = 3
) -> List[SearchResult]:
    """
    Rerank search results for cross-sequence diversity:
      1. take the best window from each sequence (breadth), strongest sequences first;
      2. fill remaining slots with extra windows from already-seen sequences, each
         penalized by score * (1 - diversity_weight * pick_number) so a 2nd/3rd window
         from a sequence must score much higher to be included.
    Args:
        candidates: List of (score, video_id, window_data) tuples
        max_results: Maximum results to return
        diversity_weight: Penalty factor for same-sequence results (0-1)
        max_per_sequence: Maximum results allowed from a single sequence

    Returns:
        List of SearchResult objects with diversity-aware ranking
    """
    if not candidates:
        return []

    # Group by sequence (video_id IS the sequence id) and sort each by score desc.
    by_sequence: Dict[str, List[Tuple[float, str, Dict[str, Any]]]] = {}
    for score, video_id, window in candidates:
        by_sequence.setdefault(video_id, []).append((score, video_id, window))
    for windows in by_sequence.values():
        windows.sort(key=lambda x: x[0], reverse=True)

    selected: List[SearchResult] = []

    # Pass 1 — breadth: the best window from each sequence, strongest sequences first.
    for windows in sorted(by_sequence.values(), key=lambda w: w[0][0], reverse=True):
        if len(selected) >= max_results:
            break
        score, video_id, window = windows[0]
        selected.append(_create_search_result(score, video_id, window, video_id))

    # Pass 2 — depth: additional windows (rank 1..max_per_sequence-1), each penalized by
    extras = [
        (score * (1 - diversity_weight * (rank + 1)), score, video_id, window)
        for windows in by_sequence.values()
        for rank, (score, video_id, window) in enumerate(windows)
        if 1 <= rank < max_per_sequence
    ]
    for _adj, score, video_id, window in sorted(extras, key=lambda x: x[0], reverse=True):
        if len(selected) >= max_results:
            break
        selected.append(_create_search_result(score, video_id, window, video_id))

    logger.info(
        f"Diversity rerank: {len(candidates)} candidates -> {len(selected)} results "
        f"from {len({r.sequence_id for r in selected})} sequences"
    )
    return selected


def _create_search_result(
    score: float,
    video_id: str,
    window: Dict[str, Any],
    seq_id: str
) -> SearchResult:
    """Create a SearchResult from raw candidate data."""
    return SearchResult(
        score=score,
        video_id=video_id,
        sequence_id=seq_id,
        frame_idx=window.get('frame_idx', 0),
        timestamp=window.get('timestamp', 0.0),
        start_time=window.get('start_time') or window.get('timestamp'),
        end_time=window.get('end_time'),
        start_frame_idx=window.get('start_frame_idx') or window.get('frame_idx'),
        end_frame_idx=window.get('end_frame_idx'),
        duration=window.get('duration', 1.0),
        merged_count=window.get('merged_count', 1),
        mp4_path=window.get('mp4_path'),
        h5_path=window.get('h5_path'),
        clip_url=window.get('clip_url')
    )


def calculate_diversity_score(results: List[SearchResult]) -> float:
    """
    Calculate a diversity score for a set of results.
    
    Diversity score ranges from 0.0 (all from same sequence) to 1.0 (all unique sequences).
    
    Args:
        results: List of SearchResult objects
    
    Returns:
        Diversity score between 0.0 and 1.0
    """
    if not results:
        return 0.0
    
    unique_sequences = len(set(r.sequence_id for r in results))
    total_results = len(results)
    
    # Diversity = unique_sequences / total_results
    # 1.0 means every result is from a different sequence
    return unique_sequences / total_results
