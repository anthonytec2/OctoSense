"""
Semantic Video Search - Window Merger Module

Merge adjacent/overlapping time windows from the same sequence.
When multiple matches occur close together in time (within max_gap_sec),
they are merged into a single result with extended time range.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)
VIDEO_FPS = 100.0


@dataclass
class MergedWindow:
    """A merged window representing multiple adjacent windows."""
    video_id: str
    sequence_id: str
    start_time: float
    end_time: float
    start_frame_idx: int
    end_frame_idx: int
    duration: float
    score: float
    merged_count: int
    mp4_path: Optional[str] = None
    h5_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)




def merge_adjacent_windows(
    windows: List[Dict[str, Any]],
    max_gap_sec: float = 10.0,
    window_duration: float = 1.0
) -> List[MergedWindow]:
    """
    Merge adjacent/overlapping windows from the same sequence.
    
    Algorithm:
    1. Sort windows by timestamp
    2. Iterate through, grouping windows that are within max_gap_sec of each other
    3. For each group, create a merged window with:
       - start_time from first window
       - end_time from last window + window_duration
       - score = max(all scores in group)
       - merged_count = number of windows merged
    
    Args:
        windows: List of window dictionaries with keys:
            - video_id: str
            - sequence_id: str (optional, defaults to video_id)
            - timestamp: float
            - frame_idx: int
            - score: float
            - mp4_path: str (optional)
            - h5_path: str (optional)
        max_gap_sec: Maximum gap between windows to merge (default 2.0 seconds)
        window_duration: Duration of each individual window (default 1.0 seconds)
    
    Returns:
        List of MergedWindow objects
    """
    if not windows:
        return []

    # Sort by timestamp
    windows_sorted = sorted(windows, key=lambda w: w['timestamp'])
    
    merged_results: List[MergedWindow] = []
    current_group: List[Dict[str, Any]] = [windows_sorted[0]]
    
    for w in windows_sorted[1:]:
        last_window = current_group[-1]
        last_start = last_window['timestamp']
        last_end = last_start + window_duration  # End time of last window
        current_start = w['timestamp']
        
        # Check if this window is adjacent or overlapping (gap between end of last and start of current)
        gap = current_start - last_end
        if gap <= max_gap_sec:
            current_group.append(w)
        else:
            # Emit the current group as a merged window
            merged = _merge_group(current_group, window_duration)
            merged_results.append(merged)
            current_group = [w]
    
    # Handle the last group
    if current_group:
        merged = _merge_group(current_group, window_duration)
        merged_results.append(merged)
    
    logger.info(
        f"Merged {len(windows)} windows into {len(merged_results)} results "
        f"(max_gap={max_gap_sec}s)"
    )
    
    return merged_results


def _merge_group(
    group: List[Dict[str, Any]],
    window_duration: float
) -> MergedWindow:
    """
    Create a MergedWindow from a group of adjacent windows.
    
    Args:
        group: List of window dictionaries (must be non-empty and sorted by timestamp)
        window_duration: Duration of each individual window
    
    Returns:
        MergedWindow representing the merged group
    """
    first = group[0]
    last = group[-1]
    
    # Get sequence_id, defaulting to video_id if not present
    sequence_id = first.get('sequence_id') or first['video_id']
    
    # Calculate time range
    start_time = first['timestamp']
    end_time = last['timestamp'] + window_duration
    duration = end_time - start_time


    start_frame_idx = int(first['frame_idx'])
    end_frame_idx = int(last['frame_idx'] + round(window_duration * VIDEO_FPS))

    # Best score in the group
    best_score = max(w['score'] for w in group)
    
    return MergedWindow(
        video_id=first['video_id'],
        sequence_id=sequence_id,
        start_time=start_time,
        end_time=end_time,
        start_frame_idx=start_frame_idx,
        end_frame_idx=end_frame_idx,
        duration=duration,
        score=best_score,
        merged_count=len(group),
        mp4_path=first.get('mp4_path'),
        h5_path=first.get('h5_path')
    )


def group_by_sequence(
    windows: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group windows by sequence ID.
    
    Args:
        windows: List of window dictionaries
    
    Returns:
        Dictionary mapping sequence_id to list of windows
    """
    by_sequence: Dict[str, List[Dict[str, Any]]] = {}
    
    for w in windows:
        seq_id = w.get('sequence_id') or w['video_id']
        if seq_id not in by_sequence:
            by_sequence[seq_id] = []
        by_sequence[seq_id].append(w)
    
    return by_sequence


def merge_all_by_sequence(
    windows: List[Dict[str, Any]],
    max_gap_sec: float = 10.0,
    window_duration: float = 1.0
) -> List[MergedWindow]:
    """
    Group windows by sequence and merge adjacent windows within each group.
    
    This is the main entry point for window merging across multiple sequences.
    
    Args:
        windows: List of window dictionaries
        max_gap_sec: Maximum gap between windows to merge
        window_duration: Duration of each individual window
    
    Returns:
        List of MergedWindow objects from all sequences
    """
    by_sequence = group_by_sequence(windows)
    
    all_merged: List[MergedWindow] = []
    
    for seq_id, seq_windows in by_sequence.items():
        merged = merge_adjacent_windows(seq_windows, max_gap_sec, window_duration)
        all_merged.extend(merged)
    
    # Sort by score (descending)
    all_merged.sort(key=lambda m: m.score, reverse=True)
    
    return all_merged
