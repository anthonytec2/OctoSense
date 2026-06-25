"""
Semantic Video Search - Ingestion Module

Timestamp-based window extraction from MP4 videos using H5 timestamp arrays.
"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import logging
import h5py
import hdf5plugin  # Required for Blosc2 compressed H5 files

try:
    from torchcodec.decoders import VideoDecoder
except ImportError:
    VideoDecoder = None

logger = logging.getLogger(__name__)


@dataclass
class SemanticWindow:
    """A searchable unit representing a time window in a video."""
    window_id: int
    video_id: str
    frame_idx: int
    timestamp: float
    embedding: Optional[np.ndarray] = None  # filled in during the embedding phase
    caption: Optional[str] = None           # filled in during the captioning phase
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            'window_id': self.window_id,
            'video_id': self.video_id,
            'frame_idx': self.frame_idx,
            'timestamp': self.timestamp,
            'caption': self.caption,
        }
        # Structured telemetry (speed_mps/turn_deg/dist_m/is_night) for hybrid search,
        # flattened into the metadata so search_api can filter on it.
        telem = getattr(self, 'telem', None)
        if telem:
            d.update(telem)
        return d




def load_timestamps(h5_path: str, dataset_path: str = 'img/left/t') -> np.ndarray:
    """
    Load timestamps from H5 file.
    
    Args:
        h5_path: Path to H5 file
        dataset_path: Path to timestamp dataset within H5 file
    
    Returns:
        Timestamp array (N,) float64, in seconds
    """
    with h5py.File(h5_path, 'r') as h5:
        if dataset_path not in h5:
            raise KeyError(f"Dataset '{dataset_path}' not found in {h5_path}")
        timestamps = h5[dataset_path][:]
    return timestamps.astype(np.float64)


def find_window_frame_indices(
    timestamps: np.ndarray,
    window_size_sec: float = 1.0
) -> np.ndarray:
    """
    Find frame indices at timestamp boundaries.
    
    Uses np.searchsorted to map K-second timestamp boundaries to frame indices.
    This is the core algorithm for timestamp-based windowing.
    
    Args:
        timestamps: Array of timestamps (N,) float64, in seconds
        window_size_sec: Window size in seconds (default 1.0)
    
    Returns:
        Array of frame indices corresponding to window start times
    """
    if len(timestamps) == 0:
        return np.array([], dtype=np.int64)
    
    t_start = timestamps[0]
    t_end = timestamps[-1]
    
    # Create window start times at regular intervals
    window_starts = np.arange(t_start, t_end, window_size_sec)
    
    # Use searchsorted to find frame indices for each timestamp boundary
    # This finds the index where each window_start would be inserted to maintain order
    frame_indices = np.searchsorted(timestamps, window_starts)
    
    # Clip to valid range [0, len(timestamps) - 1]
    frame_indices = np.clip(frame_indices, 0, len(timestamps) - 1)
    
    return frame_indices


def extract_semantic_windows(
    h5_path: str,
    video_path: str,
    video_id: str,
    window_size_sec: float = 1.0,
    device: str = 'cuda',
    timestamp_dataset: str = 'img/left/t'
) -> List[SemanticWindow]:
    """
    Extract semantic windows from a video using H5 timestamps.
    
    Frame indices are determined by timestamp boundaries, not sequential frame numbers.
    This is the main ingestion function.
    
    Args:
        h5_path: Path to H5 file containing timestamps
        video_path: Path to MP4 video file
        video_id: Unique identifier for this video (e.g., "sess8/rosbag2_...")
        window_size_sec: Window size in seconds (default 1.0)
        device: Device for video decoding ('cuda' or 'cpu')
        timestamp_dataset: Path to timestamp dataset in H5 file
    
    Returns:
        List of SemanticWindow objects ready for embedding
    """
    if VideoDecoder is None:
        raise ImportError("torchcodec is required for video decoding")
    
    # Load timestamps from H5
    timestamps = load_timestamps(h5_path, timestamp_dataset)
    logger.info(f"Loaded {len(timestamps)} timestamps from {h5_path}")
    
    # Find frame indices at window boundaries
    frame_indices = find_window_frame_indices(timestamps, window_size_sec)
    logger.info(f"Found {len(frame_indices)} windows at {window_size_sec}s intervals")
    
    # Open video decoder
    decoder = VideoDecoder(video_path, device=device)
    
    # Verify alignment
    if len(decoder) != len(timestamps):
        logger.warning(
            f"Frame count mismatch: VideoDecoder has {len(decoder)} frames, "
            f"H5 has {len(timestamps)} timestamps"
        )
    
    windows = []
    for window_id, frame_idx in enumerate(frame_indices):
        # Ensure frame_idx is within decoder range
        if frame_idx >= len(decoder):
            logger.warning(f"Frame index {frame_idx} exceeds decoder length {len(decoder)}")
            continue
        
        # Get timestamp for this frame
        timestamp = float(timestamps[frame_idx])
        
        window = SemanticWindow(
            window_id=window_id,
            video_id=video_id,
            frame_idx=int(frame_idx),
            timestamp=timestamp,
        )
        windows.append(window)
    
    return windows








