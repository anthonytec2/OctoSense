"""
Semantic Video Search - Metadata Index Module

Video metadata schema including H5 paths and timestamp information.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import h5py
import hdf5plugin  # Required for Blosc2 compressed H5 files

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """
    Metadata for a single video file with H5 timestamp information.
    
    This is the core schema for the metadata index (video_metadata.json).
    Each entry represents one searchable video unit.
    """
    # Identity
    video_id: str                          # Unique ID: "{session}/{bag_name}"
    
    # File paths
    mp4_path: str                          # Path to MP4 video file
    h5_path: str                           # Path to corresponding H5 file
    
    # Frame/timing info
    frame_count: int                       # Total frames (len of timestamp array)
    duration_sec: float                    # Total duration in seconds
    timestamp_range: Tuple[float, float]   # (t_start, t_end) in seconds
    
    # Optional metadata
    session_id: Optional[str] = None       # Session identifier (e.g., "sess8")
    bag_name: Optional[str] = None         # Bag name (e.g., "rosbag2_2026_01_08")
    
    # Camera intrinsics (cached from H5)
    resolution: Optional[Tuple[int, int]] = None  # (width, height)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert tuples to lists for JSON compatibility
        d['timestamp_range'] = list(d['timestamp_range'])
        if d['resolution']:
            d['resolution'] = list(d['resolution'])
        return d


class MetadataIndex:
    """
    In-memory index of video metadata, built during the scan step of ingest-all
    and written to video_metadata.json. That JSON is the work-list consumed by the
    captioning/embedding phases; it is not used by serving (search loads the FAISS
    index's own metadata.json).
    """
    
    def __init__(self):
        self._index: Dict[str, VideoMetadata] = {}
        self._by_session: Dict[str, List[str]] = {}
    
    def add(self, metadata: VideoMetadata) -> None:
        """Add a video to the index."""
        self._index[metadata.video_id] = metadata
        
        # Index by session for efficient filtering
        if metadata.session_id:
            if metadata.session_id not in self._by_session:
                self._by_session[metadata.session_id] = []
            self._by_session[metadata.session_id].append(metadata.video_id)
    
    def get(self, video_id: str) -> Optional[VideoMetadata]:
        """Get metadata for a video."""
        return self._index.get(video_id)
    
    def get_by_session(self, session_id: str) -> List[VideoMetadata]:
        """Get all videos in a session."""
        video_ids = self._by_session.get(session_id, [])
        return [self._index[vid] for vid in video_ids if vid in self._index]
    
    def all(self) -> List[VideoMetadata]:
        """Get all video metadata."""
        return list(self._index.values())
    
    def count(self) -> int:
        """Number of videos in index."""
        return len(self._index)
    
    def sessions(self) -> List[str]:
        """List all session IDs."""
        return list(self._by_session.keys())
    
    def save_json(self, path: str) -> None:
        """Save index to JSON file."""
        data = [m.to_dict() for m in self._index.values()]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} entries to {path}")


def extract_metadata_from_h5(
    h5_path: str,
    mp4_path: str,
    video_id: Optional[str] = None,
    timestamp_dataset: str = 'img/left/t'
) -> VideoMetadata:
    """
    Extract video metadata from H5 file.
    
    Args:
        h5_path: Path to H5 file
        mp4_path: Path to corresponding MP4 file
        video_id: Optional video ID (auto-generated from paths if not provided)
        timestamp_dataset: Path to timestamp dataset in H5
    
    Returns:
        VideoMetadata with all fields populated
    """
    h5_path = str(h5_path)
    mp4_path = str(mp4_path)
    
    # Auto-generate video_id from path if not provided
    if video_id is None:
        # Extract session/bag_name from path
        # e.g., /data/.../sess8/rosbag2_2026_01_08/rosbag2_2026_01_08.h5
        parts = Path(h5_path).parts
        if len(parts) >= 3:
            # Assume structure: .../session/bag_name/bag_name.h5
            session_id = parts[-3] if parts[-3].startswith('sess') else parts[-2]
            bag_name = parts[-2]
            video_id = f"{session_id}/{bag_name}"
        else:
            video_id = Path(h5_path).stem
    
    # Parse session_id and bag_name from video_id
    parts = video_id.split('/')
    session_id = parts[0] if len(parts) >= 2 else None
    bag_name = parts[-1] if parts else video_id
    
    # Load timestamp info from H5
    with h5py.File(h5_path, 'r') as h5:
        if timestamp_dataset not in h5:
            raise KeyError(f"Dataset '{timestamp_dataset}' not found in {h5_path}")
        
        timestamps = h5[timestamp_dataset][:]
        frame_count = len(timestamps)
        
        if frame_count == 0:
            raise ValueError(f"Empty timestamp array in {h5_path}")
        
        t_start = float(timestamps[0])
        t_end = float(timestamps[-1])
        duration = t_end - t_start
        
        # Try to get resolution
        resolution = None
        res_path = timestamp_dataset.rsplit('/t', 1)[0] + '/resolution'
        if res_path in h5:
            res = h5[res_path][:]
            resolution = (int(res[0]), int(res[1]))
    
    return VideoMetadata(
        video_id=video_id,
        mp4_path=mp4_path,
        h5_path=h5_path,
        frame_count=frame_count,
        duration_sec=duration,
        timestamp_range=(t_start, t_end),
        session_id=session_id,
        bag_name=bag_name,
        resolution=resolution
    )


def scan_data_directory(
    base_path: str,
    pattern: str = "sess*/rosbag2_*/",
    mp4_name: str = "img_left.mp4",
    h5_suffix: str = ".h5",
    sessions: Optional[List[str]] = None
) -> MetadataIndex:
    """
    Scan a data directory and build metadata index.
    
    Args:
        base_path: Root directory to scan (e.g., /data/rosbags/processed/car)
        pattern: Glob pattern for video directories
        mp4_name: Name of MP4 file in each directory
        h5_suffix: Suffix of H5 file
        sessions: Optional list of session names to include (e.g., ['sess7', 'sess8', 'sess9', 'sess10'])
    
    Returns:
        MetadataIndex with all discovered videos
    """
    base = Path(base_path)
    index = MetadataIndex()
    
    # If sessions filter specified, build session set for filtering
    session_set = None
    if sessions:
        session_set = set(sessions)
        logger.info(f"Filtering to sessions: {session_set}")
    
    logger.info(f"Scanning {base_path} with pattern {pattern}")
    
    # Find all matching directories
    for video_dir in base.glob(pattern):
        if not video_dir.is_dir():
            continue
        
        # Filter by session if specified
        if session_set:
            # Extract session name from parent directory (e.g., 'sess7' from '/path/to/sess7/rosbag2_...')
            session_name = video_dir.parent.name
            if session_name not in session_set:
                continue
        
        mp4_path = video_dir / mp4_name
        if not mp4_path.exists():
            continue
        
        # Find the per-bag H5 (named after the bag dir, e.g. <bag>.h5). The dir may also
        # hold sidecars (<bag>_events.h5, semantic.h5, captions_temp.h5, *_sparse_gt.h5)
        # that lack img/left/t, so prefer the bag-named file rather than glob order.
        h5_files = list(video_dir.glob(f"*{h5_suffix}"))
        if not h5_files:
            logger.warning(f"No H5 file found in {video_dir}")
            continue
        named = video_dir / f"{video_dir.name}{h5_suffix}"
        h5_path = named if named.exists() else h5_files[0]
        
        try:
            metadata = extract_metadata_from_h5(str(h5_path), str(mp4_path))
            index.add(metadata)
            logger.debug(f"Added {metadata.video_id}: {metadata.frame_count} frames, {metadata.duration_sec:.1f}s")
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {h5_path}: {e}")
    
    logger.info(f"Indexed {index.count()} videos from {len(index.sessions())} sessions")
    return index


