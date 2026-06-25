"""
Semantic Video Search - Clip Extractor Module

FFmpeg-based video clip extraction for remote viewing using timestamp-based seeking.
Streams clips directly

- Calculates video timestamps from frame indices: clip_start_time = first_frame_pts + frame_idx * dt
- Uses FFmpeg's -ss (input seeking) for efficient seeking to nearest keyframe
- Uses -t for duration instead of per-frame select filters
- Avoids CPU-intensive frame filtering overhead

"""
import subprocess
from typing import Iterator
import logging

logger = logging.getLogger(__name__)


class ClipExtractor:
    """
    Extract video clips using FFmpeg for remote streaming.
    """
    
    def __init__(
        self,
        ffmpeg_path: str = 'ffmpeg',
        fps: float = 100.0,
        first_frame_pts: float = 0.0
    ):
        """
        Args:
            ffmpeg_path: Path to ffmpeg binary
            fps: Video frame rate — maps frame indices to seek times (frame_idx / fps).
            first_frame_pts: PTS of first frame in video (seconds).
        """
        self.ffmpeg_path = ffmpeg_path
        self.fps = fps
        self.first_frame_pts = first_frame_pts
        self.dt = 1.0 / fps  # Frame interval in seconds

    def extract_clip_streaming(
        self,
        video_path: str,
        start_frame_idx: int,
        end_frame_idx: int,
        chunk_size: int = 4096
    ) -> Iterator[bytes]:
        """
        Extract a video clip and yield it in chunks for a streaming HTTP response.

        Args:
            video_path: Path to source video file
            start_frame_idx: Start frame index (0-based, inclusive)
            end_frame_idx: End frame index (0-based, inclusive)
            chunk_size: Size of each chunk to yield

        Yields:
            Chunks of video data
        """
        cmd = self._build_ffmpeg_cmd_by_frames(video_path, start_frame_idx, end_frame_idx)
        
        logger.debug(f"Streaming FFmpeg: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        try:
            while True:
                chunk = process.stdout.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            process.stdout.close()
            # Read stderr before closing to capture error messages
            stderr_output = process.stderr.read()
            process.stderr.close()
            process.wait()
            
            if process.returncode != 0:
                error_msg = stderr_output.decode('utf-8', errors='replace') if stderr_output else "No error message available"
                logger.warning(
                    f"FFmpeg exited with code {process.returncode}. "
                    f"Error: {error_msg[:500]}"  # Log first 500 chars of error
                )
    
    
    def _build_ffmpeg_cmd_by_frames(
        self,
        video_path: str,
        start_frame_idx: int,
        end_frame_idx: int,
    ) -> list:
        """
        Build the FFmpeg command to extract [start_frame_idx, end_frame_idx] as a clip.

        clip_start_time = first_frame_pts + start_frame_idx * dt and
        duration = (num_frames) * dt. -ss before -i is a fast input seek to the nearest
        """
        if start_frame_idx < 0:
            raise ValueError(f"start_frame_idx must be >= 0, got {start_frame_idx}")
        if end_frame_idx < start_frame_idx:
            raise ValueError(f"end_frame_idx ({end_frame_idx}) must be >= start_frame_idx ({start_frame_idx})")

        clip_start_time = self.first_frame_pts + start_frame_idx * self.dt
        num_frames = end_frame_idx - start_frame_idx + 1
        clip_duration = num_frames * self.dt

        logger.debug(
            f"Frame range [{start_frame_idx}, {end_frame_idx}] -> "
            f"Time range [{clip_start_time:.6f}s, +{clip_duration:.6f}s] (fps={self.fps})"
        )

        return [
            self.ffmpeg_path, '-y', '-loglevel', 'error',
            '-ss', f'{clip_start_time:.6f}',
            '-i', video_path,
            '-t', f'{clip_duration:.6f}',
            '-c', 'copy',                              # stream-copy source H.265, no transcode
            '-an',                                     # drop audio
            '-movflags', 'frag_keyframe+empty_moov',   # fragmented mp4, streamable over a pipe
            '-f', 'mp4', 'pipe:1',
        ]


def get_clip_url(
    video_id: str,
    start_frame_idx: int,
    end_frame_idx: int,
    base_url: str = '/api/clip'
) -> str:
    """
    Generate clip URL for API using frame indices.
    
    Args:
        video_id: Video identifier
        start_frame_idx: Start frame index (0-based, inclusive)
        end_frame_idx: End frame index (0-based, inclusive)
        base_url: Base URL for clip API
    
    Returns:
        Full clip URL with query parameters
    """
    import urllib.parse
    encoded_id = urllib.parse.quote(video_id, safe='')
    return f"{base_url}/{encoded_id}?start_frame_idx={start_frame_idx}&end_frame_idx={end_frame_idx}"
