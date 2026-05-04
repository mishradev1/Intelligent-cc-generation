"""Frame extraction utility for extracting video frames at specific timestamps.

This module provides the FrameExtractor class that uses OpenCV to extract
frames from video files, particularly around detected audio event timestamps.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from config.settings import REACTION_FRAME_COUNT, REACTION_TIME_WINDOW

logger = logging.getLogger(__name__)


class FrameExtractionError(Exception):
    """Raised when frame extraction from a video file fails."""

    pass


class FrameExtractor:
    """Extracts video frames at specific timestamps using OpenCV.

    This extractor retrieves frames around specified timestamps,
    typically centered at detected audio event times for visual
    reaction analysis.

    Attributes:
        frame_count: Number of frames to extract around each timestamp.
        time_window: Time window (seconds) before and after the timestamp.

    Example:
        >>> extractor = FrameExtractor(frame_count=10, time_window=1.5)
        >>> frames = extractor.extract_around_timestamp("video.mp4", 5.0)
        >>> print(f"Extracted {len(frames)} frames")
    """

    def __init__(
        self,
        frame_count: int = REACTION_FRAME_COUNT,
        time_window: float = REACTION_TIME_WINDOW,
    ):
        """Initialize the FrameExtractor.

        Args:
            frame_count: Number of frames to extract around each timestamp.
            time_window: Time window (seconds) before and after timestamp.
        """
        self.frame_count = frame_count
        self.time_window = time_window
        self._cv2 = None

        logger.info(
            "FrameExtractor initialized (frame_count=%d, time_window=%.1fs)",
            self.frame_count,
            self.time_window,
        )

    def _get_cv2(self):
        """Lazy import of cv2 to avoid import-time dependency."""
        if self._cv2 is None:
            try:
                import cv2

                self._cv2 = cv2
            except ImportError as e:
                raise ImportError(
                    "OpenCV (cv2) is required for frame extraction. "
                    "Install it with: pip install opencv-python"
                ) from e
        return self._cv2

    def get_video_info(self, video_path: str) -> dict:
        """Get information about a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary with video info: fps, frame_count, duration,
            width, height.

        Raises:
            FileNotFoundError: If the video file does not exist.
            FrameExtractionError: If the video cannot be opened.
        """
        cv2 = self._get_cv2()
        path = Path(video_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FrameExtractionError(f"Failed to open video: {path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            return {
                "fps": fps,
                "frame_count": total_frames,
                "duration": round(duration, 3),
                "width": width,
                "height": height,
            }
        finally:
            cap.release()

    def extract_at_timestamp(
        self, video_path: str, timestamp: float
    ) -> Optional[np.ndarray]:
        """Extract a single frame at a specific timestamp.

        Args:
            video_path: Path to the video file.
            timestamp: Time in seconds to extract the frame at.

        Returns:
            Frame as a numpy array (BGR format), or None if extraction fails.
        """
        cv2 = self._get_cv2()
        path = Path(video_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FrameExtractionError(f"Failed to open video: {path}")

        try:
            # Seek to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()

            if not ret:
                logger.warning(
                    "Failed to read frame at timestamp %.3fs from %s",
                    timestamp,
                    path,
                )
                return None

            return frame
        finally:
            cap.release()

    def extract_around_timestamp(
        self,
        video_path: str,
        timestamp: float,
        frame_count: Optional[int] = None,
        time_window: Optional[float] = None,
    ) -> List[Tuple[float, np.ndarray]]:
        """Extract multiple frames around a specific timestamp.

        Extracts frames evenly spaced within a time window centered
        on the given timestamp.

        Args:
            video_path: Path to the video file.
            timestamp: Center timestamp in seconds.
            frame_count: Number of frames to extract. Defaults to
                self.frame_count.
            time_window: Time window in seconds. Defaults to
                self.time_window.

        Returns:
            List of (timestamp, frame) tuples. Frames are numpy arrays
            in BGR format.
        """
        cv2 = self._get_cv2()
        path = Path(video_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        n_frames = frame_count or self.frame_count
        window = time_window or self.time_window

        # Calculate evenly spaced timestamps within the window
        start_time = max(0, timestamp - window)
        end_time = timestamp + window

        if n_frames <= 1:
            target_times = [timestamp]
        else:
            step = (end_time - start_time) / (n_frames - 1)
            target_times = [start_time + i * step for i in range(n_frames)]

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FrameExtractionError(f"Failed to open video: {path}")

        frames = []
        try:
            for t in target_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if ret:
                    frames.append((round(t, 3), frame))
                else:
                    logger.debug(
                        "Could not read frame at %.3fs", t
                    )
        finally:
            cap.release()

        logger.info(
            "Extracted %d/%d frames around timestamp %.3fs from %s",
            len(frames),
            n_frames,
            timestamp,
            path.name,
        )

        return frames

    def extract_at_timestamps(
        self, video_path: str, timestamps: List[float]
    ) -> List[Tuple[float, np.ndarray]]:
        """Extract single frames at multiple timestamps.

        Args:
            video_path: Path to the video file.
            timestamps: List of timestamps in seconds.

        Returns:
            List of (timestamp, frame) tuples for successfully extracted
            frames.
        """
        cv2 = self._get_cv2()
        path = Path(video_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FrameExtractionError(f"Failed to open video: {path}")

        frames = []
        try:
            for t in sorted(timestamps):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if ret:
                    frames.append((round(t, 3), frame))
        finally:
            cap.release()

        logger.info(
            "Extracted %d/%d frames at specified timestamps",
            len(frames),
            len(timestamps),
        )

        return frames
