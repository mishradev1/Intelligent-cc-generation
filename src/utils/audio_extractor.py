"""Audio extraction utility for extracting audio tracks from video files.

This module provides the AudioExtractor class that handles extracting
audio from various video formats and saving it as WAV files suitable
for downstream audio analysis.
"""

import os
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from config.settings import AUDIO_SAMPLE_RATE, AUDIO_FORMAT, AUDIO_OUTPUT_DIR

logger = logging.getLogger(__name__)


class AudioExtractionError(Exception):
    """Raised when audio extraction from a video file fails."""

    pass


class AudioExtractor:
    """Extracts audio tracks from video files using FFmpeg.

    This extractor converts video files to mono WAV audio at a configurable
    sample rate, suitable for input to audio analysis models like YAMNet.

    Attributes:
        sample_rate: Target audio sample rate in Hz.
        output_dir: Directory where extracted audio files will be saved.
        audio_format: Output audio format (default: wav).

    Example:
        >>> extractor = AudioExtractor(sample_rate=16000)
        >>> audio_path = extractor.extract("input_video.mp4")
        >>> print(f"Audio saved to: {audio_path}")
    """

    SUPPORTED_VIDEO_FORMATS = {
        ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    }

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        output_dir: str = AUDIO_OUTPUT_DIR,
        audio_format: str = AUDIO_FORMAT,
    ):
        """Initialize the AudioExtractor.

        Args:
            sample_rate: Target audio sample rate in Hz. Defaults to 16000.
            output_dir: Directory to save extracted audio files.
            audio_format: Output audio file format. Defaults to 'wav'.

        Raises:
            RuntimeError: If FFmpeg is not found on the system PATH.
        """
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.audio_format = audio_format

        # Verify FFmpeg is available
        self._ffmpeg_path = self._find_ffmpeg()
        if self._ffmpeg_path is None:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg and ensure it is "
                "on your system PATH. Visit https://ffmpeg.org/download.html"
            )

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            "AudioExtractor initialized (sample_rate=%d, format=%s, output=%s)",
            self.sample_rate,
            self.audio_format,
            self.output_dir,
        )

    @staticmethod
    def _find_ffmpeg() -> Optional[str]:
        """Locate the FFmpeg executable on the system.

        Returns:
            Path to the FFmpeg executable, or None if not found.
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            logger.debug("Found FFmpeg at: %s", ffmpeg_path)
        return ffmpeg_path

    def _validate_input(self, video_path: str) -> Path:
        """Validate the input video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Resolved Path object for the video file.

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the file format is not supported.
        """
        path = Path(video_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_VIDEO_FORMATS:
            raise ValueError(
                f"Unsupported video format: '{suffix}'. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_VIDEO_FORMATS))}"
            )

        return path

    def _build_output_path(
        self, video_path: Path, output_path: Optional[str] = None
    ) -> Path:
        """Build the output path for the extracted audio file.

        Args:
            video_path: Path to the source video file.
            output_path: Optional custom output path. If None, generates
                one in the output directory.

        Returns:
            Path for the output audio file.
        """
        if output_path:
            out = Path(output_path).resolve()
            os.makedirs(out.parent, exist_ok=True)
            return out

        filename = f"{video_path.stem}.{self.audio_format}"
        return Path(self.output_dir) / filename

    def extract(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """Extract audio from a video file.

        Extracts the audio track from the given video file and saves it
        as a mono WAV file at the configured sample rate.

        Args:
            video_path: Path to the input video file.
            output_path: Optional custom output file path. If not provided,
                the audio will be saved in the configured output directory
                with the same stem name as the video.
            overwrite: If True, overwrite existing output file.
                Defaults to False.

        Returns:
            Absolute path to the extracted audio file.

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the video format is not supported.
            AudioExtractionError: If FFmpeg fails to extract audio.
            FileExistsError: If output file exists and overwrite is False.
        """
        video = self._validate_input(video_path)
        output = self._build_output_path(video, output_path)

        if output.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output}. "
                "Use overwrite=True to replace it."
            )

        logger.info("Extracting audio from: %s", video)
        logger.info("Output: %s", output)

        cmd = [
            self._ffmpeg_path,
            "-i", str(video),       # Input file
            "-vn",                   # Disable video
            "-acodec", "pcm_s16le",  # 16-bit PCM encoding
            "-ar", str(self.sample_rate),  # Sample rate
            "-ac", "1",              # Mono channel
            "-y" if overwrite else "-n",  # Overwrite flag
            str(output),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip().split("\n")[-1]
                raise AudioExtractionError(
                    f"FFmpeg failed with return code {result.returncode}: "
                    f"{error_msg}"
                )

        except subprocess.TimeoutExpired:
            raise AudioExtractionError(
                f"Audio extraction timed out after 300 seconds for: {video}"
            )
        except FileNotFoundError:
            raise AudioExtractionError(
                "FFmpeg executable not found. It may have been removed "
                "after initialization."
            )

        if not output.exists():
            raise AudioExtractionError(
                f"Audio extraction completed but output file not found: {output}"
            )

        file_size = output.stat().st_size
        logger.info(
            "Audio extraction successful: %s (%.2f MB)",
            output,
            file_size / (1024 * 1024),
        )

        return str(output)

    def get_audio_info(self, audio_path: str) -> dict:
        """Get information about an audio file using FFprobe.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Dictionary with audio information including duration,
            sample_rate, channels, and codec.
        """
        path = Path(audio_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path is None:
            raise RuntimeError("FFprobe not found on system PATH.")

        cmd = [
            ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "a:0",
            str(path),
        ]

        try:
            import json

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]

            return {
                "duration": float(stream.get("duration", 0)),
                "sample_rate": int(stream.get("sample_rate", 0)),
                "channels": int(stream.get("channels", 0)),
                "codec": stream.get("codec_name", "unknown"),
                "bit_rate": int(stream.get("bit_rate", 0)),
            }
        except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError) as e:
            logger.error("Failed to get audio info: %s", e)
            return {}
