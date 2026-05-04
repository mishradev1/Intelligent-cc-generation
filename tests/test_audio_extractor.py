"""Unit tests for the AudioExtractor utility."""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.utils.audio_extractor import AudioExtractor, AudioExtractionError


class TestAudioExtractorInit:
    """Tests for AudioExtractor initialization."""

    @patch("src.utils.audio_extractor.shutil.which")
    def test_init_with_ffmpeg_available(self, mock_which, tmp_path):
        """Should initialize successfully when FFmpeg is found."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        extractor = AudioExtractor(output_dir=str(tmp_path))
        assert extractor.sample_rate == 16000
        assert extractor.audio_format == "wav"

    @patch("src.utils.audio_extractor.shutil.which")
    def test_init_without_ffmpeg(self, mock_which):
        """Should raise RuntimeError when FFmpeg is not found."""
        mock_which.return_value = None
        with pytest.raises(RuntimeError, match="FFmpeg not found"):
            AudioExtractor()

    @patch("src.utils.audio_extractor.shutil.which")
    def test_init_custom_sample_rate(self, mock_which, tmp_path):
        """Should accept custom sample rate."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        extractor = AudioExtractor(sample_rate=44100, output_dir=str(tmp_path))
        assert extractor.sample_rate == 44100

    @patch("src.utils.audio_extractor.shutil.which")
    def test_init_creates_output_directory(self, mock_which, tmp_path):
        """Should create the output directory if it doesn't exist."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        output_dir = str(tmp_path / "new_dir" / "audio")
        AudioExtractor(output_dir=output_dir)
        assert os.path.isdir(output_dir)


class TestAudioExtractorValidation:
    """Tests for input validation."""

    @patch("src.utils.audio_extractor.shutil.which")
    def test_validate_nonexistent_file(self, mock_which, tmp_path):
        """Should raise FileNotFoundError for missing files."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        extractor = AudioExtractor(output_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            extractor.extract("nonexistent_video.mp4")

    @patch("src.utils.audio_extractor.shutil.which")
    def test_validate_unsupported_format(self, mock_which, tmp_path):
        """Should raise ValueError for unsupported video formats."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        extractor = AudioExtractor(output_dir=str(tmp_path))

        # Create a dummy file with unsupported extension
        dummy_file = tmp_path / "test.xyz"
        dummy_file.touch()

        with pytest.raises(ValueError, match="Unsupported video format"):
            extractor.extract(str(dummy_file))

    @patch("src.utils.audio_extractor.shutil.which")
    def test_validate_directory_instead_of_file(self, mock_which, tmp_path):
        """Should raise ValueError when path points to a directory."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        extractor = AudioExtractor(output_dir=str(tmp_path))

        dir_path = tmp_path / "somedir.mp4"
        dir_path.mkdir()

        with pytest.raises(ValueError, match="Path is not a file"):
            extractor.extract(str(dir_path))

    @patch("src.utils.audio_extractor.shutil.which")
    def test_validate_supported_formats(self, mock_which, tmp_path):
        """Should accept all supported video formats."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        extractor = AudioExtractor(output_dir=str(tmp_path))

        for ext in AudioExtractor.SUPPORTED_VIDEO_FORMATS:
            path = tmp_path / f"test{ext}"
            path.touch()
            # Validation should pass (extraction will fail but that's OK)
            validated = extractor._validate_input(str(path))
            assert validated.exists()


class TestAudioExtractorExtract:
    """Tests for the extract method."""

    @patch("src.utils.audio_extractor.shutil.which")
    def test_file_exists_error_without_overwrite(self, mock_which, tmp_path):
        """Should raise FileExistsError when output exists and overwrite=False."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        extractor = AudioExtractor(output_dir=str(tmp_path))

        # Create dummy input and output files
        input_file = tmp_path / "test.mp4"
        input_file.touch()
        output_file = tmp_path / "test.wav"
        output_file.touch()

        with pytest.raises(FileExistsError, match="Output file already exists"):
            extractor.extract(str(input_file))

    @patch("src.utils.audio_extractor.subprocess.run")
    @patch("src.utils.audio_extractor.shutil.which")
    def test_successful_extraction(self, mock_which, mock_run, tmp_path):
        """Should successfully extract audio when FFmpeg succeeds."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        extractor = AudioExtractor(output_dir=str(tmp_path))

        input_file = tmp_path / "test.mp4"
        input_file.touch()

        output_file = tmp_path / "test.wav"

        # Use side_effect to create the output file when subprocess.run is called
        # (simulating FFmpeg creating the file during execution)
        def fake_ffmpeg(*args, **kwargs):
            output_file.write_bytes(b"\x00" * 1024)
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = fake_ffmpeg

        result = extractor.extract(str(input_file))
        assert result == str(output_file)
        assert mock_run.called

    @patch("src.utils.audio_extractor.subprocess.run")
    @patch("src.utils.audio_extractor.shutil.which")
    def test_ffmpeg_failure(self, mock_which, mock_run, tmp_path):
        """Should raise AudioExtractionError when FFmpeg fails."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        extractor = AudioExtractor(output_dir=str(tmp_path))

        input_file = tmp_path / "test.mp4"
        input_file.touch()

        mock_run.return_value = MagicMock(
            returncode=1, stderr="Error: Invalid data found"
        )

        with pytest.raises(AudioExtractionError, match="FFmpeg failed"):
            extractor.extract(str(input_file))

    @patch("src.utils.audio_extractor.subprocess.run")
    @patch("src.utils.audio_extractor.shutil.which")
    def test_extraction_timeout(self, mock_which, mock_run, tmp_path):
        """Should raise AudioExtractionError on timeout."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        extractor = AudioExtractor(output_dir=str(tmp_path))

        input_file = tmp_path / "test.mp4"
        input_file.touch()

        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300)

        with pytest.raises(AudioExtractionError, match="timed out"):
            extractor.extract(str(input_file))

    @patch("src.utils.audio_extractor.subprocess.run")
    @patch("src.utils.audio_extractor.shutil.which")
    def test_custom_output_path(self, mock_which, mock_run, tmp_path):
        """Should save to custom output path when specified."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        extractor = AudioExtractor(output_dir=str(tmp_path))

        input_file = tmp_path / "test.mp4"
        input_file.touch()

        custom_output = tmp_path / "custom" / "output.wav"

        # Use side_effect to create output file during FFmpeg execution
        def fake_ffmpeg(*args, **kwargs):
            custom_output.parent.mkdir(parents=True, exist_ok=True)
            custom_output.write_bytes(b"\x00" * 512)
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = fake_ffmpeg

        result = extractor.extract(str(input_file), output_path=str(custom_output))
        assert result == str(custom_output)


class TestBuildOutputPath:
    """Tests for output path generation."""

    @patch("src.utils.audio_extractor.shutil.which")
    def test_default_output_path(self, mock_which, tmp_path):
        """Should generate output path in the configured output directory."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        extractor = AudioExtractor(output_dir=str(tmp_path))

        video_path = Path("/some/path/my_video.mp4")
        result = extractor._build_output_path(video_path)

        assert result == Path(tmp_path) / "my_video.wav"

    @patch("src.utils.audio_extractor.shutil.which")
    def test_custom_output_path(self, mock_which, tmp_path):
        """Should use custom output path when provided."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        extractor = AudioExtractor(output_dir=str(tmp_path))

        video_path = Path("/some/path/my_video.mp4")
        custom_path = str(tmp_path / "custom_name.wav")
        result = extractor._build_output_path(video_path, custom_path)

        assert result == Path(custom_path).resolve()
