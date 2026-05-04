"""Unit tests for the SRT Generator."""

import pytest
from pathlib import Path

from src.output.srt_generator import SRTGenerator
from src.models.cc_suggestion import CCSuggestion


class TestTimestampFormatting:
    """Tests for SRT timestamp formatting."""

    def test_zero(self):
        """Should format 0 seconds correctly."""
        assert SRTGenerator._format_timestamp(0.0) == "00:00:00,000"

    def test_seconds_only(self):
        """Should format seconds correctly."""
        assert SRTGenerator._format_timestamp(5.5) == "00:00:05,500"

    def test_minutes(self):
        """Should format minutes correctly."""
        assert SRTGenerator._format_timestamp(65.0) == "00:01:05,000"

    def test_hours(self):
        """Should format hours correctly."""
        assert SRTGenerator._format_timestamp(3661.5) == "01:01:01,500"

    def test_millisecond_precision(self):
        """Should handle millisecond precision."""
        assert SRTGenerator._format_timestamp(1.123) == "00:00:01,123"

    def test_negative_clamped_to_zero(self):
        """Negative timestamps should be clamped to 0."""
        assert SRTGenerator._format_timestamp(-5.0) == "00:00:00,000"

    def test_large_value(self):
        """Should handle large timestamps."""
        # 2 hours, 30 minutes, 45.678 seconds
        result = SRTGenerator._format_timestamp(9045.678)
        assert result == "02:30:45,678"


class TestSRTEntryFormatting:
    """Tests for individual SRT entry formatting."""

    def test_basic_entry(self):
        """Should format a basic SRT entry correctly."""
        suggestion = CCSuggestion(
            cc_text="[honking]",
            start_time=1.5,
            end_time=3.5,
            audio_confidence=0.9,
            visual_confidence=0.7,
            combined_score=0.82,
            sound_label="honking",
            include=True,
        )
        entry = SRTGenerator._format_entry(1, suggestion)
        assert "1\n" in entry
        assert "00:00:01,500 --> 00:00:03,500\n" in entry
        assert "[honking]" in entry


class TestSRTGeneration:
    """Tests for SRT file generation."""

    def _make_suggestion(self, text, start, end, include=True, score=0.8):
        return CCSuggestion(
            cc_text=text,
            start_time=start,
            end_time=end,
            audio_confidence=0.8,
            visual_confidence=0.6,
            combined_score=score,
            sound_label=text.strip("[]"),
            include=include,
        )

    def test_generate_file(self, tmp_path):
        """Should generate a valid SRT file."""
        generator = SRTGenerator(output_dir=str(tmp_path))
        suggestions = [
            self._make_suggestion("[honking]", 1.0, 3.0),
            self._make_suggestion("[laughter]", 5.0, 7.0),
        ]
        output = generator.generate(suggestions, "test.srt")
        assert Path(output).exists()

        content = Path(output).read_text(encoding="utf-8")
        assert "[honking]" in content
        assert "[laughter]" in content
        assert "1\n" in content
        assert "2\n" in content

    def test_excluded_not_in_output(self, tmp_path):
        """Should not include excluded suggestions."""
        generator = SRTGenerator(output_dir=str(tmp_path))
        suggestions = [
            self._make_suggestion("[honking]", 1.0, 3.0, include=True),
            self._make_suggestion("[wind]", 5.0, 7.0, include=False),
        ]
        output = generator.generate(suggestions, "test.srt")
        content = Path(output).read_text(encoding="utf-8")
        assert "[honking]" in content
        assert "[wind]" not in content

    def test_include_all_flag(self, tmp_path):
        """Should include all when include_all=True."""
        generator = SRTGenerator(output_dir=str(tmp_path))
        suggestions = [
            self._make_suggestion("[honking]", 1.0, 3.0, include=True),
            self._make_suggestion("[wind]", 5.0, 7.0, include=False),
        ]
        output = generator.generate(suggestions, "test.srt", include_all=True)
        content = Path(output).read_text(encoding="utf-8")
        assert "[honking]" in content
        assert "[wind]" in content

    def test_sorted_by_start_time(self, tmp_path):
        """Output should be sorted by start_time."""
        generator = SRTGenerator(output_dir=str(tmp_path))
        suggestions = [
            self._make_suggestion("[laughter]", 5.0, 7.0),
            self._make_suggestion("[honking]", 1.0, 3.0),
        ]
        output = generator.generate(suggestions, "test.srt")
        content = Path(output).read_text(encoding="utf-8")

        # honking should come before laughter
        honk_pos = content.index("[honking]")
        laugh_pos = content.index("[laughter]")
        assert honk_pos < laugh_pos

    def test_empty_suggestions(self, tmp_path):
        """Should handle empty suggestions list."""
        generator = SRTGenerator(output_dir=str(tmp_path))
        output = generator.generate([], "empty.srt")
        content = Path(output).read_text(encoding="utf-8")
        assert content == ""

    def test_generate_string(self):
        """Should generate SRT content as string."""
        generator = SRTGenerator()
        suggestions = [
            self._make_suggestion("[honking]", 1.0, 3.0),
        ]
        content = generator.generate_string(suggestions)
        assert "[honking]" in content
        assert "00:00:01,000 --> 00:00:03,000" in content


class TestCLIParsing:
    """Tests for CLI argument parsing."""

    def test_required_input(self):
        """Should require --input argument."""
        from src.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([])

    def test_default_values(self):
        """Should set default values when not specified."""
        from src.cli import parse_args

        args = parse_args(["--input", "video.mp4"])
        assert args.input == "video.mp4"
        assert args.output == "captions.srt"
        assert args.audio_only is False
        assert args.verbose is False

    def test_custom_values(self):
        """Should accept custom values."""
        from src.cli import parse_args

        args = parse_args([
            "--input", "video.mp4",
            "--output", "my_captions.srt",
            "--audio-threshold", "0.5",
            "--audio-only",
            "--verbose",
        ])
        assert args.input == "video.mp4"
        assert args.output == "my_captions.srt"
        assert args.audio_threshold == 0.5
        assert args.audio_only is True
        assert args.verbose is True
