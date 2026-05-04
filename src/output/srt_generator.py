"""SRT (SubRip Subtitle) file generator.

This module generates standard SRT subtitle files from CC suggestions.
SRT is the most widely supported subtitle format for video players
and editing software.

SRT Format:
    1
    00:00:01,500 --> 00:00:03,500
    [honking]

    2
    00:00:05,000 --> 00:00:07,000
    [crowd cheering]
"""

import logging
from pathlib import Path
from typing import List, Optional

from src.models.cc_suggestion import CCSuggestion
from config.settings import OUTPUT_DIR, OUTPUT_FORMAT

logger = logging.getLogger(__name__)


class SRTGenerator:
    """Generates SRT subtitle files from CC suggestions.

    Converts a list of CCSuggestion objects (with include=True) into
    a properly formatted SRT file with sequential numbering and
    timestamps.

    Attributes:
        output_dir: Directory to save generated SRT files.

    Example:
        >>> generator = SRTGenerator()
        >>> output_path = generator.generate(suggestions, "output.srt")
        >>> print(f"SRT file saved to: {output_path}")
    """

    def __init__(self, output_dir: str = OUTPUT_DIR):
        """Initialize the SRTGenerator.

        Args:
            output_dir: Directory for output SRT files.
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("SRTGenerator initialized (output_dir=%s)", self.output_dir)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

        Args:
            seconds: Time in seconds.

        Returns:
            Formatted timestamp string.
        """
        if seconds < 0:
            seconds = 0

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds % 1) * 1000))

        # Handle edge case where rounding gives 1000ms
        if millis >= 1000:
            millis = 0
            secs += 1
            if secs >= 60:
                secs = 0
                minutes += 1
                if minutes >= 60:
                    minutes = 0
                    hours += 1

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_entry(index: int, suggestion: CCSuggestion) -> str:
        """Format a single SRT entry.

        Args:
            index: 1-based sequence number.
            suggestion: The CC suggestion to format.

        Returns:
            Formatted SRT entry string.
        """
        start = SRTGenerator._format_timestamp(suggestion.start_time)
        end = SRTGenerator._format_timestamp(suggestion.end_time)

        return f"{index}\n{start} --> {end}\n{suggestion.cc_text}\n"

    def generate(
        self,
        suggestions: List[CCSuggestion],
        output_filename: Optional[str] = None,
        include_all: bool = False,
    ) -> str:
        """Generate an SRT file from CC suggestions.

        Only includes suggestions where include=True, unless
        include_all is set.

        Args:
            suggestions: List of CCSuggestion objects.
            output_filename: Output filename. If None, generates a
                default name.
            include_all: If True, includes all suggestions regardless
                of the include flag (useful for debugging).

        Returns:
            Absolute path to the generated SRT file.
        """
        # Filter to included suggestions only
        if include_all:
            filtered = sorted(suggestions, key=lambda s: s.start_time)
        else:
            filtered = sorted(
                [s for s in suggestions if s.include],
                key=lambda s: s.start_time,
            )

        if not filtered:
            logger.warning("No CC suggestions to write to SRT file")

        # Build SRT content
        entries = []
        for i, suggestion in enumerate(filtered, start=1):
            entries.append(self._format_entry(i, suggestion))

        srt_content = "\n".join(entries)

        # Determine output path
        if output_filename is None:
            output_filename = "captions.srt"

        output_path = Path(self.output_dir) / output_filename

        # Write SRT file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(
            "SRT file generated: %s (%d entries)",
            output_path,
            len(filtered),
        )

        return str(output_path.resolve())

    def generate_string(
        self,
        suggestions: List[CCSuggestion],
        include_all: bool = False,
    ) -> str:
        """Generate SRT content as a string without writing to file.

        Args:
            suggestions: List of CCSuggestion objects.
            include_all: If True, includes all suggestions.

        Returns:
            SRT formatted string.
        """
        if include_all:
            filtered = sorted(suggestions, key=lambda s: s.start_time)
        else:
            filtered = sorted(
                [s for s in suggestions if s.include],
                key=lambda s: s.start_time,
            )

        entries = []
        for i, suggestion in enumerate(filtered, start=1):
            entries.append(self._format_entry(i, suggestion))

        return "\n".join(entries)
