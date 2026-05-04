"""Data model for detected sound events.

This module defines the SoundEvent dataclass used to represent
non-speech audio events detected by the sound event detection pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SoundEvent:
    """Represents a detected non-speech audio event.

    Attributes:
        label: Human-readable label for the sound event
            (e.g., 'honking', 'explosion', 'laughter').
        raw_label: Original label from the detection model
            (e.g., 'Vehicle horn, car horn, honking').
        confidence: Detection confidence score between 0.0 and 1.0.
        start_time: Start timestamp of the event in seconds.
        end_time: End timestamp of the event in seconds.
        cc_text: Formatted closed caption text
            (e.g., '[honking]', '[crowd cheering]').
    """

    label: str
    raw_label: str
    confidence: float
    start_time: float
    end_time: float
    cc_text: Optional[str] = None

    def __post_init__(self):
        """Validate fields and auto-generate cc_text if not provided."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        if self.start_time < 0:
            raise ValueError(
                f"start_time must be non-negative, got {self.start_time}"
            )
        if self.end_time < self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be >= start_time ({self.start_time})"
            )
        if self.cc_text is None:
            self.cc_text = f"[{self.label}]"

    @property
    def duration(self) -> float:
        """Duration of the sound event in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all event attributes.
        """
        return {
            "label": self.label,
            "raw_label": self.raw_label,
            "confidence": round(self.confidence, 4),
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.duration, 3),
            "cc_text": self.cc_text,
        }

    def __repr__(self) -> str:
        return (
            f"SoundEvent(label='{self.label}', confidence={self.confidence:.2f}, "
            f"time={self.start_time:.2f}s-{self.end_time:.2f}s)"
        )
