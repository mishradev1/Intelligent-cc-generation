"""Data model for CC suggestions produced by the decision engine."""

from dataclasses import dataclass


@dataclass
class CCSuggestion:
    """Represents a closed caption suggestion to include in the output.

    Attributes:
        cc_text: The formatted CC text (e.g., '[honking]').
        start_time: Start timestamp in seconds.
        end_time: End timestamp in seconds.
        audio_confidence: Confidence score from the audio detector.
        visual_confidence: Confidence score from the visual detector.
        combined_score: Weighted combination of audio and visual scores.
        sound_label: Original sound event label.
        include: Whether this CC should be included in the output.
    """

    cc_text: str
    start_time: float
    end_time: float
    audio_confidence: float
    visual_confidence: float
    combined_score: float
    sound_label: str
    include: bool

    def __post_init__(self):
        """Validate fields."""
        if self.start_time < 0:
            raise ValueError(
                f"start_time must be non-negative, got {self.start_time}"
            )
        if self.end_time < self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be >= start_time ({self.start_time})"
            )

    @property
    def duration(self) -> float:
        """Duration of the CC display in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "cc_text": self.cc_text,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.duration, 3),
            "audio_confidence": round(self.audio_confidence, 4),
            "visual_confidence": round(self.visual_confidence, 4),
            "combined_score": round(self.combined_score, 4),
            "sound_label": self.sound_label,
            "include": self.include,
        }

    def __repr__(self) -> str:
        status = "✓" if self.include else "✗"
        return (
            f"CCSuggestion({status} {self.cc_text}, "
            f"score={self.combined_score:.2f}, "
            f"time={self.start_time:.2f}s-{self.end_time:.2f}s)"
        )
