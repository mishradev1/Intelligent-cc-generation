"""Data model for detected speaker/scene reactions.

This module defines the ReactionEvent dataclass used to represent
visual reactions detected in video frames at audio event timestamps.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ReactionType(Enum):
    """Types of visual reactions that can be detected."""

    HEAD_TURN = "head_turn"
    STARTLED = "startled"
    FACIAL_EXPRESSION = "facial_expression"
    SPEECH_PAUSE = "speech_pause"
    BODY_MOVEMENT = "body_movement"
    GAZE_SHIFT = "gaze_shift"
    NONE = "none"


@dataclass
class ReactionEvent:
    """Represents a detected visual reaction to an audio event.

    Attributes:
        reaction_type: Type of reaction detected.
        confidence: Reaction confidence score between 0.0 and 1.0.
        timestamp: Video timestamp where the reaction was detected (seconds).
        description: Human-readable description of the detected reaction.
        associated_sound_label: Label of the audio event this reaction is
            associated with, if any.
        head_angle_change: Measured head rotation angle change in degrees,
            if applicable.
        pose_displacement: Measured body pose displacement (normalized),
            if applicable.
    """

    reaction_type: ReactionType
    confidence: float
    timestamp: float
    description: str = ""
    associated_sound_label: Optional[str] = None
    head_angle_change: Optional[float] = None
    pose_displacement: Optional[float] = None

    def __post_init__(self):
        """Validate fields after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        if self.timestamp < 0:
            raise ValueError(
                f"Timestamp must be non-negative, got {self.timestamp}"
            )

    @property
    def is_significant(self) -> bool:
        """Whether this reaction is significant enough to warrant a CC.

        Returns:
            True if the reaction type is not NONE and confidence > 0.
        """
        return self.reaction_type != ReactionType.NONE and self.confidence > 0

    def to_dict(self) -> dict:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all reaction attributes.
        """
        return {
            "reaction_type": self.reaction_type.value,
            "confidence": round(self.confidence, 4),
            "timestamp": round(self.timestamp, 3),
            "description": self.description,
            "associated_sound_label": self.associated_sound_label,
            "head_angle_change": (
                round(self.head_angle_change, 2)
                if self.head_angle_change is not None
                else None
            ),
            "pose_displacement": (
                round(self.pose_displacement, 4)
                if self.pose_displacement is not None
                else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"ReactionEvent(type={self.reaction_type.value}, "
            f"confidence={self.confidence:.2f}, "
            f"time={self.timestamp:.2f}s)"
        )
