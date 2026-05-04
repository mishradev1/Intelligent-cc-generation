"""Unit tests for the ReactionEvent data model."""

import pytest
from src.models.reaction import ReactionEvent, ReactionType


class TestReactionEventCreation:
    """Tests for ReactionEvent initialization."""

    def test_basic_creation(self):
        """Should create a ReactionEvent with valid parameters."""
        event = ReactionEvent(
            reaction_type=ReactionType.HEAD_TURN,
            confidence=0.75,
            timestamp=5.0,
            description="Head turn of 20° detected",
        )
        assert event.reaction_type == ReactionType.HEAD_TURN
        assert event.confidence == 0.75
        assert event.timestamp == 5.0

    def test_with_associated_sound(self):
        """Should store associated sound label."""
        event = ReactionEvent(
            reaction_type=ReactionType.STARTLED,
            confidence=0.8,
            timestamp=3.0,
            associated_sound_label="explosion",
        )
        assert event.associated_sound_label == "explosion"

    def test_with_head_angle(self):
        """Should store head angle change."""
        event = ReactionEvent(
            reaction_type=ReactionType.HEAD_TURN,
            confidence=0.6,
            timestamp=2.0,
            head_angle_change=25.5,
        )
        assert event.head_angle_change == 25.5

    def test_with_pose_displacement(self):
        """Should store pose displacement."""
        event = ReactionEvent(
            reaction_type=ReactionType.BODY_MOVEMENT,
            confidence=0.5,
            timestamp=7.0,
            pose_displacement=0.15,
        )
        assert event.pose_displacement == 0.15

    def test_invalid_confidence(self):
        """Should raise ValueError for invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            ReactionEvent(
                reaction_type=ReactionType.NONE,
                confidence=1.5,
                timestamp=0.0,
            )

    def test_negative_timestamp(self):
        """Should raise ValueError for negative timestamp."""
        with pytest.raises(ValueError, match="Timestamp must be non-negative"):
            ReactionEvent(
                reaction_type=ReactionType.NONE,
                confidence=0.5,
                timestamp=-1.0,
            )


class TestReactionEventProperties:
    """Tests for ReactionEvent properties and methods."""

    def test_is_significant_true(self):
        """Should be significant when type is not NONE and confidence > 0."""
        event = ReactionEvent(
            reaction_type=ReactionType.HEAD_TURN,
            confidence=0.5,
            timestamp=3.0,
        )
        assert event.is_significant is True

    def test_is_significant_false_none_type(self):
        """Should not be significant when type is NONE."""
        event = ReactionEvent(
            reaction_type=ReactionType.NONE,
            confidence=0.5,
            timestamp=3.0,
        )
        assert event.is_significant is False

    def test_is_significant_false_zero_confidence(self):
        """Should not be significant when confidence is 0."""
        event = ReactionEvent(
            reaction_type=ReactionType.HEAD_TURN,
            confidence=0.0,
            timestamp=3.0,
        )
        assert event.is_significant is False

    def test_to_dict(self):
        """Should convert to dict with all fields."""
        event = ReactionEvent(
            reaction_type=ReactionType.BODY_MOVEMENT,
            confidence=0.6543,
            timestamp=12.345,
            description="Startled movement",
            associated_sound_label="gunshot",
            head_angle_change=15.78,
            pose_displacement=0.12345,
        )
        d = event.to_dict()
        assert d["reaction_type"] == "body_movement"
        assert d["confidence"] == 0.6543
        assert d["timestamp"] == 12.345
        assert d["description"] == "Startled movement"
        assert d["associated_sound_label"] == "gunshot"
        assert d["head_angle_change"] == 15.78
        assert d["pose_displacement"] == 0.1235  # rounded to 4 decimals

    def test_to_dict_none_values(self):
        """Should handle None optional fields in dict."""
        event = ReactionEvent(
            reaction_type=ReactionType.NONE,
            confidence=0.0,
            timestamp=0.0,
        )
        d = event.to_dict()
        assert d["head_angle_change"] is None
        assert d["pose_displacement"] is None
        assert d["associated_sound_label"] is None

    def test_repr(self):
        """Should have a readable repr."""
        event = ReactionEvent(
            reaction_type=ReactionType.GAZE_SHIFT,
            confidence=0.65,
            timestamp=8.5,
        )
        r = repr(event)
        assert "gaze_shift" in r
        assert "0.65" in r
        assert "8.50" in r


class TestReactionType:
    """Tests for ReactionType enum."""

    def test_all_types(self):
        """Should have all expected reaction types."""
        types = [t.value for t in ReactionType]
        assert "head_turn" in types
        assert "startled" in types
        assert "facial_expression" in types
        assert "speech_pause" in types
        assert "body_movement" in types
        assert "gaze_shift" in types
        assert "none" in types

    def test_enum_values(self):
        """Enum values should be lowercase strings."""
        for t in ReactionType:
            assert t.value == t.value.lower()
