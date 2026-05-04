"""Unit tests for the SoundEvent data model."""

import pytest
from src.models.event import SoundEvent


class TestSoundEventCreation:
    """Tests for SoundEvent initialization and validation."""

    def test_basic_creation(self):
        """Should create a SoundEvent with valid parameters."""
        event = SoundEvent(
            label="honking",
            raw_label="Vehicle horn, car horn, honking",
            confidence=0.85,
            start_time=1.5,
            end_time=2.5,
        )
        assert event.label == "honking"
        assert event.confidence == 0.85
        assert event.start_time == 1.5
        assert event.end_time == 2.5
        assert event.cc_text == "[honking]"

    def test_auto_generated_cc_text(self):
        """Should auto-generate cc_text from label."""
        event = SoundEvent(
            label="explosion",
            raw_label="Explosion",
            confidence=0.9,
            start_time=0.0,
            end_time=1.0,
        )
        assert event.cc_text == "[explosion]"

    def test_custom_cc_text(self):
        """Should use provided cc_text over auto-generated."""
        event = SoundEvent(
            label="honking",
            raw_label="Vehicle horn",
            confidence=0.5,
            start_time=0.0,
            end_time=1.0,
            cc_text="[loud car horn]",
        )
        assert event.cc_text == "[loud car horn]"

    def test_invalid_confidence_too_high(self):
        """Should raise ValueError for confidence > 1.0."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            SoundEvent(
                label="test",
                raw_label="test",
                confidence=1.5,
                start_time=0.0,
                end_time=1.0,
            )

    def test_invalid_confidence_negative(self):
        """Should raise ValueError for negative confidence."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            SoundEvent(
                label="test",
                raw_label="test",
                confidence=-0.1,
                start_time=0.0,
                end_time=1.0,
            )

    def test_invalid_start_time_negative(self):
        """Should raise ValueError for negative start_time."""
        with pytest.raises(ValueError, match="start_time must be non-negative"):
            SoundEvent(
                label="test",
                raw_label="test",
                confidence=0.5,
                start_time=-1.0,
                end_time=1.0,
            )

    def test_invalid_end_before_start(self):
        """Should raise ValueError when end_time < start_time."""
        with pytest.raises(ValueError, match="end_time .* must be >= start_time"):
            SoundEvent(
                label="test",
                raw_label="test",
                confidence=0.5,
                start_time=5.0,
                end_time=3.0,
            )

    def test_zero_duration_event(self):
        """Should allow events with zero duration (instantaneous)."""
        event = SoundEvent(
            label="gunshot",
            raw_label="Gunshot, gunfire",
            confidence=0.7,
            start_time=2.0,
            end_time=2.0,
        )
        assert event.duration == 0.0


class TestSoundEventProperties:
    """Tests for SoundEvent properties and methods."""

    def test_duration(self):
        """Should correctly calculate duration."""
        event = SoundEvent(
            label="laughter",
            raw_label="Laughter",
            confidence=0.6,
            start_time=10.0,
            end_time=12.5,
        )
        assert event.duration == 2.5

    def test_to_dict(self):
        """Should convert to dictionary with all fields."""
        event = SoundEvent(
            label="applause",
            raw_label="Applause",
            confidence=0.7654321,
            start_time=5.1234,
            end_time=7.5678,
        )
        d = event.to_dict()
        assert d["label"] == "applause"
        assert d["raw_label"] == "Applause"
        assert d["confidence"] == 0.7654  # rounded to 4 decimals
        assert d["start_time"] == 5.123  # rounded to 3 decimals
        assert d["end_time"] == 7.568
        assert d["duration"] == 2.444
        assert d["cc_text"] == "[applause]"

    def test_repr(self):
        """Should have a readable repr."""
        event = SoundEvent(
            label="siren",
            raw_label="Siren",
            confidence=0.45,
            start_time=3.0,
            end_time=5.0,
        )
        r = repr(event)
        assert "siren" in r
        assert "0.45" in r
        assert "3.00" in r
        assert "5.00" in r

    def test_boundary_confidence_values(self):
        """Should accept confidence exactly at 0.0 and 1.0."""
        event_zero = SoundEvent(
            label="test", raw_label="test",
            confidence=0.0, start_time=0.0, end_time=1.0,
        )
        event_one = SoundEvent(
            label="test", raw_label="test",
            confidence=1.0, start_time=0.0, end_time=1.0,
        )
        assert event_zero.confidence == 0.0
        assert event_one.confidence == 1.0
