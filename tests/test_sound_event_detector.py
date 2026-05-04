"""Unit tests for the SoundEventDetector.

These tests mock the TensorFlow/YAMNet dependencies to allow testing
the detection logic without requiring the actual model or GPU.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from src.detectors.sound_event_detector import SoundEventDetector, CC_LABEL_MAP
from src.models.event import SoundEvent


class TestSoundEventDetectorInit:
    """Tests for SoundEventDetector initialization."""

    def test_default_initialization(self):
        """Should initialize with default settings."""
        detector = SoundEventDetector()
        assert detector.confidence_threshold == 0.3
        assert detector.sample_rate == 16000
        assert len(detector.target_events) > 0

    def test_custom_threshold(self):
        """Should accept custom confidence threshold."""
        detector = SoundEventDetector(confidence_threshold=0.5)
        assert detector.confidence_threshold == 0.5

    def test_custom_target_events(self):
        """Should accept custom target events list."""
        targets = ["Laughter", "Applause"]
        detector = SoundEventDetector(target_events=targets)
        assert detector.target_events == set(targets)

    def test_model_not_loaded_on_init(self):
        """Model should not be loaded during initialization (lazy loading)."""
        detector = SoundEventDetector()
        assert detector._model is None
        assert detector._class_names is None


class TestMergeConsecutiveEvents:
    """Tests for the event merging logic."""

    def test_merge_same_label_consecutive(self):
        """Should merge consecutive events with the same label."""
        events = [
            SoundEvent(
                label="honking", raw_label="Horn",
                confidence=0.8, start_time=0.0, end_time=0.96,
            ),
            SoundEvent(
                label="honking", raw_label="Horn",
                confidence=0.7, start_time=0.48, end_time=1.44,
            ),
            SoundEvent(
                label="honking", raw_label="Horn",
                confidence=0.9, start_time=0.96, end_time=1.92,
            ),
        ]
        merged = SoundEventDetector._merge_consecutive_events(events)
        assert len(merged) == 1
        assert merged[0].label == "honking"
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 1.92
        assert merged[0].confidence == 0.9  # max confidence

    def test_no_merge_different_labels(self):
        """Should not merge events with different labels."""
        events = [
            SoundEvent(
                label="honking", raw_label="Horn",
                confidence=0.8, start_time=0.0, end_time=0.96,
            ),
            SoundEvent(
                label="laughter", raw_label="Laughter",
                confidence=0.7, start_time=0.48, end_time=1.44,
            ),
        ]
        merged = SoundEventDetector._merge_consecutive_events(events)
        assert len(merged) == 2

    def test_no_merge_large_gap(self):
        """Should not merge events with a large time gap."""
        events = [
            SoundEvent(
                label="honking", raw_label="Horn",
                confidence=0.8, start_time=0.0, end_time=0.96,
            ),
            SoundEvent(
                label="honking", raw_label="Horn",
                confidence=0.7, start_time=5.0, end_time=5.96,
            ),
        ]
        merged = SoundEventDetector._merge_consecutive_events(events)
        assert len(merged) == 2

    def test_merge_empty_list(self):
        """Should return empty list for empty input."""
        merged = SoundEventDetector._merge_consecutive_events([])
        assert merged == []

    def test_merge_single_event(self):
        """Should return the single event unchanged."""
        events = [
            SoundEvent(
                label="explosion", raw_label="Explosion",
                confidence=0.9, start_time=3.0, end_time=3.96,
            ),
        ]
        merged = SoundEventDetector._merge_consecutive_events(events)
        assert len(merged) == 1
        assert merged[0].confidence == 0.9

    def test_merge_preserves_max_confidence(self):
        """Merged event should have the maximum confidence of constituents."""
        events = [
            SoundEvent(
                label="siren", raw_label="Siren",
                confidence=0.3, start_time=0.0, end_time=0.96,
            ),
            SoundEvent(
                label="siren", raw_label="Siren",
                confidence=0.95, start_time=0.48, end_time=1.44,
            ),
            SoundEvent(
                label="siren", raw_label="Siren",
                confidence=0.5, start_time=0.96, end_time=1.92,
            ),
        ]
        merged = SoundEventDetector._merge_consecutive_events(events)
        assert len(merged) == 1
        assert merged[0].confidence == 0.95


class TestCCLabelMap:
    """Tests for the CC label mapping."""

    def test_all_target_events_have_mapping(self):
        """All default target events should have a CC label mapping."""
        from config.settings import TARGET_SOUND_EVENTS

        for event in TARGET_SOUND_EVENTS:
            assert event in CC_LABEL_MAP, (
                f"Target event '{event}' has no CC label mapping"
            )

    def test_label_format(self):
        """CC labels should be lowercase and human-readable."""
        for raw_label, cc_label in CC_LABEL_MAP.items():
            assert cc_label == cc_label.lower(), (
                f"CC label '{cc_label}' should be lowercase"
            )
            assert len(cc_label) > 0, (
                f"CC label for '{raw_label}' should not be empty"
            )


class TestDetectFromAudio:
    """Tests for the detect_from_audio method with mocked YAMNet."""

    @patch.object(SoundEventDetector, "_load_model")
    def test_empty_waveform(self, mock_load):
        """Should return empty list for empty audio."""
        detector = SoundEventDetector()
        events = detector.detect_from_audio(np.array([], dtype=np.float32))
        assert events == []

    @patch.object(SoundEventDetector, "_load_model")
    @patch.object(SoundEventDetector, "_get_target_indices")
    def test_detection_with_mocked_model(self, mock_indices, mock_load):
        """Should detect events when scores exceed threshold."""
        detector = SoundEventDetector(confidence_threshold=0.3)

        # Mock: class index 0 = 'honking', class index 1 = 'laughter'
        mock_indices.return_value = {0: "Vehicle horn, car horn, honking", 1: "Laughter"}

        # Create fake scores: 3 frames, 2 classes
        # Frame 0: honking=0.9, laughter=0.1
        # Frame 1: honking=0.1, laughter=0.8
        # Frame 2: honking=0.1, laughter=0.1
        fake_scores = np.array([
            [0.9, 0.1],
            [0.1, 0.8],
            [0.1, 0.1],
        ])

        # Mock the model call
        mock_model = MagicMock()
        mock_model.return_value = (
            MagicMock(numpy=MagicMock(return_value=fake_scores)),
            MagicMock(),  # embeddings
            MagicMock(),  # spectrogram
        )
        detector._model = mock_model

        waveform = np.random.randn(48000).astype(np.float32)
        events = detector.detect_from_audio(waveform)

        # Should detect honking in frame 0 and laughter in frame 1
        labels = [e.label for e in events]
        assert "honking" in labels
        assert "laughter" in labels

    @patch.object(SoundEventDetector, "_load_model")
    @patch.object(SoundEventDetector, "_get_target_indices")
    def test_threshold_filtering(self, mock_indices, mock_load):
        """Should filter out events below confidence threshold."""
        detector = SoundEventDetector(confidence_threshold=0.5)
        mock_indices.return_value = {0: "Laughter"}

        # Score below threshold
        fake_scores = np.array([[0.3]])
        mock_model = MagicMock()
        mock_model.return_value = (
            MagicMock(numpy=MagicMock(return_value=fake_scores)),
            MagicMock(),
            MagicMock(),
        )
        detector._model = mock_model

        waveform = np.random.randn(16000).astype(np.float32)
        events = detector.detect_from_audio(waveform)

        assert len(events) == 0

    @patch.object(SoundEventDetector, "_load_model")
    @patch.object(SoundEventDetector, "_get_target_indices")
    def test_events_sorted_by_start_time(self, mock_indices, mock_load):
        """Returned events should be sorted by start_time."""
        detector = SoundEventDetector(confidence_threshold=0.3)
        mock_indices.return_value = {0: "Laughter", 1: "Applause"}

        # Frame 0: laughter low, applause high
        # Frame 1: laughter high, applause low
        fake_scores = np.array([
            [0.1, 0.8],
            [0.8, 0.1],
        ])

        mock_model = MagicMock()
        mock_model.return_value = (
            MagicMock(numpy=MagicMock(return_value=fake_scores)),
            MagicMock(),
            MagicMock(),
        )
        detector._model = mock_model

        waveform = np.random.randn(32000).astype(np.float32)
        events = detector.detect_from_audio(waveform)

        for i in range(len(events) - 1):
            assert events[i].start_time <= events[i + 1].start_time
