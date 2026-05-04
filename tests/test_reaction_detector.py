"""Unit tests for the ReactionDetector and FrameExtractor.

These tests mock OpenCV and MediaPipe dependencies to allow testing
the detection logic without requiring actual video files or GPU.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from src.detectors.reaction_detector import ReactionDetector
from src.models.event import SoundEvent
from src.models.reaction import ReactionType


class TestReactionDetectorInit:
    """Tests for ReactionDetector initialization."""

    def test_default_initialization(self):
        """Should initialize with default settings."""
        detector = ReactionDetector()
        assert detector.confidence_threshold == 0.4
        assert detector.head_turn_threshold == 15.0
        assert detector.pose_change_threshold == 0.1

    def test_custom_parameters(self):
        """Should accept custom parameters."""
        detector = ReactionDetector(
            confidence_threshold=0.6,
            head_turn_threshold=20.0,
            pose_change_threshold=0.2,
        )
        assert detector.confidence_threshold == 0.6
        assert detector.head_turn_threshold == 20.0
        assert detector.pose_change_threshold == 0.2

    def test_mediapipe_not_loaded_on_init(self):
        """MediaPipe should not be loaded during initialization."""
        detector = ReactionDetector()
        assert detector._mp_face_mesh is None
        assert detector._mp_pose is None


class TestHeadAngleComputation:
    """Tests for head angle change computation."""

    def test_max_angle_change(self):
        """Should compute max angle change between consecutive frames."""
        head_angles = [
            (0.0, 0.0),
            (0.1, 5.0),
            (0.2, 25.0),  # biggest change: 20°
            (0.3, 22.0),
        ]
        result = ReactionDetector._compute_max_angle_change(head_angles)
        assert result == 20.0

    def test_max_angle_change_empty(self):
        """Should return 0.0 for empty list."""
        result = ReactionDetector._compute_max_angle_change([])
        assert result == 0.0

    def test_max_angle_change_single(self):
        """Should return 0.0 for single element."""
        result = ReactionDetector._compute_max_angle_change([(0.0, 10.0)])
        assert result == 0.0

    def test_no_change(self):
        """Should return 0.0 when all angles are the same."""
        head_angles = [(0.0, 5.0), (0.1, 5.0), (0.2, 5.0)]
        result = ReactionDetector._compute_max_angle_change(head_angles)
        assert result == 0.0


class TestPoseDisplacementComputation:
    """Tests for body pose displacement computation."""

    def test_max_displacement(self):
        """Should compute max Euclidean displacement between frames."""
        positions = [
            (0.0, np.array([0.0, 0.0, 0.0, 0.0])),
            (0.1, np.array([0.1, 0.0, 0.0, 0.0])),
            (0.2, np.array([0.5, 0.0, 0.0, 0.0])),  # biggest change
            (0.3, np.array([0.55, 0.0, 0.0, 0.0])),
        ]
        result = ReactionDetector._compute_max_displacement(positions)
        assert abs(result - 0.4) < 1e-6

    def test_displacement_empty(self):
        """Should return 0.0 for empty list."""
        result = ReactionDetector._compute_max_displacement([])
        assert result == 0.0

    def test_displacement_single(self):
        """Should return 0.0 for single element."""
        result = ReactionDetector._compute_max_displacement(
            [(0.0, np.array([0.5, 0.5]))]
        )
        assert result == 0.0


class TestHeadTurnConfidence:
    """Tests for head turn confidence computation."""

    def test_above_threshold(self):
        """Should return positive confidence when angle > threshold."""
        detector = ReactionDetector(head_turn_threshold=15.0)
        head_angles = [(0.0, 0.0), (0.1, 20.0)]
        conf = detector._compute_head_turn_confidence(head_angles)
        assert conf > 0.0

    def test_below_threshold(self):
        """Should return 0.0 when angle change < threshold."""
        detector = ReactionDetector(head_turn_threshold=15.0)
        head_angles = [(0.0, 0.0), (0.1, 5.0)]
        conf = detector._compute_head_turn_confidence(head_angles)
        assert conf == 0.0

    def test_max_confidence_capped(self):
        """Confidence should be capped at 1.0."""
        detector = ReactionDetector(head_turn_threshold=5.0)
        head_angles = [(0.0, 0.0), (0.1, 100.0)]
        conf = detector._compute_head_turn_confidence(head_angles)
        assert conf == 1.0


class TestBodyMovementConfidence:
    """Tests for body movement confidence computation."""

    def test_above_threshold(self):
        """Should return positive confidence for large displacement."""
        detector = ReactionDetector(pose_change_threshold=0.1)
        positions = [
            (0.0, np.array([0.0, 0.0])),
            (0.1, np.array([0.2, 0.0])),
        ]
        conf = detector._compute_body_movement_confidence(positions)
        assert conf > 0.0

    def test_below_threshold(self):
        """Should return 0.0 for small displacement."""
        detector = ReactionDetector(pose_change_threshold=0.1)
        positions = [
            (0.0, np.array([0.0, 0.0])),
            (0.1, np.array([0.01, 0.0])),
        ]
        conf = detector._compute_body_movement_confidence(positions)
        assert conf == 0.0


class TestDetectWithNoEvents:
    """Tests for edge cases in the detect method."""

    def test_empty_events_list(self):
        """Should return empty list when no sound events provided."""
        detector = ReactionDetector()
        result = detector.detect("video.mp4", [])
        assert result == []
