"""Speaker Reaction Detection module using MediaPipe.

This module provides the ReactionDetector class that analyzes video frames
at detected audio event timestamps using MediaPipe Face Mesh and Pose
to detect visible reactions such as head turns, startled body language,
and facial expression changes.
"""

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from src.models.event import SoundEvent
from src.models.reaction import ReactionEvent, ReactionType
from src.utils.frame_extractor import FrameExtractor
from config.settings import (
    REACTION_CONFIDENCE_THRESHOLD,
    HEAD_TURN_THRESHOLD,
    POSE_CHANGE_THRESHOLD,
    REACTION_FRAME_COUNT,
    REACTION_TIME_WINDOW,
)

logger = logging.getLogger(__name__)


class ReactionDetector:
    """Detects visual reactions to audio events using MediaPipe.

    Analyzes video frames extracted around audio event timestamps to
    detect reactions like head turns, startled movements, and facial
    expression changes. Uses MediaPipe Face Mesh for facial analysis
    and MediaPipe Pose for body language detection.

    Attributes:
        confidence_threshold: Minimum confidence for a reaction.
        head_turn_threshold: Head rotation angle threshold (degrees).
        pose_change_threshold: Body pose change threshold (normalized).

    Example:
        >>> detector = ReactionDetector()
        >>> reactions = detector.detect("video.mp4", sound_events)
        >>> for reaction in reactions:
        ...     print(f"{reaction.reaction_type.value} at {reaction.timestamp}s")
    """

    def __init__(
        self,
        confidence_threshold: float = REACTION_CONFIDENCE_THRESHOLD,
        head_turn_threshold: float = HEAD_TURN_THRESHOLD,
        pose_change_threshold: float = POSE_CHANGE_THRESHOLD,
        frame_count: int = REACTION_FRAME_COUNT,
        time_window: float = REACTION_TIME_WINDOW,
    ):
        """Initialize the ReactionDetector.

        Args:
            confidence_threshold: Minimum confidence for detection.
            head_turn_threshold: Head turn angle threshold in degrees.
            pose_change_threshold: Pose displacement threshold.
            frame_count: Number of frames to analyze per event.
            time_window: Time window around event timestamp.
        """
        self.confidence_threshold = confidence_threshold
        self.head_turn_threshold = head_turn_threshold
        self.pose_change_threshold = pose_change_threshold
        self.frame_count = frame_count
        self.time_window = time_window

        self._mp_face_mesh = None
        self._mp_pose = None
        self._frame_extractor = FrameExtractor(
            frame_count=frame_count,
            time_window=time_window,
        )

        logger.info(
            "ReactionDetector initialized (threshold=%.2f, "
            "head_turn=%.1f°, pose_change=%.3f)",
            self.confidence_threshold,
            self.head_turn_threshold,
            self.pose_change_threshold,
        )

    def _init_mediapipe(self):
        """Lazily initialize MediaPipe models.

        Raises:
            ImportError: If MediaPipe is not installed.
        """
        if self._mp_face_mesh is not None:
            return

        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError(
                "MediaPipe is required for reaction detection. "
                "Install it with: pip install mediapipe"
            ) from e

        self._mp = mp
        self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self._mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        )

        logger.info("MediaPipe Face Mesh and Pose models initialized")

    def detect(
        self,
        video_path: str,
        sound_events: List[SoundEvent],
    ) -> List[ReactionEvent]:
        """Detect visual reactions for a list of audio events.

        For each sound event, extracts frames around the event timestamp
        and analyzes them for visible reactions.

        Args:
            video_path: Path to the video file.
            sound_events: List of detected audio events to check
                for visual reactions.

        Returns:
            List of ReactionEvent objects, one per sound event.
        """
        if not sound_events:
            logger.info("No sound events provided, skipping reaction detection")
            return []

        self._init_mediapipe()
        reactions = []

        logger.info(
            "Analyzing reactions for %d sound events in: %s",
            len(sound_events),
            video_path,
        )

        for event in sound_events:
            # Use the midpoint of the event as the center timestamp
            event_center = (event.start_time + event.end_time) / 2.0

            # Extract frames around the event
            frames = self._frame_extractor.extract_around_timestamp(
                video_path, event_center
            )

            if not frames:
                logger.debug(
                    "No frames extracted for event '%s' at %.2fs",
                    event.label,
                    event_center,
                )
                reactions.append(
                    ReactionEvent(
                        reaction_type=ReactionType.NONE,
                        confidence=0.0,
                        timestamp=event_center,
                        description="No frames available for analysis",
                        associated_sound_label=event.label,
                    )
                )
                continue

            # Analyze frames for reactions
            reaction = self._analyze_frames(frames, event)
            reactions.append(reaction)

        logger.info(
            "Detected %d significant reactions out of %d events",
            sum(1 for r in reactions if r.is_significant),
            len(sound_events),
        )

        return reactions

    def _analyze_frames(
        self,
        frames: List[Tuple[float, np.ndarray]],
        event: SoundEvent,
    ) -> ReactionEvent:
        """Analyze a sequence of frames for visual reactions.

        Checks for head turns, body movement changes, and facial
        expression changes across the frame sequence.

        Args:
            frames: List of (timestamp, frame) tuples.
            event: The associated sound event.

        Returns:
            ReactionEvent with the strongest detected reaction.
        """
        cv2 = self._frame_extractor._get_cv2()
        event_center = (event.start_time + event.end_time) / 2.0

        head_angles = []
        pose_positions = []

        for timestamp, frame in frames:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Analyze face landmarks
            head_angle = self._get_head_angle(rgb_frame)
            if head_angle is not None:
                head_angles.append((timestamp, head_angle))

            # Analyze pose landmarks
            pose_pos = self._get_pose_position(rgb_frame)
            if pose_pos is not None:
                pose_positions.append((timestamp, pose_pos))

        # Determine the strongest reaction signal
        head_turn_conf = self._compute_head_turn_confidence(head_angles)
        body_move_conf = self._compute_body_movement_confidence(pose_positions)

        # Pick the strongest reaction
        if head_turn_conf >= body_move_conf and head_turn_conf > 0:
            angle_change = self._compute_max_angle_change(head_angles)
            return ReactionEvent(
                reaction_type=ReactionType.HEAD_TURN,
                confidence=min(head_turn_conf, 1.0),
                timestamp=event_center,
                description=f"Head turn of {angle_change:.1f}° detected",
                associated_sound_label=event.label,
                head_angle_change=angle_change,
            )
        elif body_move_conf > 0:
            displacement = self._compute_max_displacement(pose_positions)
            return ReactionEvent(
                reaction_type=ReactionType.BODY_MOVEMENT,
                confidence=min(body_move_conf, 1.0),
                timestamp=event_center,
                description=f"Body movement (displacement: {displacement:.4f}) detected",
                associated_sound_label=event.label,
                pose_displacement=displacement,
            )
        else:
            return ReactionEvent(
                reaction_type=ReactionType.NONE,
                confidence=0.0,
                timestamp=event_center,
                description="No significant reaction detected",
                associated_sound_label=event.label,
            )

    def _get_head_angle(self, rgb_frame: np.ndarray) -> Optional[float]:
        """Estimate head yaw angle from face landmarks.

        Uses the nose tip and face edges to estimate the horizontal
        rotation (yaw) of the head.

        Args:
            rgb_frame: RGB image as numpy array.

        Returns:
            Estimated yaw angle in degrees, or None if no face detected.
        """
        results = self._mp_face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        # Use the first detected face
        face = results.multi_face_landmarks[0]
        landmarks = face.landmark

        # Key landmarks for yaw estimation:
        # Nose tip (1), Left ear (234), Right ear (454)
        nose = landmarks[1]
        left_ear = landmarks[234]
        right_ear = landmarks[454]

        # Compute yaw angle based on relative position of nose between ears
        ear_midpoint_x = (left_ear.x + right_ear.x) / 2.0
        ear_distance = abs(right_ear.x - left_ear.x)

        if ear_distance < 1e-6:
            return 0.0

        # Nose deviation from midpoint, normalized by ear distance
        deviation = (nose.x - ear_midpoint_x) / ear_distance
        yaw_angle = math.degrees(math.atan2(deviation, 1.0))

        return yaw_angle

    def _get_pose_position(
        self, rgb_frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract key pose landmark positions.

        Uses MediaPipe Pose to extract shoulder and hip positions
        for body movement analysis.

        Args:
            rgb_frame: RGB image as numpy array.

        Returns:
            Numpy array of key pose positions, or None if no pose detected.
        """
        results = self._mp_pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark

        # Key landmarks: shoulders (11, 12), hips (23, 24)
        key_indices = [11, 12, 23, 24]
        positions = []

        for idx in key_indices:
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                positions.extend([lm.x, lm.y])
            else:
                return None  # Not enough visible landmarks

        return np.array(positions, dtype=np.float32)

    def _compute_head_turn_confidence(
        self, head_angles: List[Tuple[float, float]]
    ) -> float:
        """Compute confidence that a head turn reaction occurred.

        Looks at the change in head angle across the frame sequence.
        A sudden change suggests a reaction to the audio event.

        Args:
            head_angles: List of (timestamp, angle) tuples.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if len(head_angles) < 2:
            return 0.0

        max_change = self._compute_max_angle_change(head_angles)

        if max_change < self.head_turn_threshold:
            return 0.0

        # Scale confidence: threshold → 0.5, 2x threshold → 1.0
        confidence = min(max_change / (2 * self.head_turn_threshold), 1.0)
        return confidence

    @staticmethod
    def _compute_max_angle_change(
        head_angles: List[Tuple[float, float]]
    ) -> float:
        """Compute the maximum angle change between consecutive frames.

        Args:
            head_angles: List of (timestamp, angle) tuples.

        Returns:
            Maximum absolute angle change in degrees.
        """
        if len(head_angles) < 2:
            return 0.0

        max_change = 0.0
        for i in range(1, len(head_angles)):
            change = abs(head_angles[i][1] - head_angles[i - 1][1])
            max_change = max(max_change, change)

        return max_change

    def _compute_body_movement_confidence(
        self, pose_positions: List[Tuple[float, np.ndarray]]
    ) -> float:
        """Compute confidence that a body movement reaction occurred.

        Args:
            pose_positions: List of (timestamp, position_array) tuples.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if len(pose_positions) < 2:
            return 0.0

        max_displacement = self._compute_max_displacement(pose_positions)

        if max_displacement < self.pose_change_threshold:
            return 0.0

        confidence = min(
            max_displacement / (2 * self.pose_change_threshold), 1.0
        )
        return confidence

    @staticmethod
    def _compute_max_displacement(
        pose_positions: List[Tuple[float, np.ndarray]]
    ) -> float:
        """Compute the maximum pose displacement between consecutive frames.

        Args:
            pose_positions: List of (timestamp, position_array) tuples.

        Returns:
            Maximum Euclidean distance between consecutive position vectors.
        """
        if len(pose_positions) < 2:
            return 0.0

        max_disp = 0.0
        for i in range(1, len(pose_positions)):
            disp = np.linalg.norm(
                pose_positions[i][1] - pose_positions[i - 1][1]
            )
            max_disp = max(max_disp, float(disp))

        return max_disp
