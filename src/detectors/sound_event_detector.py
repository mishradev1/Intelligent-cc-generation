"""Sound Event Detection module using YAMNet.

This module provides the SoundEventDetector class that uses Google's
YAMNet model (via TensorFlow Hub) to detect and classify non-speech
audio events in video/audio files with confidence scores and timestamps.

YAMNet is trained on the AudioSet ontology and can classify 521 audio
event categories. We filter for non-speech events that are relevant
for closed captioning.

Reference: https://tfhub.dev/google/yamnet/1
"""

import logging
from typing import List, Optional

import numpy as np

from src.models.event import SoundEvent
from src.utils.audio_extractor import AudioExtractor
from config.settings import (
    SOUND_CONFIDENCE_THRESHOLD,
    ANALYSIS_WINDOW_SIZE,
    ANALYSIS_HOP_LENGTH,
    TARGET_SOUND_EVENTS,
    AUDIO_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


# Mapping from YAMNet class names to simplified CC labels
# This mapping converts verbose model labels to concise caption text
CC_LABEL_MAP = {
    "Gunshot, gunfire": "gunshot",
    "Explosion": "explosion",
    "Glass": "glass breaking",
    "Breaking": "breaking",
    "Siren": "siren",
    "Car alarm": "car alarm",
    "Vehicle horn, car horn, honking": "honking",
    "Screaming": "screaming",
    "Crying, sobbing": "crying",
    "Laughter": "laughter",
    "Applause": "applause",
    "Cheering": "cheering",
    "Crowd": "crowd noise",
    "Dog": "dog barking",
    "Thunder": "thunder",
    "Alarm": "alarm",
    "Bell": "bell ringing",
    "Door": "door",
    "Knock": "knocking",
    "Telephone": "phone ringing",
    "Music": "music playing",
    "Singing": "singing",
    "Drum": "drumming",
    "Fire": "fire crackling",
    "Water": "water",
    "Rain": "rain",
    "Wind": "wind",
}


class SoundEventDetector:
    """Detects non-speech audio events using Google's YAMNet model.

    The detector extracts audio from video files, runs the YAMNet model
    on sliding windows, and returns a list of detected SoundEvent objects
    filtered for non-speech events above the confidence threshold.

    Attributes:
        confidence_threshold: Minimum confidence to consider an event.
        sample_rate: Audio sample rate expected by the model.
        target_events: Set of YAMNet class names to detect.

    Example:
        >>> detector = SoundEventDetector(confidence_threshold=0.3)
        >>> events = detector.detect("input_video.mp4")
        >>> for event in events:
        ...     print(f"{event.cc_text} at {event.start_time:.1f}s")
    """

    def __init__(
        self,
        confidence_threshold: float = SOUND_CONFIDENCE_THRESHOLD,
        target_events: Optional[List[str]] = None,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ):
        """Initialize the SoundEventDetector.

        Args:
            confidence_threshold: Minimum confidence score (0.0-1.0) for
                a detected event to be included. Defaults to 0.3.
            target_events: List of YAMNet class names to detect.
                If None, uses the default TARGET_SOUND_EVENTS list.
            sample_rate: Expected audio sample rate in Hz.
        """
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.target_events = set(target_events or TARGET_SOUND_EVENTS)

        self._model = None
        self._class_names = None

        logger.info(
            "SoundEventDetector initialized (threshold=%.2f, targets=%d events)",
            self.confidence_threshold,
            len(self.target_events),
        )

    def _load_model(self):
        """Load the YAMNet model from TensorFlow Hub.

        The model is loaded lazily on first use to avoid loading TensorFlow
        at import time.

        Raises:
            ImportError: If TensorFlow or TensorFlow Hub is not installed.
            RuntimeError: If the model fails to load.
        """
        if self._model is not None:
            return

        logger.info("Loading YAMNet model from TensorFlow Hub...")

        try:
            import tensorflow_hub as hub
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "TensorFlow and TensorFlow Hub are required for sound event "
                "detection. Install them with: pip install tensorflow tensorflow-hub"
            ) from e

        try:
            self._model = hub.load("https://tfhub.dev/google/yamnet/1")
        except Exception as e:
            raise RuntimeError(f"Failed to load YAMNet model: {e}") from e

        # Load class names from the model
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path, "r") as f:
            # Skip header line, parse CSV: index, mid, display_name
            self._class_names = [
                line.strip().split(",")[2]
                for line in f.readlines()[1:]
            ]

        logger.info(
            "YAMNet model loaded successfully (%d classes)",
            len(self._class_names),
        )

    def _get_target_indices(self) -> dict:
        """Get the YAMNet class indices for our target events.

        Returns:
            Dictionary mapping class index to class name for target events.
        """
        self._load_model()
        target_indices = {}

        for idx, name in enumerate(self._class_names):
            if name in self.target_events:
                target_indices[idx] = name

        logger.debug("Found %d target event classes in model", len(target_indices))
        return target_indices

    def detect_from_audio(
        self, audio_waveform: np.ndarray, sample_rate: int = None
    ) -> List[SoundEvent]:
        """Detect sound events from a raw audio waveform.

        This is the core detection method. It runs the YAMNet model on
        the provided audio waveform and returns detected events.

        Args:
            audio_waveform: 1D numpy array of audio samples (float32,
                values between -1.0 and 1.0).
            sample_rate: Sample rate of the audio. Defaults to
                self.sample_rate.

        Returns:
            List of SoundEvent objects sorted by start_time.
        """
        if len(audio_waveform) == 0:
            logger.warning("Empty audio waveform provided")
            return []

        self._load_model()
        sample_rate = sample_rate or self.sample_rate
        target_indices = self._get_target_indices()

        # Ensure float32
        if audio_waveform.dtype != np.float32:
            audio_waveform = audio_waveform.astype(np.float32)

        # Normalize if needed
        max_val = np.max(np.abs(audio_waveform))
        if max_val > 1.0:
            audio_waveform = audio_waveform / max_val

        logger.info(
            "Running YAMNet on audio (%.2f seconds, %d samples)",
            len(audio_waveform) / sample_rate,
            len(audio_waveform),
        )

        # Run YAMNet
        # Returns: scores (frames x classes), embeddings, spectrogram
        scores, embeddings, spectrogram = self._model(audio_waveform)
        scores = scores.numpy()

        # YAMNet processes audio in ~0.96s patches with 0.48s hop
        patch_duration = ANALYSIS_WINDOW_SIZE
        hop_duration = ANALYSIS_HOP_LENGTH

        events = []
        num_frames = scores.shape[0]

        for frame_idx in range(num_frames):
            frame_scores = scores[frame_idx]
            start_time = frame_idx * hop_duration
            end_time = start_time + patch_duration

            # Check each target event category
            for class_idx, class_name in target_indices.items():
                score = float(frame_scores[class_idx])

                if score >= self.confidence_threshold:
                    simplified_label = CC_LABEL_MAP.get(class_name, class_name.lower())

                    event = SoundEvent(
                        label=simplified_label,
                        raw_label=class_name,
                        confidence=score,
                        start_time=round(start_time, 3),
                        end_time=round(end_time, 3),
                    )
                    events.append(event)

        # Merge consecutive events of the same type
        merged_events = self._merge_consecutive_events(events)

        logger.info(
            "Detected %d raw events, merged to %d events",
            len(events),
            len(merged_events),
        )

        return sorted(merged_events, key=lambda e: e.start_time)

    def detect(self, video_path: str) -> List[SoundEvent]:
        """Detect sound events in a video file.

        This is the high-level method that handles the full pipeline:
        video → audio extraction → event detection.

        Args:
            video_path: Path to the input video file.

        Returns:
            List of SoundEvent objects sorted by start_time.
        """
        logger.info("Detecting sound events in: %s", video_path)

        # Extract audio from video
        extractor = AudioExtractor(sample_rate=self.sample_rate)
        audio_path = extractor.extract(video_path, overwrite=True)

        # Load audio waveform
        audio_waveform = self._load_audio(audio_path)

        # Run detection
        return self.detect_from_audio(audio_waveform)

    @staticmethod
    def _load_audio(audio_path: str) -> np.ndarray:
        """Load an audio file as a numpy waveform.

        Args:
            audio_path: Path to the audio file (WAV format).

        Returns:
            1D numpy array of float32 audio samples.
        """
        try:
            import soundfile as sf

            data, _ = sf.read(audio_path, dtype="float32")
            # Convert stereo to mono if needed
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            return data
        except ImportError:
            # Fallback to scipy
            from scipy.io import wavfile

            rate, data = wavfile.read(audio_path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            return data

    @staticmethod
    def _merge_consecutive_events(
        events: List[SoundEvent],
        max_gap: float = 0.5,
    ) -> List[SoundEvent]:
        """Merge consecutive events of the same type.

        If two events with the same label are close together (within
        max_gap seconds), they are merged into a single event with
        the maximum confidence score.

        Args:
            events: List of SoundEvent objects to merge.
            max_gap: Maximum time gap (seconds) between events to merge.

        Returns:
            List of merged SoundEvent objects.
        """
        if not events:
            return []

        # Sort by label then start_time
        sorted_events = sorted(events, key=lambda e: (e.label, e.start_time))

        merged = []
        current = sorted_events[0]

        for next_event in sorted_events[1:]:
            if (
                next_event.label == current.label
                and next_event.start_time - current.end_time <= max_gap
            ):
                # Merge: extend end_time, take max confidence
                current = SoundEvent(
                    label=current.label,
                    raw_label=current.raw_label,
                    confidence=max(current.confidence, next_event.confidence),
                    start_time=current.start_time,
                    end_time=next_event.end_time,
                )
            else:
                merged.append(current)
                current = next_event

        merged.append(current)
        return merged
