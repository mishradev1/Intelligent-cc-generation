"""Configuration settings for the Intelligent CC Suggestion Tool."""

import os


# =============================================================================
# Audio Extraction Settings
# =============================================================================

# Default audio sample rate for extracted audio (Hz)
AUDIO_SAMPLE_RATE = 16000

# Default audio format for extracted files
AUDIO_FORMAT = "wav"

# Default output directory for extracted audio files
AUDIO_OUTPUT_DIR = os.path.join(os.getcwd(), "output", "audio")


# =============================================================================
# Sound Event Detection Settings
# =============================================================================

# Minimum confidence threshold for a sound event to be considered
SOUND_CONFIDENCE_THRESHOLD = 0.3

# Analysis window size in seconds for the sound event detector
ANALYSIS_WINDOW_SIZE = 0.96  # YAMNet default patch size

# Hop length between analysis windows in seconds
ANALYSIS_HOP_LENGTH = 0.48

# Non-speech event categories to detect (YAMNet class names)
# Full list: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
TARGET_SOUND_EVENTS = [
    "Gunshot, gunfire",
    "Explosion",
    "Glass",
    "Breaking",
    "Siren",
    "Car alarm",
    "Vehicle horn, car horn, honking",
    "Screaming",
    "Crying, sobbing",
    "Laughter",
    "Applause",
    "Cheering",
    "Crowd",
    "Dog",
    "Thunder",
    "Alarm",
    "Bell",
    "Door",
    "Knock",
    "Telephone",
    "Music",
    "Singing",
    "Drum",
    "Fire",
    "Water",
    "Rain",
    "Wind",
]


# =============================================================================
# Reaction Detection Settings
# =============================================================================

# Number of frames to extract around each event timestamp
REACTION_FRAME_COUNT = 10

# Time window (seconds) before and after event to look for reactions
REACTION_TIME_WINDOW = 1.5

# Minimum confidence for a reaction to be considered significant
REACTION_CONFIDENCE_THRESHOLD = 0.4

# Head turn angle threshold (degrees) to consider as a reaction
HEAD_TURN_THRESHOLD = 15.0

# Pose change threshold (normalized) for startled body language
POSE_CHANGE_THRESHOLD = 0.1


# =============================================================================
# CC Decision Engine Settings
# =============================================================================

# Weight for audio event confidence in the final decision
AUDIO_WEIGHT = 0.6

# Weight for visual reaction confidence in the final decision
VISUAL_WEIGHT = 0.4

# Combined confidence threshold for generating a CC annotation
CC_DECISION_THRESHOLD = 0.5

# Minimum duration (seconds) between consecutive CC annotations
# to avoid overwhelming the viewer
MIN_CC_GAP = 2.0


# =============================================================================
# Output Settings
# =============================================================================

# Default output format
OUTPUT_FORMAT = "srt"

# Default output directory for generated subtitle files
OUTPUT_DIR = os.path.join(os.getcwd(), "output")

# Default CC display duration (seconds) if not determined by event duration
DEFAULT_CC_DURATION = 2.0
