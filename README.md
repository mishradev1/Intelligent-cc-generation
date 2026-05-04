# Intelligent Closed Caption (CC) Suggestion Tool

An AI-powered tool that intelligently identifies moments in a video where a Closed Caption (CC) annotation is genuinely necessary — such as when a non-speech audio event meaningfully affects the speakers or the scene — and suggests contextually relevant CC text, without over-captioning routine or low-impact sounds.

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│  Video File  │───▶│ Audio Extractor  │───▶│ Sound Event Detector │
│  (input)     │    │ (ffmpeg/moviepy) │    │ (YAMNet)             │
└──────┬───────┘    └──────────────────┘    └──────────┬───────────┘
       │                                               │
       │            ┌──────────────────┐               │
       └───────────▶│ Frame Extractor  │               │
                    │ (OpenCV)         │               │
                    └────────┬─────────┘               │
                             │                         │
                    ┌────────▼─────────┐               │
                    │ Reaction Detector│               │
                    │ (MediaPipe)      │               │
                    └────────┬─────────┘               │
                             │                         │
                    ┌────────▼─────────────────────────▼┐
                    │      CC Decision Engine            │
                    │  Combines audio + visual signals   │
                    └────────────────┬───────────────────┘
                                     │
                            ┌────────▼────────┐
                            │  SRT Generator  │
                            └────────┬────────┘
                                     │
                            ┌────────▼────────┐
                            │  output.srt     │
                            └─────────────────┘
```

## Features

- **Sound Event Detection** — Automatically detects and classifies non-speech audio events (honking, explosions, laughter, music, alarms, applause, etc.) with confidence scores and timestamps using YAMNet.
- **Speaker Reaction Detection** — Analyzes video frames at detected event timestamps using MediaPipe to identify visible reactions (head turns, startled body language, facial expressions).
- **Intelligent CC Decisions** — Combines audio and visual signals to determine whether a CC annotation is truly warranted, avoiding over-captioning of ambient sounds.
- **SRT Output** — Generates standard SRT subtitle files with properly formatted timestamps and descriptive CC labels like `[honking]`, `[crowd cheering]`, `[gunshot]`.

## Prerequisites

- **Python 3.9+**
- **FFmpeg** — Must be installed and available on your system PATH
  - Windows: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PlanetRead/Intelligent-cc-generation.git
   cd Intelligent-cc-generation
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install in development mode** (optional)
   ```bash
   pip install -e .
   ```

## 🎯 Usage

### Extract audio from a video file
```python
from src.utils.audio_extractor import AudioExtractor

extractor = AudioExtractor()
audio_path = extractor.extract("input_video.mp4")
print(f"Audio saved to: {audio_path}")
```

### Full pipeline (coming soon)
```bash
python -m src.cli --input video.mp4 --output captions.srt
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
Intelligent-cc-generation/
├── src/
│   ├── __init__.py
│   ├── cli.py                     # CLI entry point
│   ├── utils/
│   │   ├── __init__.py
│   │   └── audio_extractor.py     # Video → Audio extraction
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── sound_event_detector.py  # YAMNet-based audio analysis
│   │   └── reaction_detector.py     # MediaPipe-based visual analysis
│   ├── models/
│   │   ├── __init__.py
│   │   ├── event.py               # SoundEvent dataclass
│   │   ├── reaction.py            # ReactionEvent dataclass
│   │   └── cc_suggestion.py       # CCSuggestion dataclass
│   ├── engine/
│   │   ├── __init__.py
│   │   └── decision_engine.py     # CC decision combiner
│   └── output/
│       ├── __init__.py
│       └── srt_generator.py       # SRT file writer
├── config/
│   └── settings.py                # Configuration defaults
├── tests/
│   ├── __init__.py
│   ├── test_audio_extractor.py
│   └── fixtures/
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Audio Event Detection | [YAMNet](https://tfhub.dev/google/yamnet/1) (TensorFlow Hub) |
| Frame Extraction | [OpenCV](https://opencv.org/) |
| Pose & Expression Analysis | [MediaPipe](https://mediapipe.dev/) |
| Audio Extraction | [FFmpeg](https://ffmpeg.org/) via moviepy |
| Output Format | SRT (SubRip Subtitle) |


## License

This project is part of the [Planet Read](https://www.planetread.org/) initiative under the DMP 2026 program.
