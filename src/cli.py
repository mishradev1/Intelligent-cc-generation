"""Command-line interface for the Intelligent CC Suggestion Tool.

Usage:
    python -m src.cli --input video.mp4 --output captions.srt
    python -m src.cli --input video.mp4 --audio-threshold 0.4 --visual-weight 0.3
"""

import argparse
import logging
import sys
import json

from src.detectors.sound_event_detector import SoundEventDetector
from src.detectors.reaction_detector import ReactionDetector
from src.engine.decision_engine import DecisionEngine
from src.output.srt_generator import SRTGenerator
from config.settings import (
    SOUND_CONFIDENCE_THRESHOLD,
    AUDIO_WEIGHT,
    VISUAL_WEIGHT,
    CC_DECISION_THRESHOLD,
    MIN_CC_GAP,
)


def setup_logging(verbose: bool = False):
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(args=None):
    """Parse command-line arguments.

    Args:
        args: Optional list of arguments (for testing).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="cc-suggest",
        description=(
            "Intelligent CC Suggestion Tool — Analyze videos to generate "
            "contextually relevant closed caption annotations for non-speech "
            "audio events."
        ),
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--output", "-o",
        default="captions.srt",
        help="Output SRT filename (default: captions.srt)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--audio-threshold",
        type=float,
        default=SOUND_CONFIDENCE_THRESHOLD,
        help=f"Audio event confidence threshold (default: {SOUND_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--audio-weight",
        type=float,
        default=AUDIO_WEIGHT,
        help=f"Weight for audio confidence in decision (default: {AUDIO_WEIGHT})",
    )
    parser.add_argument(
        "--visual-weight",
        type=float,
        default=VISUAL_WEIGHT,
        help=f"Weight for visual confidence in decision (default: {VISUAL_WEIGHT})",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=CC_DECISION_THRESHOLD,
        help=f"Combined score threshold for CC inclusion (default: {CC_DECISION_THRESHOLD})",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=MIN_CC_GAP,
        help=f"Minimum gap (seconds) between CCs (default: {MIN_CC_GAP})",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Skip visual reaction detection (faster, audio-only mode)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output detailed results as JSON alongside SRT",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )

    return parser.parse_args(args)


def main(args=None):
    """Main entry point for the CC suggestion pipeline.

    Args:
        args: Optional list of arguments (for testing).
    """
    parsed = parse_args(args)
    setup_logging(parsed.verbose)

    logger = logging.getLogger("cc-suggest")
    logger.info("=" * 60)
    logger.info("Intelligent CC Suggestion Tool")
    logger.info("=" * 60)
    logger.info("Input: %s", parsed.input)
    logger.info("Output: %s/%s", parsed.output_dir, parsed.output)

    # Step 1: Sound Event Detection
    logger.info("")
    logger.info("Step 1: Detecting non-speech audio events...")
    sound_detector = SoundEventDetector(
        confidence_threshold=parsed.audio_threshold
    )
    sound_events = sound_detector.detect(parsed.input)
    logger.info("Detected %d sound events", len(sound_events))

    for event in sound_events:
        logger.info(
            "  %s (confidence=%.2f, %.2fs-%.2fs)",
            event.cc_text,
            event.confidence,
            event.start_time,
            event.end_time,
        )

    # Step 2: Visual Reaction Detection (optional)
    reactions = []
    if not parsed.audio_only and sound_events:
        logger.info("")
        logger.info("Step 2: Detecting visual reactions...")
        reaction_detector = ReactionDetector()
        reactions = reaction_detector.detect(parsed.input, sound_events)
        significant = sum(1 for r in reactions if r.is_significant)
        logger.info(
            "Detected %d significant reactions out of %d events",
            significant,
            len(sound_events),
        )
    elif parsed.audio_only:
        logger.info("")
        logger.info("Step 2: Skipped (audio-only mode)")

    # Step 3: CC Decision Engine
    logger.info("")
    logger.info("Step 3: Making CC decisions...")
    engine = DecisionEngine(
        audio_weight=parsed.audio_weight if not parsed.audio_only else 1.0,
        visual_weight=parsed.visual_weight if not parsed.audio_only else 0.0,
        decision_threshold=parsed.decision_threshold,
        min_cc_gap=parsed.min_gap,
    )
    suggestions = engine.decide(sound_events, reactions)

    included = [s for s in suggestions if s.include]
    logger.info(
        "Result: %d/%d events accepted as CC annotations",
        len(included),
        len(suggestions),
    )

    # Step 4: Generate SRT Output
    logger.info("")
    logger.info("Step 4: Generating SRT file...")
    generator = SRTGenerator(output_dir=parsed.output_dir)
    output_path = generator.generate(suggestions, parsed.output)
    logger.info("SRT file saved to: %s", output_path)

    # Optional JSON output
    if parsed.json:
        json_filename = parsed.output.replace(".srt", ".json")
        json_path = f"{parsed.output_dir}/{json_filename}"
        results = {
            "input": parsed.input,
            "sound_events": [e.to_dict() for e in sound_events],
            "reactions": [r.to_dict() for r in reactions],
            "suggestions": [s.to_dict() for s in suggestions],
            "summary": {
                "total_events": len(sound_events),
                "total_reactions": len(reactions),
                "accepted_ccs": len(included),
                "rejected_ccs": len(suggestions) - len(included),
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("JSON results saved to: %s", json_path)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info("Sound events detected: %d", len(sound_events))
    logger.info("Visual reactions found: %d", len(reactions))
    logger.info("CC annotations generated: %d", len(included))
    logger.info("Output file: %s", output_path)

    if included:
        logger.info("")
        logger.info("Generated captions:")
        for s in included:
            logger.info(
                "  %s  %.2fs - %.2fs  (score: %.2f)",
                s.cc_text,
                s.start_time,
                s.end_time,
                s.combined_score,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
