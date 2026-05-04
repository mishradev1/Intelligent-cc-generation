"""CC Decision Engine for combining audio and visual signals.

This module provides the DecisionEngine class that combines audio event
confidence scores with visual reaction confidence scores to decide
whether a CC annotation is warranted for each detected event.
"""

import logging
from typing import List, Tuple

from src.models.event import SoundEvent
from src.models.reaction import ReactionEvent, ReactionType
from src.models.cc_suggestion import CCSuggestion
from config.settings import (
    AUDIO_WEIGHT,
    VISUAL_WEIGHT,
    CC_DECISION_THRESHOLD,
    MIN_CC_GAP,
    DEFAULT_CC_DURATION,
)

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Combines audio and visual signals to make CC decisions.

    The engine uses a weighted combination of audio event confidence
    and visual reaction confidence to determine whether a closed
    caption should be generated for each detected event.

    The formula is:
        combined_score = audio_confidence × audio_weight
                       + visual_confidence × visual_weight

    Events with a combined_score above the decision threshold are
    accepted as CC suggestions.

    Attributes:
        audio_weight: Weight for audio confidence (0.0-1.0).
        visual_weight: Weight for visual confidence (0.0-1.0).
        decision_threshold: Minimum combined score to include a CC.
        min_cc_gap: Minimum seconds between consecutive CC annotations.

    Example:
        >>> engine = DecisionEngine(audio_weight=0.6, visual_weight=0.4)
        >>> suggestions = engine.decide(sound_events, reactions)
        >>> for s in suggestions:
        ...     if s.include:
        ...         print(f"{s.cc_text} at {s.start_time}s")
    """

    def __init__(
        self,
        audio_weight: float = AUDIO_WEIGHT,
        visual_weight: float = VISUAL_WEIGHT,
        decision_threshold: float = CC_DECISION_THRESHOLD,
        min_cc_gap: float = MIN_CC_GAP,
    ):
        """Initialize the DecisionEngine.

        Args:
            audio_weight: Weight for audio event confidence.
            visual_weight: Weight for visual reaction confidence.
            decision_threshold: Minimum combined score for CC inclusion.
            min_cc_gap: Minimum seconds between consecutive CCs.
        """
        self.audio_weight = audio_weight
        self.visual_weight = visual_weight
        self.decision_threshold = decision_threshold
        self.min_cc_gap = min_cc_gap

        logger.info(
            "DecisionEngine initialized (audio_w=%.2f, visual_w=%.2f, "
            "threshold=%.2f, min_gap=%.1fs)",
            self.audio_weight,
            self.visual_weight,
            self.decision_threshold,
            self.min_cc_gap,
        )

    def _compute_combined_score(
        self,
        audio_confidence: float,
        visual_confidence: float,
    ) -> float:
        """Compute the weighted combined score.

        Args:
            audio_confidence: Audio event confidence (0.0-1.0).
            visual_confidence: Visual reaction confidence (0.0-1.0).

        Returns:
            Combined score between 0.0 and 1.0.
        """
        score = (
            audio_confidence * self.audio_weight
            + visual_confidence * self.visual_weight
        )
        return min(score, 1.0)

    def decide(
        self,
        sound_events: List[SoundEvent],
        reactions: List[ReactionEvent],
    ) -> List[CCSuggestion]:
        """Make CC decisions for paired audio events and visual reactions.

        Each sound event is paired with its corresponding reaction (by
        index). If there are more sound events than reactions, missing
        reactions are treated as having zero confidence.

        Args:
            sound_events: List of detected audio events.
            reactions: List of visual reactions, one per sound event.

        Returns:
            List of CCSuggestion objects, one per sound event, indicating
            whether each should be included in the output.
        """
        if not sound_events:
            logger.info("No sound events to process")
            return []

        # Pad reactions list if shorter than sound_events
        padded_reactions = list(reactions)
        while len(padded_reactions) < len(sound_events):
            padded_reactions.append(
                ReactionEvent(
                    reaction_type=ReactionType.NONE,
                    confidence=0.0,
                    timestamp=0.0,
                )
            )

        suggestions = []

        for event, reaction in zip(sound_events, padded_reactions):
            visual_conf = reaction.confidence if reaction.is_significant else 0.0
            combined = self._compute_combined_score(event.confidence, visual_conf)

            # Determine CC display duration
            event_duration = event.duration
            if event_duration < DEFAULT_CC_DURATION:
                end_time = event.start_time + DEFAULT_CC_DURATION
            else:
                end_time = event.end_time

            suggestion = CCSuggestion(
                cc_text=event.cc_text,
                start_time=event.start_time,
                end_time=end_time,
                audio_confidence=event.confidence,
                visual_confidence=visual_conf,
                combined_score=round(combined, 4),
                sound_label=event.label,
                include=combined >= self.decision_threshold,
            )
            suggestions.append(suggestion)

        # Apply minimum gap filtering
        filtered = self._apply_min_gap(suggestions)

        included = sum(1 for s in filtered if s.include)
        excluded = sum(1 for s in filtered if not s.include)
        logger.info(
            "CC decisions: %d included, %d excluded (total: %d events)",
            included,
            excluded,
            len(filtered),
        )

        return filtered

    def _apply_min_gap(
        self, suggestions: List[CCSuggestion]
    ) -> List[CCSuggestion]:
        """Apply minimum gap filtering between consecutive CCs.

        If two included CCs are too close together, the one with the
        lower combined score is excluded.

        Args:
            suggestions: List of CCSuggestion objects.

        Returns:
            Updated list with gap filtering applied.
        """
        if len(suggestions) <= 1:
            return suggestions

        # Sort included suggestions by start_time
        included = [
            (i, s) for i, s in enumerate(suggestions) if s.include
        ]
        included.sort(key=lambda x: x[1].start_time)

        to_exclude = set()

        for j in range(1, len(included)):
            prev_idx, prev_s = included[j - 1]
            curr_idx, curr_s = included[j]

            gap = curr_s.start_time - prev_s.end_time
            if gap < self.min_cc_gap:
                # Exclude the one with lower score
                if prev_s.combined_score < curr_s.combined_score:
                    to_exclude.add(prev_idx)
                else:
                    to_exclude.add(curr_idx)

        # Create updated list
        result = []
        for i, s in enumerate(suggestions):
            if i in to_exclude:
                result.append(
                    CCSuggestion(
                        cc_text=s.cc_text,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        audio_confidence=s.audio_confidence,
                        visual_confidence=s.visual_confidence,
                        combined_score=s.combined_score,
                        sound_label=s.sound_label,
                        include=False,
                    )
                )
            else:
                result.append(s)

        return result
