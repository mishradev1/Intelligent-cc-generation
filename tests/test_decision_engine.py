"""Unit tests for the CC Decision Engine."""

import pytest
from src.engine.decision_engine import DecisionEngine
from src.models.event import SoundEvent
from src.models.reaction import ReactionEvent, ReactionType
from src.models.cc_suggestion import CCSuggestion


class TestDecisionEngineInit:
    """Tests for DecisionEngine initialization."""

    def test_default_init(self):
        """Should initialize with default weights and threshold."""
        engine = DecisionEngine()
        assert engine.audio_weight == 0.6
        assert engine.visual_weight == 0.4
        assert engine.decision_threshold == 0.5
        assert engine.min_cc_gap == 2.0

    def test_custom_weights(self):
        """Should accept custom weights."""
        engine = DecisionEngine(audio_weight=0.8, visual_weight=0.2)
        assert engine.audio_weight == 0.8
        assert engine.visual_weight == 0.2


class TestCombinedScore:
    """Tests for score computation."""

    def test_weighted_combination(self):
        """Should correctly compute weighted score."""
        engine = DecisionEngine(audio_weight=0.6, visual_weight=0.4)
        score = engine._compute_combined_score(1.0, 1.0)
        assert score == 1.0

    def test_audio_only(self):
        """Should work with zero visual confidence."""
        engine = DecisionEngine(audio_weight=0.6, visual_weight=0.4)
        score = engine._compute_combined_score(0.8, 0.0)
        assert abs(score - 0.48) < 1e-6

    def test_visual_only(self):
        """Should work with zero audio confidence."""
        engine = DecisionEngine(audio_weight=0.6, visual_weight=0.4)
        score = engine._compute_combined_score(0.0, 0.8)
        assert abs(score - 0.32) < 1e-6

    def test_capped_at_one(self):
        """Score should be capped at 1.0."""
        engine = DecisionEngine(audio_weight=0.8, visual_weight=0.8)
        score = engine._compute_combined_score(1.0, 1.0)
        assert score == 1.0


class TestDecide:
    """Tests for the decide method."""

    def _make_event(self, label, confidence, start, end):
        return SoundEvent(
            label=label,
            raw_label=label,
            confidence=confidence,
            start_time=start,
            end_time=end,
        )

    def _make_reaction(self, rtype, confidence, timestamp):
        return ReactionEvent(
            reaction_type=rtype,
            confidence=confidence,
            timestamp=timestamp,
        )

    def test_empty_events(self):
        """Should return empty list for no events."""
        engine = DecisionEngine()
        result = engine.decide([], [])
        assert result == []

    def test_high_confidence_included(self):
        """High confidence events should be included."""
        engine = DecisionEngine(
            audio_weight=0.6, visual_weight=0.4, decision_threshold=0.3
        )
        events = [self._make_event("honking", 0.9, 1.0, 2.0)]
        reactions = [
            self._make_reaction(ReactionType.HEAD_TURN, 0.8, 1.5)
        ]
        result = engine.decide(events, reactions)
        assert len(result) == 1
        assert result[0].include is True
        assert result[0].cc_text == "[honking]"

    def test_low_confidence_excluded(self):
        """Low confidence events should be excluded."""
        engine = DecisionEngine(
            audio_weight=0.6, visual_weight=0.4, decision_threshold=0.8
        )
        events = [self._make_event("wind", 0.3, 5.0, 6.0)]
        reactions = [
            self._make_reaction(ReactionType.NONE, 0.0, 5.5)
        ]
        result = engine.decide(events, reactions)
        assert len(result) == 1
        assert result[0].include is False

    def test_padded_reactions(self):
        """Should handle more events than reactions by padding."""
        engine = DecisionEngine(decision_threshold=0.3)
        events = [
            self._make_event("honking", 0.9, 1.0, 2.0),
            self._make_event("siren", 0.8, 5.0, 6.0),
        ]
        reactions = [
            self._make_reaction(ReactionType.HEAD_TURN, 0.7, 1.5)
        ]
        result = engine.decide(events, reactions)
        assert len(result) == 2

    def test_audio_only_mode(self):
        """Should work in audio-only mode with visual_weight=0."""
        engine = DecisionEngine(
            audio_weight=1.0, visual_weight=0.0, decision_threshold=0.5
        )
        events = [self._make_event("explosion", 0.7, 3.0, 4.0)]
        result = engine.decide(events, [])
        assert len(result) == 1
        assert result[0].include is True
        assert result[0].combined_score >= 0.5


class TestMinGapFiltering:
    """Tests for minimum gap between consecutive CCs."""

    def _make_event(self, label, confidence, start, end):
        return SoundEvent(
            label=label,
            raw_label=label,
            confidence=confidence,
            start_time=start,
            end_time=end,
        )

    def test_gap_filtering_removes_lower_score(self):
        """Should exclude the lower-scoring CC when gap is too small."""
        engine = DecisionEngine(
            audio_weight=1.0, visual_weight=0.0,
            decision_threshold=0.3, min_cc_gap=3.0,
        )
        events = [
            self._make_event("honking", 0.5, 1.0, 2.0),
            self._make_event("siren", 0.9, 2.5, 3.5),  # gap < 3.0
        ]
        result = engine.decide(events, [])

        included = [s for s in result if s.include]
        # Siren has higher score, honking should be excluded
        assert len(included) == 1
        assert included[0].sound_label == "siren"

    def test_no_filtering_when_gap_sufficient(self):
        """Should not filter when gap is sufficient."""
        engine = DecisionEngine(
            audio_weight=1.0, visual_weight=0.0,
            decision_threshold=0.3, min_cc_gap=1.0,
        )
        events = [
            self._make_event("honking", 0.8, 1.0, 2.0),
            self._make_event("siren", 0.7, 10.0, 11.0),
        ]
        result = engine.decide(events, [])
        included = [s for s in result if s.include]
        assert len(included) == 2


class TestCCSuggestionModel:
    """Tests for the CCSuggestion dataclass."""

    def test_basic_creation(self):
        """Should create a valid CCSuggestion."""
        s = CCSuggestion(
            cc_text="[honking]",
            start_time=1.0,
            end_time=3.0,
            audio_confidence=0.9,
            visual_confidence=0.7,
            combined_score=0.82,
            sound_label="honking",
            include=True,
        )
        assert s.duration == 2.0
        assert s.include is True

    def test_invalid_times(self):
        """Should raise ValueError for invalid times."""
        with pytest.raises(ValueError):
            CCSuggestion(
                cc_text="[test]",
                start_time=5.0,
                end_time=3.0,
                audio_confidence=0.5,
                visual_confidence=0.5,
                combined_score=0.5,
                sound_label="test",
                include=True,
            )

    def test_to_dict(self):
        """Should convert to dictionary."""
        s = CCSuggestion(
            cc_text="[laughter]",
            start_time=2.5,
            end_time=4.5,
            audio_confidence=0.8,
            visual_confidence=0.6,
            combined_score=0.72,
            sound_label="laughter",
            include=True,
        )
        d = s.to_dict()
        assert d["cc_text"] == "[laughter]"
        assert d["duration"] == 2.0
        assert d["include"] is True

    def test_repr_included(self):
        """Repr should show checkmark for included."""
        s = CCSuggestion(
            cc_text="[honking]",
            start_time=1.0,
            end_time=2.0,
            audio_confidence=0.9,
            visual_confidence=0.7,
            combined_score=0.82,
            sound_label="honking",
            include=True,
        )
        assert "✓" in repr(s)

    def test_repr_excluded(self):
        """Repr should show X for excluded."""
        s = CCSuggestion(
            cc_text="[wind]",
            start_time=1.0,
            end_time=2.0,
            audio_confidence=0.2,
            visual_confidence=0.0,
            combined_score=0.12,
            sound_label="wind",
            include=False,
        )
        assert "✗" in repr(s)
