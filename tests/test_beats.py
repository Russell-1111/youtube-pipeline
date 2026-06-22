import pytest

from youtube_pipeline.beats import build_beats
from youtube_pipeline.config import BeatConfig
from youtube_pipeline.errors import TimingError
from youtube_pipeline.models import TranscriptSegment
from youtube_pipeline.time_utils import seconds_to_timestamp


BEAT_CONFIG = BeatConfig(
    min_duration=6,
    target_duration=9,
    max_duration=12,
    min_gap_beat_duration=1.5,
    min_intro_beat_duration=1.0,
    max_preview_chars=80,
)


def segment(index: int, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        index=index,
        start=seconds_to_timestamp(start),
        end=seconds_to_timestamp(end),
        start_seconds=start,
        end_seconds=end,
        duration_seconds=end - start,
        text=text,
    )


def test_beat_grouping_into_duration_windows(tmp_path):
    beats = build_beats(
        [
            segment(1, 0, 4, "Opening"),
            segment(2, 4, 8, "Middle"),
            segment(3, 8, 14, "Next idea"),
        ],
        BEAT_CONFIG,
        tmp_path,
    )

    assert [beat.beat_type for beat in beats] == ["normal", "normal"]
    assert beats[0].segment_indexes == [1, 2]
    assert beats[0].duration_seconds == 8
    assert beats[1].segment_indexes == [3]


def test_long_single_segment_becomes_one_beat(tmp_path):
    beats = build_beats([segment(1, 0, 14, "Long caption")], BEAT_CONFIG, tmp_path)

    assert len(beats) == 1
    assert beats[0].beat_type == "normal"
    assert beats[0].duration_seconds == 14
    assert beats[0].segment_indexes == [1]


def test_transcript_start_after_zero_creates_intro_beat(tmp_path):
    beats = build_beats([segment(1, 2, 8, "Starts later")], BEAT_CONFIG, tmp_path)

    assert beats[0].beat_type == "intro"
    assert beats[0].segment_indexes == []
    assert beats[0].text_preview == "Intro / silence"
    assert beats[0].start_seconds == 0.0
    assert beats[0].end_seconds == 2
    assert beats[1].beat_type == "normal"


def test_short_intro_below_threshold_is_absorbed_into_first_beat(tmp_path):
    beats = build_beats([segment(1, 0.2, 8, "Starts almost immediately")], BEAT_CONFIG, tmp_path)

    assert [beat.beat_type for beat in beats] == ["normal"]
    assert beats[0].start_seconds == 0.0
    assert beats[0].segment_indexes == [1]


def test_large_gap_between_segments_creates_gap_beat(tmp_path):
    beats = build_beats(
        [
            segment(1, 0, 5, "Before pause"),
            segment(2, 15, 20, "After pause"),
        ],
        BEAT_CONFIG,
        tmp_path,
    )

    assert [beat.beat_type for beat in beats] == ["normal", "gap", "normal"]
    assert beats[1].segment_indexes == []
    assert beats[1].text_preview == "Pause / silence"
    assert beats[1].start_seconds == 5
    assert beats[1].end_seconds == 15


def test_short_gap_below_threshold_extends_previous_visual_beat(tmp_path):
    beats = build_beats(
        [
            segment(1, 0, 6, "Before short pause"),
            segment(2, 6.8, 14, "After short pause"),
        ],
        BEAT_CONFIG,
        tmp_path,
    )

    assert [beat.beat_type for beat in beats] == ["normal", "normal"]
    assert beats[0].end_seconds == pytest.approx(6.8)
    assert beats[0].segment_indexes == [1]
    assert beats[1].start_seconds == pytest.approx(6.8)
    assert beats[1].segment_indexes == [2]


def test_turboscribe_watermark_is_removed_from_preview(tmp_path):
    beats = build_beats(
        [
            segment(
                1,
                0,
                6,
                "(Transcribed by TurboScribe. Go Unlimited to remove this message.) Actual narration starts here.",
            )
        ],
        BEAT_CONFIG,
        tmp_path,
    )

    assert "TurboScribe" not in beats[0].text_preview
    assert "Go Unlimited" not in beats[0].text_preview
    assert beats[0].text_preview == "Actual narration starts here."


def test_preview_text_is_capped_at_word_boundary(tmp_path):
    config = BeatConfig(
        min_duration=6,
        target_duration=9,
        max_duration=12,
        min_gap_beat_duration=1.5,
        min_intro_beat_duration=1.0,
        max_preview_chars=40,
    )
    beats = build_beats(
        [segment(1, 0, 6, "This preview should be capped cleanly without cutting a word in half.")],
        config,
        tmp_path,
    )

    assert len(beats[0].text_preview) <= 40
    assert beats[0].text_preview == "This preview should be capped..."


def test_final_beat_extends_when_audio_is_slightly_longer(tmp_path):
    beats = build_beats(
        [segment(1, 0, 5, "Short transcript")],
        BEAT_CONFIG,
        tmp_path,
        audio_duration=5.8,
        duration_mismatch_tolerance=1.0,
    )

    assert beats[-1].end_seconds == pytest.approx(5.8)
    assert beats[-1].duration_seconds == pytest.approx(5.8)


def test_duration_mismatch_exceeding_tolerance_is_rejected(tmp_path):
    with pytest.raises(TimingError, match="mismatch exceeds tolerance"):
        build_beats(
            [segment(1, 0, 5, "Short transcript")],
            BEAT_CONFIG,
            tmp_path,
            audio_duration=8.0,
            duration_mismatch_tolerance=1.0,
        )
