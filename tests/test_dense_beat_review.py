from pathlib import Path

import pytest

from youtube_pipeline.config import BeatConfig
from youtube_pipeline.dense_beat_review import build_dense_beat_review
from youtube_pipeline.errors import PipelineError
from youtube_pipeline.models import Beat, TranscriptSegment
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


def beat(tmp_path: Path, number: int, start: float, end: float, indexes: list[int], preview: str = "Preview.") -> Beat:
    return Beat(
        beat_number=number,
        beat_type="normal",
        start=seconds_to_timestamp(start),
        end=seconds_to_timestamp(end),
        start_seconds=start,
        end_seconds=end,
        duration_seconds=end - start,
        text_preview=preview,
        segment_indexes=indexes,
        image_path=(tmp_path / "assets" / "images" / f"beat_{number:03}.png").as_posix(),
    )


def dense_plan(records: list[dict]) -> dict:
    return {
        "planner_version": 1,
        "planner_strategy": "hybrid_dense_rebuild",
        "warnings": [],
        "dense_group_records": records,
    }


def record(number: int, boundary: str, source_standard: int | None = 1, reason: str = "dense_rebuild_preferred_boundary") -> dict:
    return {
        "dense_beat_number": number,
        "beat_type": "normal",
        "source_standard_beat_number": source_standard,
        "start_seconds": float(number),
        "end_seconds": float(number + 6),
        "duration_seconds": 6.0,
        "segment_indexes": [number],
        "grouping_score": 0.0,
        "boundary_confidence": boundary,
        "reason": reason,
    }


def test_dense_review_report_shape_and_classifications(tmp_path):
    beats = [
        beat(tmp_path, 1, 0, 6.5, [1]),
        beat(tmp_path, 2, 6.5, 13.0, [2]),
        beat(tmp_path, 3, 13.0, 19.0, [3]),
        beat(tmp_path, 4, 19.0, 25.0, [4]),
        beat(tmp_path, 5, 25.0, 29.0, [5]),
    ]
    segments = [
        segment(1, 0, 6.5, "A complete symbolic image arrives clearly."),
        segment(2, 6.5, 13.0, "This idea ends on a weak boundary"),
        segment(3, 13.0, 19.0, "So time became money in the modern world."),
        segment(4, 19.0, 25.0, "Time"),
        segment(5, 25.0, 29.0, "123 456"),
    ]
    plan = dense_plan(
        [
            record(1, "high", 1),
            record(2, "weak", 2),
            record(3, "high", 3),
            record(4, "high", 4),
            record(5, "high", 5, "dense_rebuild_final_tail"),
        ]
    )

    report = build_dense_beat_review(beats, plan, segments, BEAT_CONFIG, base_dir=tmp_path)

    assert report["review_schema_version"] == 1
    assert report["planner_strategy"] == "hybrid_dense_rebuild"
    assert report["total_dense_beats"] == 5
    assert set(report["recommendation_counts"]) == {"approve", "review", "risky", "blocked"}
    assert report["readiness"] == "not_ready"

    rows = {row["dense_beat_number"]: row for row in report["beats"]}
    assert rows[1]["recommendation"] == "approve"
    assert rows[1]["source_coherence_label"] == "coherent"
    assert rows[2]["recommendation"] == "review"
    assert rows[2]["source_coherence_label"] == "fragmented"
    assert "WEAK_PUNCTUATION_BOUNDARY" in rows[2]["warning_codes"]
    assert rows[3]["source_coherence_label"] == "minor_boundary_issue"
    assert rows[4]["recommendation"] == "risky"
    assert rows[4]["source_coherence_label"] == "not_promptable"
    assert rows[5]["recommendation"] == "blocked"
    assert "NO_MEANINGFUL_SOURCE_TEXT" in rows[5]["warning_codes"]


def test_problem_beats_are_prioritized_before_approved_beats(tmp_path):
    beats = [
        beat(tmp_path, 1, 0, 6, [1]),
        beat(tmp_path, 2, 6, 12, [2]),
        beat(tmp_path, 3, 12, 18, [3]),
    ]
    segments = [
        segment(1, 0, 6, "A complete useful visual idea arrives clearly."),
        segment(2, 6, 12, "Time"),
        segment(3, 12, 18, "123"),
    ]
    plan = dense_plan([record(1, "high"), record(2, "high"), record(3, "high")])

    report = build_dense_beat_review(beats, plan, segments, BEAT_CONFIG, base_dir=tmp_path)

    assert [row["recommendation"] for row in report["top_problem_beats"]] == ["blocked", "risky"]
    assert [row["dense_beat_number"] for row in report["top_problem_beats"]] == [3, 2]


def test_invalid_dense_plan_structure_is_rejected(tmp_path):
    beats = [beat(tmp_path, 1, 0, 6, [1])]
    segments = [segment(1, 0, 6, "A complete useful visual idea.")]

    with pytest.raises(PipelineError, match="dense_group_records"):
        build_dense_beat_review(beats, {"warnings": []}, segments, BEAT_CONFIG, base_dir=tmp_path)


def test_missing_dense_group_record_is_reported_as_risky(tmp_path):
    beats = [beat(tmp_path, 1, 0, 6, [1])]
    segments = [segment(1, 0, 6, "A complete useful visual idea.")]
    plan = dense_plan([])

    report = build_dense_beat_review(beats, plan, segments, BEAT_CONFIG, base_dir=tmp_path)

    row = report["beats"][0]
    assert row["recommendation"] == "risky"
    assert row["boundary_confidence"] == "unknown"
    assert "MISSING_DENSE_GROUP_RECORD" in row["warning_codes"]
