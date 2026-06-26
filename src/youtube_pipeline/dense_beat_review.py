from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .beat_io import load_beats, load_transcript_segments
from .beats import _clean_preview_text, _preview
from .config import BeatConfig, PipelineConfig
from .errors import InputFileError, PipelineError
from .models import Beat, TranscriptSegment

REVIEW_SCHEMA_VERSION = 1
JSON_REPORT_NAME = "dense_beat_review.json"
MARKDOWN_REPORT_NAME = "dense_beat_review.md"
PREVIEW_BEATS_NAME = "beats_dense_preview.json"
DENSE_PLAN_NAME = "dense_beat_plan.json"

RECOMMENDATIONS = ("approve", "review", "risky", "blocked")
RECOMMENDATION_SEVERITY = {
    "blocked": 0,
    "risky": 1,
    "review": 2,
    "approve": 3,
}
CONTINUATION_WORDS = {
    "and",
    "but",
    "because",
    "that",
    "while",
    "so",
}
TERMINAL_PUNCTUATION = {".", "!", "?", '"', "'", ";"}
WEAK_BOUNDARIES = {"medium", "weak"}


@dataclass(frozen=True)
class DenseBeatReviewResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class ReviewScore:
    recommendation: str
    priority: int
    coherence_label: str
    reasons: list[str]


def run_dense_beat_review(config: PipelineConfig, base_dir: Path) -> DenseBeatReviewResult:
    data_dir = config.outputs.data_dir
    preview_path = data_dir / PREVIEW_BEATS_NAME
    dense_plan_path = data_dir / DENSE_PLAN_NAME
    transcript_path = data_dir / "transcript_segments.json"
    json_path = data_dir / JSON_REPORT_NAME
    markdown_path = data_dir / MARKDOWN_REPORT_NAME

    preview_beats = load_beats(preview_path)
    dense_plan = _load_dense_plan(dense_plan_path)
    segments = load_transcript_segments(transcript_path)

    report = build_dense_beat_review(
        preview_beats=preview_beats,
        dense_plan=dense_plan,
        segments=segments,
        beat_config=config.beats,
        preview_path=preview_path,
        dense_plan_path=dense_plan_path,
        transcript_path=transcript_path,
        json_path=json_path,
        markdown_path=markdown_path,
        base_dir=base_dir,
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return DenseBeatReviewResult(report=report, json_path=json_path, markdown_path=markdown_path)


def build_dense_beat_review(
    preview_beats: list[Beat],
    dense_plan: dict[str, Any],
    segments: list[TranscriptSegment],
    beat_config: BeatConfig,
    preview_path: Path = Path("data/beats_dense_preview.json"),
    dense_plan_path: Path = Path("data/dense_beat_plan.json"),
    transcript_path: Path = Path("data/transcript_segments.json"),
    json_path: Path = Path("data/dense_beat_review.json"),
    markdown_path: Path = Path("data/dense_beat_review.md"),
    base_dir: Path | None = None,
) -> dict[str, Any]:
    if not preview_beats:
        raise PipelineError("Cannot review dense beats without preview beats.")
    if not segments:
        raise PipelineError("Cannot review dense beats without transcript segments.")

    records = _dense_group_records(dense_plan)
    warning_examples = _warning_examples_by_beat(dense_plan)
    record_by_number = _record_lookup(records)
    segment_by_index = {segment.index: segment for segment in segments}

    rows = [
        _review_row(
            beat=beat,
            record=record_by_number.get(beat.beat_number),
            warning_examples=warning_examples.get(beat.beat_number, []),
            segment_by_index=segment_by_index,
            beat_config=beat_config,
        )
        for beat in preview_beats
    ]
    counts = {recommendation: 0 for recommendation in RECOMMENDATIONS}
    counts.update(Counter(row["recommendation"] for row in rows))
    problem_rows = [row for row in rows if row["recommendation"] != "approve"]
    top_problem_beats = sorted(
        problem_rows,
        key=lambda row: (
            RECOMMENDATION_SEVERITY[row["recommendation"]],
            -row["review_priority"],
            row["dense_beat_number"],
        ),
    )[:10]
    readiness, readiness_reasons = _readiness(counts)
    base = base_dir or Path.cwd()

    return {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source_preview_beats_path": _display_path(preview_path, base),
        "source_dense_plan_path": _display_path(dense_plan_path, base),
        "source_transcript_segments_path": _display_path(transcript_path, base),
        "review_json_path": _display_path(json_path, base),
        "review_markdown_path": _display_path(markdown_path, base),
        "planner_version": dense_plan.get("planner_version"),
        "planner_strategy": dense_plan.get("planner_strategy"),
        "total_dense_beats": len(preview_beats),
        "recommendation_counts": counts,
        "readiness": readiness,
        "readiness_reasons": readiness_reasons,
        "top_problem_beats": top_problem_beats,
        "beats": rows,
    }


def print_dense_review_summary(result: DenseBeatReviewResult, base_dir: Path) -> None:
    report = result.report
    counts = report["recommendation_counts"]
    print("Dense beat review complete")
    print(f"Total dense beats: {report['total_dense_beats']}")
    print(
        "Recommendations: "
        f"{counts['approve']} approve, {counts['review']} review, "
        f"{counts['risky']} risky, {counts['blocked']} blocked"
    )
    print(f"Readiness: {report['readiness']}")
    print("Reports written:")
    print(f"- {_display_path(result.json_path, base_dir)}")
    print(f"- {_display_path(result.markdown_path, base_dir)}")


def _load_dense_plan(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise InputFileError(f"Missing dense beat plan file: {path}. Run: python -m youtube_pipeline --plan-dense-beats")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid dense beat plan JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise PipelineError(f"Could not read dense beat plan JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"Invalid dense beat plan JSON: expected an object in {path}")
    return payload


def _dense_group_records(dense_plan: dict[str, Any]) -> list[dict[str, Any]]:
    records = dense_plan.get("dense_group_records")
    if not isinstance(records, list):
        raise PipelineError("Invalid dense beat plan JSON: expected dense_group_records list.")
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise PipelineError(f"Invalid dense group record at index {index}: expected an object.")
        if not isinstance(record.get("dense_beat_number"), int):
            raise PipelineError(f"Invalid dense group record at index {index}: missing dense_beat_number.")
    return records


def _record_lookup(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup = {}
    for record in records:
        beat_number = record["dense_beat_number"]
        if beat_number in lookup:
            raise PipelineError(f"Invalid dense beat plan JSON: duplicate dense beat record {beat_number}.")
        lookup[beat_number] = record
    return lookup


def _warning_examples_by_beat(dense_plan: dict[str, Any]) -> dict[int, list[str]]:
    warnings = dense_plan.get("warnings", [])
    if not isinstance(warnings, list):
        raise PipelineError("Invalid dense beat plan JSON: expected warnings list.")
    by_beat: dict[int, list[str]] = {}
    for warning in warnings:
        if not isinstance(warning, dict):
            continue
        code = warning.get("code")
        examples = warning.get("examples", [])
        if not isinstance(code, str) or not isinstance(examples, list):
            continue
        for example in examples:
            if not isinstance(example, dict) or not isinstance(example.get("beat_number"), int):
                continue
            by_beat.setdefault(example["beat_number"], []).append(code)
    return by_beat


def _review_row(
    beat: Beat,
    record: dict[str, Any] | None,
    warning_examples: list[str],
    segment_by_index: dict[int, TranscriptSegment],
    beat_config: BeatConfig,
) -> dict[str, Any]:
    source_text = _source_text(beat, segment_by_index)
    boundary_confidence = _boundary_confidence(record)
    warning_codes = _warning_codes(beat, record, warning_examples, source_text, boundary_confidence, beat_config)
    score = _score_beat(beat, source_text, warning_codes, boundary_confidence, record, beat_config)
    nearest_standard = None if record is None else record.get("source_standard_beat_number")

    return {
        "dense_beat_number": beat.beat_number,
        "beat_type": beat.beat_type,
        "start_seconds": round(beat.start_seconds, 3),
        "end_seconds": round(beat.end_seconds, 3),
        "duration_seconds": round(beat.duration_seconds, 3),
        "source_text_preview": _preview(source_text, 140),
        "warning_codes": warning_codes,
        "boundary_confidence": boundary_confidence,
        "nearest_standard_beat_number": nearest_standard if isinstance(nearest_standard, int) else None,
        "source_coherence_label": score.coherence_label,
        "review_priority": score.priority,
        "recommendation": score.recommendation,
        "score_reasons": score.reasons,
    }


def _source_text(beat: Beat, segment_by_index: dict[int, TranscriptSegment]) -> str:
    if beat.segment_indexes:
        text = " ".join(
            segment_by_index[index].text.strip()
            for index in beat.segment_indexes
            if index in segment_by_index and segment_by_index[index].text.strip()
        )
        if text:
            return _clean_preview_text(text)
    return _clean_preview_text(beat.text_preview)


def _boundary_confidence(record: dict[str, Any] | None) -> str:
    if record is None:
        return "unknown"
    value = record.get("boundary_confidence")
    if value in {"high", "medium", "weak", "not_applicable"}:
        return value
    return "unknown"


def _warning_codes(
    beat: Beat,
    record: dict[str, Any] | None,
    warning_examples: list[str],
    source_text: str,
    boundary_confidence: str,
    beat_config: BeatConfig,
) -> list[str]:
    codes = set(warning_examples)
    words = _meaningful_words(source_text)
    compact = " ".join(source_text.split())
    if not words:
        codes.add("NO_MEANINGFUL_SOURCE_TEXT")
    elif len(words) < 6:
        codes.add("VERY_SHORT_SOURCE_TEXT")
    if words and words[0].lower() in CONTINUATION_WORDS:
        codes.add("AWKWARD_CONTINUATION_START")
    if compact and compact[-1] not in TERMINAL_PUNCTUATION:
        codes.add("ENDS_MID_THOUGHT")
    if boundary_confidence in WEAK_BOUNDARIES:
        codes.add("WEAK_PUNCTUATION_BOUNDARY")
    if beat.duration_seconds + 0.001 < beat_config.dense_min_duration:
        if record and record.get("reason") == "dense_rebuild_final_tail":
            codes.add("UNAVOIDABLE_SHORT_FINAL_TAIL")
        else:
            codes.add("DURATION_BELOW_MINIMUM")
    if beat.duration_seconds > beat_config.dense_hard_max_duration + 0.001:
        codes.add("DURATION_ABOVE_HARD_MAX")
    if record is None:
        codes.add("MISSING_DENSE_GROUP_RECORD")
    return sorted(codes)


def _score_beat(
    beat: Beat,
    source_text: str,
    warning_codes: list[str],
    boundary_confidence: str,
    record: dict[str, Any] | None,
    beat_config: BeatConfig,
) -> ReviewScore:
    score = 0
    reasons = []
    words = _meaningful_words(source_text)
    promptable = _is_visually_promptable(words)

    def add(points: int, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append(f"{reason}:+{points}")

    if not words:
        add(100, "no_meaningful_source_text")
    if not promptable:
        add(70, "not_visually_promptable")
    if "DURATION_ABOVE_HARD_MAX" in warning_codes:
        add(60, "duration_above_hard_max")
    if "DURATION_BELOW_MINIMUM" in warning_codes:
        add(40, "duration_below_minimum")
    elif "UNAVOIDABLE_SHORT_FINAL_TAIL" in warning_codes:
        add(35, "unavoidable_short_final_tail")
    if "VERY_SHORT_SOURCE_TEXT" in warning_codes:
        add(30, "very_short_source_text")
    if "ENDS_MID_THOUGHT" in warning_codes:
        add(22, "ends_mid_thought")
    if boundary_confidence == "weak":
        add(12, "weak_punctuation_boundary")
    elif boundary_confidence == "medium":
        add(6, "medium_punctuation_boundary")
    if "AWKWARD_CONTINUATION_START" in warning_codes:
        add(8, "awkward_continuation_start")
    if "ENDS_MID_THOUGHT" in warning_codes and "WEAK_PUNCTUATION_BOUNDARY" in warning_codes:
        add(10, "combined_weak_mid_thought_boundary")
    if record is None:
        add(50, "missing_dense_group_record")

    if not words:
        recommendation = "blocked"
    elif record is None or "DURATION_ABOVE_HARD_MAX" in warning_codes or not promptable or score >= 60:
        recommendation = "risky"
    elif score >= 20 or "WEAK_PUNCTUATION_BOUNDARY" in warning_codes:
        recommendation = "review"
    else:
        recommendation = "approve"

    if not words or not promptable:
        coherence = "not_promptable"
    elif "VERY_SHORT_SOURCE_TEXT" in warning_codes or (
        "ENDS_MID_THOUGHT" in warning_codes and "WEAK_PUNCTUATION_BOUNDARY" in warning_codes
    ):
        coherence = "fragmented"
    elif any(code in warning_codes for code in ("ENDS_MID_THOUGHT", "WEAK_PUNCTUATION_BOUNDARY", "AWKWARD_CONTINUATION_START")):
        coherence = "minor_boundary_issue"
    else:
        coherence = "coherent"

    if not reasons:
        reasons.append("no_review_flags:+0")
    return ReviewScore(
        recommendation=recommendation,
        priority=score,
        coherence_label=coherence,
        reasons=reasons,
    )


def _readiness(counts: dict[str, int]) -> tuple[str, list[str]]:
    if counts["blocked"]:
        return "not_ready", [f"{counts['blocked']} blocked beats must be fixed before dense prompt generation."]
    if counts["risky"]:
        return "ready_with_review", [f"{counts['risky']} risky beats require manual review before dense prompt generation."]
    if counts["review"]:
        return "ready_with_review", [f"{counts['review']} review beats should be checked before dense prompt generation."]
    return "ready", ["All dense beats are approved for future dense prompt generation."]


def _markdown_report(report: dict[str, Any]) -> str:
    counts = report["recommendation_counts"]
    lines = [
        "# Dense Beat Review",
        "",
        f"- Created at: {report['created_at']}",
        f"- Preview beats: `{report['source_preview_beats_path']}`",
        f"- Dense plan: `{report['source_dense_plan_path']}`",
        f"- Transcript segments: `{report['source_transcript_segments_path']}`",
        f"- Total dense beats: {report['total_dense_beats']}",
        f"- Recommendations: {counts['approve']} approve, {counts['review']} review, {counts['risky']} risky, {counts['blocked']} blocked",
        f"- Readiness: `{report['readiness']}`",
        "",
        "## Readiness Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in report["readiness_reasons"])

    lines.extend(["", "## Top 10 Problem Beats", ""])
    _append_rows_table(lines, report["top_problem_beats"], empty="- No problem beats.")

    problem_rows = [row for row in report["beats"] if row["recommendation"] != "approve"]
    lines.extend(["", "## Problem Beats", ""])
    _append_rows_table(lines, problem_rows, empty="- No problem beats.")

    approved = [row for row in report["beats"] if row["recommendation"] == "approve"]
    approved_numbers = [row["dense_beat_number"] for row in approved]
    lines.extend(["", "## Approved Beats Summary", ""])
    if approved:
        lines.append(f"- Approved beats: {len(approved)}")
        lines.append(f"- First approved beat: {approved_numbers[0]}")
        lines.append(f"- Last approved beat: {approved_numbers[-1]}")
    else:
        lines.append("- Approved beats: 0")
    lines.append("")
    return "\n".join(lines)


def _append_rows_table(lines: list[str], rows: list[dict[str, Any]], empty: str) -> None:
    if not rows:
        lines.append(empty)
        return
    lines.append("| Beat | Time | Duration | Standard | Boundary | Warnings | Coherence | Priority | Recommendation | Source Preview |")
    lines.append("| ---: | --- | ---: | ---: | --- | --- | --- | ---: | --- | --- |")
    for row in rows:
        standard = row["nearest_standard_beat_number"] if row["nearest_standard_beat_number"] is not None else "-"
        warnings = ", ".join(f"`{code}`" for code in row["warning_codes"]) or "-"
        lines.append(
            f"| {row['dense_beat_number']} | {row['start_seconds']:.3f}-{row['end_seconds']:.3f} | "
            f"{row['duration_seconds']:.3f}s | {standard} | {row['boundary_confidence']} | "
            f"{warnings} | {row['source_coherence_label']} | {row['review_priority']} | "
            f"{row['recommendation']} | {_md_cell(row['source_text_preview'])} |"
        )


def _is_visually_promptable(words: list[str]) -> bool:
    if not words:
        return False
    normalized = [word.lower() for word in words]
    if len(normalized) >= 3:
        return True
    return len(set(normalized)) >= 2 and not all(word in {"intro", "pause", "silence"} for word in normalized)


def _meaningful_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z']*", text)


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
