from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .beat_io import load_beats, load_transcript_segments
from .config import PipelineConfig
from .errors import InputFileError, PipelineError
from .models import Beat, TranscriptSegment
from .prompts import (
    build_standard_image_prompt,
    sanitize_prompt_source_text,
    visual_concepts_for_text,
)

PROMPT_SCHEMA_VERSION = 1
PREVIEW_BEATS_NAME = "beats_dense_preview.json"
DENSE_REVIEW_NAME = "dense_beat_review.json"
DENSE_PLAN_NAME = "dense_beat_plan.json"
TRANSCRIPT_SEGMENTS_NAME = "transcript_segments.json"
JSON_OUTPUT_NAME = "image_prompts_dense_preview.json"
MARKDOWN_OUTPUT_NAME = "image_prompts_dense_preview.md"
ALLOWED_READINESS = {"ready", "ready_with_review"}
ALLOWED_RECOMMENDATIONS = {"approve", "review"}
RECOMMENDATIONS = ("approve", "review", "risky", "blocked")
CONTEXT_POLICY = (
    "Approve beats use only the target dense beat text. Review beats include available previous and next dense beat text "
    "only to complete fragments, avoid misleading imagery, and preserve meaning; final prompts must visualize the target beat only."
)
STYLE_CONSTRAINTS = (
    "Consistent Melancholic Minimalist 2D Explainer visual language; dark but readable 16:9 cinematic 2D symbolic imagery; "
    "no subtitles, embedded narration text, captions, logos, or watermarks; not photorealistic, not corporate, not whiteboard."
)


@dataclass(frozen=True)
class DensePromptResult:
    payload: dict[str, Any]
    json_path: Path
    markdown_path: Path


def run_dense_prompt_generation(config: PipelineConfig, base_dir: Path) -> DensePromptResult:
    data_dir = config.outputs.data_dir
    preview_path = data_dir / PREVIEW_BEATS_NAME
    review_path = data_dir / DENSE_REVIEW_NAME
    plan_path = data_dir / DENSE_PLAN_NAME
    transcript_path = data_dir / TRANSCRIPT_SEGMENTS_NAME
    json_path = data_dir / JSON_OUTPUT_NAME
    markdown_path = data_dir / MARKDOWN_OUTPUT_NAME

    preview_beats = load_beats(preview_path)
    review = _load_json_object(review_path, "dense beat review")
    dense_plan = _load_json_object(plan_path, "dense beat plan")
    segments = load_transcript_segments(transcript_path)

    payload = build_dense_prompt_payload(
        preview_beats=preview_beats,
        review=review,
        dense_plan=dense_plan,
        segments=segments,
        preview_path=preview_path,
        review_path=review_path,
        plan_path=plan_path,
        transcript_path=transcript_path,
        json_path=json_path,
        markdown_path=markdown_path,
        base_dir=base_dir,
    )
    markdown = _markdown_report(payload)

    data_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return DensePromptResult(payload=payload, json_path=json_path, markdown_path=markdown_path)


def build_dense_prompt_payload(
    preview_beats: list[Beat],
    review: dict[str, Any],
    dense_plan: dict[str, Any],
    segments: list[TranscriptSegment],
    preview_path: Path = Path("data/beats_dense_preview.json"),
    review_path: Path = Path("data/dense_beat_review.json"),
    plan_path: Path = Path("data/dense_beat_plan.json"),
    transcript_path: Path = Path("data/transcript_segments.json"),
    json_path: Path = Path("data/image_prompts_dense_preview.json"),
    markdown_path: Path = Path("data/image_prompts_dense_preview.md"),
    base_dir: Path | None = None,
) -> dict[str, Any]:
    base = base_dir or Path.cwd()
    review_rows = _validate_inputs(preview_beats, review, dense_plan, segments)
    segment_by_index = {segment.index: segment for segment in segments}
    source_text_by_number = {
        beat.beat_number: _source_text(beat, segment_by_index)
        for beat in preview_beats
    }
    prompts = [
        _prompt_row(
            beat=beat,
            review_row=review_rows[beat.beat_number],
            source_text_by_number=source_text_by_number,
            previous_beat=preview_beats[index - 1] if index > 0 else None,
            next_beat=preview_beats[index + 1] if index < len(preview_beats) - 1 else None,
        )
        for index, beat in enumerate(preview_beats)
    ]
    counts = _recommendation_counts(review, prompts)
    return {
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source_dense_review_readiness": review["readiness"],
        "total_dense_beats": len(preview_beats),
        "prompt_count": len(prompts),
        "context_policy": CONTEXT_POLICY,
        "source_paths": {
            "dense_preview_beats": _display_path(preview_path, base),
            "dense_review": _display_path(review_path, base),
            "dense_plan": _display_path(plan_path, base),
            "transcript_segments": _display_path(transcript_path, base),
        },
        "output_paths": {
            "json": _display_path(json_path, base),
            "markdown": _display_path(markdown_path, base),
        },
        "recommendation_counts": counts,
        "planner_metadata": _planner_metadata(dense_plan),
        "prompts": prompts,
    }


def print_dense_prompt_summary(result: DensePromptResult, base_dir: Path) -> None:
    payload = result.payload
    counts = payload["recommendation_counts"]
    context_count = sum(1 for row in payload["prompts"] if row["context_used"])
    print("Dense prompt generation complete")
    print(f"Total dense beats: {payload['total_dense_beats']}")
    print(f"Prompts: {payload['prompt_count']}")
    print(f"Review readiness: {payload['source_dense_review_readiness']}")
    print(
        "Recommendations: "
        f"{counts['approve']} approve, {counts['review']} review, "
        f"{counts['risky']} risky, {counts['blocked']} blocked"
    )
    print(f"Review-context prompts: {context_count}")
    print("Reports written:")
    print(f"- {_display_path(result.json_path, base_dir)}")
    print(f"- {_display_path(result.markdown_path, base_dir)}")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise InputFileError(f"Missing {label} file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid {label} JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise PipelineError(f"Could not read {label} JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"Invalid {label} JSON: expected an object in {path}")
    return payload


def _validate_inputs(
    preview_beats: list[Beat],
    review: dict[str, Any],
    dense_plan: dict[str, Any],
    segments: list[TranscriptSegment],
) -> dict[int, dict[str, Any]]:
    if not preview_beats:
        raise PipelineError("Cannot generate dense prompts without dense preview beats.")
    if not segments:
        raise PipelineError("Cannot generate dense prompts without transcript segments.")
    if not isinstance(dense_plan.get("dense_group_records"), list):
        raise PipelineError("Invalid dense beat plan JSON: expected dense_group_records list.")

    readiness = review.get("readiness")
    if readiness not in ALLOWED_READINESS:
        raise PipelineError(f"Dense review readiness must be ready or ready_with_review; found {readiness!r}.")

    rows = review.get("beats")
    if not isinstance(rows, list):
        raise PipelineError("Invalid dense beat review JSON: expected beats list.")
    total_dense_beats = review.get("total_dense_beats")
    if not isinstance(total_dense_beats, int):
        raise PipelineError("Invalid dense beat review JSON: missing total_dense_beats.")
    if len(preview_beats) != total_dense_beats or len(preview_beats) != len(rows):
        raise PipelineError(
            "Dense beat count mismatch: "
            f"{len(preview_beats)} preview beats, {total_dense_beats} review total, {len(rows)} review rows."
        )

    review_by_number: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise PipelineError(f"Invalid dense review row at index {index}: expected an object.")
        beat_number = row.get("dense_beat_number")
        if not isinstance(beat_number, int):
            raise PipelineError(f"Invalid dense review row at index {index}: missing dense_beat_number.")
        if beat_number in review_by_number:
            raise PipelineError(f"Invalid dense beat review JSON: duplicate review row for beat {beat_number}.")
        recommendation = row.get("recommendation")
        if recommendation not in set(RECOMMENDATIONS):
            raise PipelineError(f"Invalid dense review row for beat {beat_number}: unknown recommendation {recommendation!r}.")
        review_by_number[beat_number] = row

    missing = [beat.beat_number for beat in preview_beats if beat.beat_number not in review_by_number]
    extra = sorted(set(review_by_number) - {beat.beat_number for beat in preview_beats})
    if missing or extra:
        raise PipelineError(f"Dense beat/review rows do not match. Missing: {missing or []}. Extra: {extra or []}.")

    counts = _normalized_counts(review.get("recommendation_counts"), rows)
    if counts["risky"]:
        raise PipelineError(f"Dense prompt generation is blocked: {counts['risky']} risky beats exist.")
    if counts["blocked"]:
        raise PipelineError(f"Dense prompt generation is blocked: {counts['blocked']} blocked beats exist.")
    disallowed = [
        row["dense_beat_number"]
        for row in rows
        if row.get("recommendation") not in ALLOWED_RECOMMENDATIONS
    ]
    if disallowed:
        raise PipelineError(f"Dense prompt generation is blocked by disallowed recommendations on beats: {disallowed}.")
    return review_by_number


def _normalized_counts(source_counts: object, rows: list[dict[str, Any]]) -> dict[str, int]:
    actual = Counter(row.get("recommendation") for row in rows)
    counts = {recommendation: int(actual.get(recommendation, 0)) for recommendation in RECOMMENDATIONS}
    if source_counts is not None:
        if not isinstance(source_counts, dict):
            raise PipelineError("Invalid dense beat review JSON: recommendation_counts must be an object.")
        for recommendation in RECOMMENDATIONS:
            value = source_counts.get(recommendation, 0)
            if not isinstance(value, int):
                raise PipelineError(f"Invalid dense beat review JSON: recommendation_counts.{recommendation} must be an integer.")
            if value != counts[recommendation]:
                raise PipelineError(
                    f"Dense review recommendation_counts mismatch for {recommendation}: "
                    f"report has {value}, rows have {counts[recommendation]}."
                )
    return counts


def _recommendation_counts(review: dict[str, Any], prompts: list[dict[str, Any]]) -> dict[str, int]:
    rows = [{"recommendation": prompt["review_recommendation"]} for prompt in prompts]
    return _normalized_counts(review.get("recommendation_counts"), rows)


def _prompt_row(
    beat: Beat,
    review_row: dict[str, Any],
    source_text_by_number: dict[int, str],
    previous_beat: Beat | None,
    next_beat: Beat | None,
) -> dict[str, Any]:
    recommendation = review_row["recommendation"]
    target_text = source_text_by_number[beat.beat_number]
    context_used = recommendation == "review"
    previous_context = source_text_by_number.get(previous_beat.beat_number) if context_used and previous_beat else None
    next_context = source_text_by_number.get(next_beat.beat_number) if context_used and next_beat else None
    visual_concept_text, visual_concept_keywords = visual_concepts_for_text(target_text)
    final_prompt = build_standard_image_prompt(beat, target_text, visual_concept_text)
    if context_used:
        final_prompt = _with_review_context(final_prompt, target_text, previous_context, next_context)

    return {
        "dense_beat_number": beat.beat_number,
        "beat_type": beat.beat_type,
        "start_seconds": round(beat.start_seconds, 3),
        "end_seconds": round(beat.end_seconds, 3),
        "start_timecode": beat.start,
        "end_timecode": beat.end,
        "duration_seconds": round(beat.duration_seconds, 3),
        "review_recommendation": recommendation,
        "target_text": target_text,
        "previous_context_text": previous_context,
        "next_context_text": next_context,
        "context_used": context_used,
        "visual_concept_text": visual_concept_text,
        "visual_concept_keywords": visual_concept_keywords,
        "final_image_prompt": final_prompt,
        "style_constraints": STYLE_CONSTRAINTS,
        "prompt_risk_notes": _prompt_risk_notes(review_row),
    }


def _with_review_context(
    base_prompt: str,
    target_text: str,
    previous_context: str | None,
    next_context: str | None,
) -> str:
    previous = previous_context or "None available."
    next_text = next_context or "None available."
    return (
        f"{base_prompt} Dense review context policy: visualize only the target beat: {target_text} "
        "Use adjacent context only to complete fragments, avoid misleading imagery, and preserve meaning. "
        f"Previous beat context: {previous} Next beat context: {next_text} "
        "Do not depict a different beat's event, metaphor, setting, or action."
    )


def _prompt_risk_notes(review_row: dict[str, Any]) -> list[str]:
    notes = []
    for key in ("warning_codes", "score_reasons"):
        value = review_row.get(key, [])
        if isinstance(value, list):
            notes.extend(str(item) for item in value)
    return notes


def _source_text(beat: Beat, segment_by_index: dict[int, TranscriptSegment]) -> str:
    if beat.segment_indexes:
        text = " ".join(
            segment_by_index[index].text.strip()
            for index in beat.segment_indexes
            if index in segment_by_index and segment_by_index[index].text.strip()
        )
        if text:
            return sanitize_prompt_source_text(text)
    return sanitize_prompt_source_text(beat.text_preview)


def _planner_metadata(dense_plan: dict[str, Any]) -> dict[str, Any]:
    summary = dense_plan.get("summary")
    return {
        "planner_version": dense_plan.get("planner_version"),
        "planner_strategy": dense_plan.get("planner_strategy"),
        "planner_created_at": dense_plan.get("created_at"),
        "dense_preview_beat_count": dense_plan.get("dense_preview_beat_count"),
        "safe_to_apply": dense_plan.get("safe_to_apply"),
        "target_range_min": dense_plan.get("target_range_min"),
        "target_range_max": dense_plan.get("target_range_max"),
        "warnings_count": len(dense_plan.get("warnings", [])) if isinstance(dense_plan.get("warnings"), list) else None,
        "summary": summary if isinstance(summary, dict) else {},
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    counts = payload["recommendation_counts"]
    context_rows = [row for row in payload["prompts"] if row["context_used"]]
    lines = [
        "# Dense Image Prompts Preview",
        "",
        "## Summary",
        "",
        f"- Dense beat count: {payload['total_dense_beats']}",
        f"- Prompt count: {payload['prompt_count']}",
        f"- Review readiness: `{payload['source_dense_review_readiness']}`",
        f"- Recommendation counts: {counts['approve']} approve, {counts['review']} review, {counts['risky']} risky, {counts['blocked']} blocked",
        f"- Context policy: {payload['context_policy']}",
        "",
        "## Prompt-generation policy",
        "",
        "- This preview writes dense prompt artifacts only and does not overwrite production prompt, beat, or transcript files.",
        "- Approved beats are generated from target dense beat text only.",
        "- Review beats may use neighboring context only to preserve the target beat's meaning; the image prompt must stay focused on the target beat.",
        "",
        "## Review beats using neighboring context",
        "",
    ]
    if context_rows:
        lines.append("| Beat | Previous Context | Target | Next Context |")
        lines.append("| ---: | --- | --- | --- |")
        for row in context_rows:
            lines.append(
                f"| {row['dense_beat_number']} | {_md_cell(row['previous_context_text'] or '-')} | "
                f"{_md_cell(row['target_text'])} | {_md_cell(row['next_context_text'] or '-')} |"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Compact prompt list", ""])
    lines.append("| Beat | Time | Recommendation | Context | Visual Concept | Target Preview |")
    lines.append("| ---: | --- | --- | --- | --- | --- |")
    for row in payload["prompts"]:
        lines.append(
            f"| {row['dense_beat_number']} | {row['start_seconds']:.3f}-{row['end_seconds']:.3f}s | "
            f"{row['review_recommendation']} | {row['context_used']} | "
            f"{_md_cell(row['visual_concept_text'])} | {_md_cell(_preview(row['target_text'], 90))} |"
        )

    lines.extend(["", "## Output file paths", ""])
    lines.append(f"- JSON: `{payload['output_paths']['json']}`")
    lines.append(f"- Markdown: `{payload['output_paths']['markdown']}`")
    lines.append("")
    return "\n".join(lines)


def _preview(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
