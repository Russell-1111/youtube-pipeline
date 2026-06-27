from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .beat_io import load_beats
from .config import PipelineConfig
from .errors import InputFileError, PipelineError
from .generated_images import MIN_HEIGHT, MIN_WIDTH, PREFERRED_HEIGHT, PREFERRED_WIDTH
from .models import Beat

REPORT_SCHEMA_VERSION = 1
PROMPTS_NAME = "image_prompts_dense_preview.json"
PREVIEW_BEATS_NAME = "beats_dense_preview.json"
DENSE_REVIEW_NAME = "dense_beat_review.json"
DENSE_PLAN_NAME = "dense_beat_plan.json"
JSON_REPORT_NAME = "dense_image_generation_report.json"
MARKDOWN_REPORT_NAME = "dense_image_generation_report.md"
DENSE_IMAGE_DIR_NAME = "generated_images_dense_preview"
CURRENT_REVIEW_BEATS = (1, 7, 8, 13, 33, 48, 56, 63)


@dataclass(frozen=True)
class DenseImagePreparationResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path
    image_dir: Path


def run_dense_image_preparation(config: PipelineConfig, base_dir: Path) -> DenseImagePreparationResult:
    data_dir = config.outputs.data_dir
    prompts_path = data_dir / PROMPTS_NAME
    beats_path = data_dir / PREVIEW_BEATS_NAME
    review_path = data_dir / DENSE_REVIEW_NAME
    plan_path = data_dir / DENSE_PLAN_NAME
    json_path = data_dir / JSON_REPORT_NAME
    markdown_path = data_dir / MARKDOWN_REPORT_NAME
    image_dir = config.outputs.images_dir.parent / DENSE_IMAGE_DIR_NAME

    prompts_payload = _load_json_object(prompts_path, "dense image prompts")
    preview_beats = load_beats(beats_path)
    review_payload = _load_json_object(review_path, "dense beat review")
    dense_plan = _load_optional_json_object(plan_path, "dense beat plan")

    report = build_dense_image_report(
        prompts_payload=prompts_payload,
        preview_beats=preview_beats,
        review_payload=review_payload,
        dense_plan=dense_plan,
        image_dir=image_dir,
        prompts_path=prompts_path,
        beats_path=beats_path,
        review_path=review_path,
        plan_path=plan_path,
        json_path=json_path,
        markdown_path=markdown_path,
        base_dir=base_dir,
    )
    markdown = _markdown_report(report)

    image_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return DenseImagePreparationResult(report=report, json_path=json_path, markdown_path=markdown_path, image_dir=image_dir)


def build_dense_image_report(
    prompts_payload: dict[str, Any],
    preview_beats: list[Beat],
    review_payload: dict[str, Any],
    dense_plan: dict[str, Any] | None,
    image_dir: Path,
    prompts_path: Path = Path("data/image_prompts_dense_preview.json"),
    beats_path: Path = Path("data/beats_dense_preview.json"),
    review_path: Path = Path("data/dense_beat_review.json"),
    plan_path: Path = Path("data/dense_beat_plan.json"),
    json_path: Path = Path("data/dense_image_generation_report.json"),
    markdown_path: Path = Path("data/dense_image_generation_report.md"),
    base_dir: Path | None = None,
) -> dict[str, Any]:
    base = base_dir or Path.cwd()
    prompt_rows = _validate_inputs(prompts_payload, preview_beats, review_payload, dense_plan)
    images = [_image_row(prompt_row, image_dir, base) for prompt_row in prompt_rows]
    images.extend(_extra_image_rows(image_dir, {row["expected_filename"] for row in images}, base))

    present_count = sum(1 for row in images if row["status"] in {"valid", "warning", "invalid"})
    missing_count = sum(1 for row in images if row["status"] == "missing")
    invalid_count = sum(1 for row in images if row["status"] == "invalid")
    warning_count = sum(1 for row in images if row["status"] in {"warning", "extra"})
    duplicate_expected = _duplicate_expected_filenames(images)
    sample_batch = _sample_batch_recommendation(prompt_rows)
    readiness_reasons = _readiness_reasons(
        expected_count=len(prompt_rows),
        present_count=present_count,
        missing_count=missing_count,
        invalid_count=invalid_count,
        duplicate_expected=duplicate_expected,
    )
    ready = not readiness_reasons

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source_paths": {
            "dense_image_prompts": _display_path(prompts_path, base),
            "dense_preview_beats": _display_path(beats_path, base),
            "dense_beat_review": _display_path(review_path, base),
            "dense_beat_plan": _display_path(plan_path, base) if dense_plan is not None else None,
        },
        "output_paths": {
            "json": _display_path(json_path, base),
            "markdown": _display_path(markdown_path, base),
        },
        "image_directory": _display_path(image_dir, base),
        "total_dense_prompts": len(prompt_rows),
        "expected_image_count": len(prompt_rows),
        "present_image_count": present_count,
        "missing_image_count": missing_count,
        "invalid_image_count": invalid_count,
        "warning_count": warning_count,
        "ready_for_dense_render": ready,
        "readiness_reasons": ["All expected dense preview images are valid."] if ready else readiness_reasons,
        "validation_policy": {
            "format": "PNG only",
            "minimum_size": f"{MIN_WIDTH}x{MIN_HEIGHT}",
            "aspect_ratio": "exact 16:9",
            "preferred_size": f"{PREFERRED_WIDTH}x{PREFERRED_HEIGHT}",
            "missing_images_fail_cli": False,
        },
        "review_context_image_count": sum(1 for row in prompt_rows if row["context_used"]),
        "sample_batch_recommendation": sample_batch,
        "images": images,
    }


def print_dense_image_summary(result: DenseImagePreparationResult, base_dir: Path) -> None:
    report = result.report
    print("Dense image preparation complete")
    print(f"Expected dense images: {report['expected_image_count']}")
    print(f"Present dense images: {report['present_image_count']}")
    print(f"Missing dense images: {report['missing_image_count']}")
    print(f"Invalid dense images: {report['invalid_image_count']}")
    print(f"Warnings: {report['warning_count']}")
    print(f"Ready for dense render: {report['ready_for_dense_render']}")
    print(f"Dense image directory: {_display_path(result.image_dir, base_dir)}")
    print("Reports written:")
    print(f"- {_display_path(result.json_path, base_dir)}")
    print(f"- {_display_path(result.markdown_path, base_dir)}")


def dense_image_filename(dense_beat_number: int) -> str:
    return f"dense_beat_{dense_beat_number:03}.png"


def _validate_inputs(
    prompts_payload: dict[str, Any],
    preview_beats: list[Beat],
    review_payload: dict[str, Any],
    dense_plan: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    prompt_rows = prompts_payload.get("prompts")
    if not isinstance(prompt_rows, list) or not prompt_rows:
        raise PipelineError("Invalid dense image prompts JSON: expected non-empty prompts list.")
    if not preview_beats:
        raise PipelineError("Cannot prepare dense images without dense preview beats.")

    review_rows = review_payload.get("beats")
    if not isinstance(review_rows, list):
        raise PipelineError("Invalid dense beat review JSON: expected beats list.")
    review_total = review_payload.get("total_dense_beats")
    if not isinstance(review_total, int):
        raise PipelineError("Invalid dense beat review JSON: missing total_dense_beats.")
    if review_total != len(review_rows):
        raise PipelineError(f"Dense review count mismatch: total {review_total}, rows {len(review_rows)}.")

    prompt_by_number = _rows_by_dense_number(prompt_rows, "dense image prompt")
    review_by_number = _rows_by_dense_number(review_rows, "dense beat review")
    beat_numbers = [beat.beat_number for beat in preview_beats]
    if len(set(beat_numbers)) != len(beat_numbers):
        raise PipelineError("Invalid dense preview beats: duplicate dense beat numbers.")
    if set(prompt_by_number) != set(beat_numbers) or set(review_by_number) != set(beat_numbers):
        raise PipelineError(
            "Prompt/beat/review mismatch: "
            f"prompt beats {sorted(prompt_by_number)}, preview beats {sorted(beat_numbers)}, "
            f"review beats {sorted(review_by_number)}."
        )

    total_dense_beats = prompts_payload.get("total_dense_beats")
    prompt_count = prompts_payload.get("prompt_count")
    if total_dense_beats is not None and total_dense_beats != len(preview_beats):
        raise PipelineError(f"Dense prompt total mismatch: {total_dense_beats} prompts total, {len(preview_beats)} preview beats.")
    if prompt_count is not None and prompt_count != len(prompt_rows):
        raise PipelineError(f"Dense prompt count mismatch: {prompt_count} prompt_count, {len(prompt_rows)} prompt rows.")

    if dense_plan is not None:
        plan_count = dense_plan.get("dense_preview_beat_count")
        if isinstance(plan_count, int) and plan_count != len(preview_beats):
            raise PipelineError(f"Dense plan count mismatch: {plan_count} planned beats, {len(preview_beats)} preview beats.")

    normalized_rows = [prompt_by_number[number] for number in sorted(beat_numbers)]
    expected_filenames = [dense_image_filename(row["dense_beat_number"]) for row in normalized_rows]
    if len(set(expected_filenames)) != len(expected_filenames):
        raise PipelineError("Duplicate expected dense image filenames.")
    return normalized_rows


def _rows_by_dense_number(rows: list[Any], label: str) -> dict[int, dict[str, Any]]:
    by_number: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise PipelineError(f"Invalid {label} row at index {index}: expected an object.")
        beat_number = row.get("dense_beat_number")
        if not isinstance(beat_number, int):
            raise PipelineError(f"Invalid {label} row at index {index}: missing dense_beat_number.")
        if beat_number in by_number:
            raise PipelineError(f"Invalid {label} JSON: duplicate row for dense beat {beat_number}.")
        by_number[beat_number] = row
    return by_number


def _image_row(prompt_row: dict[str, Any], image_dir: Path, base_dir: Path) -> dict[str, Any]:
    beat_number = prompt_row["dense_beat_number"]
    expected_filename = dense_image_filename(beat_number)
    expected_path = image_dir / expected_filename
    validation_messages: list[str] = []
    width: int | None = None
    height: int | None = None
    aspect_ratio: str | None = None
    status = "missing"

    if expected_path.exists():
        if not expected_path.is_file():
            status = "invalid"
            validation_messages.append("Expected dense image path exists but is not a file.")
        else:
            status, width, height, aspect_ratio, validation_messages = _validate_image_file(expected_path)
    else:
        validation_messages.append("Missing dense preview image.")

    return {
        "dense_beat_number": beat_number,
        "expected_filename": expected_filename,
        "expected_path": _display_path(expected_path, base_dir),
        "prompt_review_recommendation": prompt_row.get("review_recommendation"),
        "context_used": bool(prompt_row.get("context_used")),
        "visual_concept_text": prompt_row.get("visual_concept_text"),
        "final_image_prompt": prompt_row.get("final_image_prompt"),
        "status": status,
        "exists": expected_path.exists(),
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "validation_messages": validation_messages,
        "retry_recommended": status in {"missing", "invalid", "warning"},
    }


def _validate_image_file(path: Path) -> tuple[str, int | None, int | None, str | None, list[str]]:
    messages: list[str] = []
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
    except (OSError, UnidentifiedImageError) as exc:
        return "invalid", None, None, None, [f"Could not open dense image: {exc}"]

    aspect_ratio = f"{width}:{height}"
    status = "valid"
    if image_format != "PNG":
        status = "invalid"
        messages.append(f"Dense image must be PNG; detected {image_format}.")
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        status = "invalid"
        messages.append(f"Dense image is too small: {width}x{height}, minimum is {MIN_WIDTH}x{MIN_HEIGHT}.")
    if width * 9 != height * 16:
        status = "invalid"
        messages.append(f"Dense image must be exact 16:9; found {width}x{height}.")
    elif (width, height) != (PREFERRED_WIDTH, PREFERRED_HEIGHT) and status != "invalid":
        status = "warning"
        messages.append(f"Dense image is {width}x{height}; preferred size is {PREFERRED_WIDTH}x{PREFERRED_HEIGHT}.")
    if not messages:
        messages.append("Dense image is valid.")
    return status, width, height, aspect_ratio, messages


def _extra_image_rows(image_dir: Path, expected_names: set[str], base_dir: Path) -> list[dict[str, Any]]:
    if not image_dir.exists():
        return []
    rows = []
    for path in sorted(image_dir.glob("*.png")):
        if path.name in expected_names:
            continue
        rows.append(
            {
                "dense_beat_number": None,
                "expected_filename": path.name,
                "expected_path": _display_path(path, base_dir),
                "prompt_review_recommendation": None,
                "context_used": False,
                "visual_concept_text": None,
                "final_image_prompt": None,
                "status": "extra",
                "exists": True,
                "width": None,
                "height": None,
                "aspect_ratio": None,
                "validation_messages": ["Extra PNG file in dense preview image directory."],
                "retry_recommended": False,
            }
        )
    return rows


def _duplicate_expected_filenames(images: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for row in images:
        if row["status"] == "extra":
            continue
        counts[row["expected_filename"]] = counts.get(row["expected_filename"], 0) + 1
    return sorted(name for name, count in counts.items() if count > 1)


def _readiness_reasons(
    expected_count: int,
    present_count: int,
    missing_count: int,
    invalid_count: int,
    duplicate_expected: list[str],
) -> list[str]:
    reasons = []
    if present_count != expected_count:
        reasons.append(f"{present_count} of {expected_count} expected dense images are present.")
    if missing_count:
        reasons.append(f"{missing_count} expected dense images are missing.")
    if invalid_count:
        reasons.append(f"{invalid_count} dense images are invalid.")
    if duplicate_expected:
        reasons.append(f"Duplicate expected dense image filenames: {', '.join(duplicate_expected)}.")
    return reasons


def _sample_batch_recommendation(prompt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_number = {row["dense_beat_number"]: row for row in prompt_rows}
    total = len(prompt_rows)
    candidate_numbers: list[int] = []
    anchors = [1, 2, max(1, total // 3), max(1, total // 2), max(1, (total * 2) // 3), max(1, total - 1), total]
    for number in [*CURRENT_REVIEW_BEATS, *anchors]:
        if number in by_number and number not in candidate_numbers:
            candidate_numbers.append(number)
    for row in prompt_rows:
        if row.get("context_used") and row["dense_beat_number"] not in candidate_numbers:
            candidate_numbers.append(row["dense_beat_number"])
        if len(candidate_numbers) >= 12:
            break
    for row in prompt_rows:
        if len(candidate_numbers) >= 10:
            break
        if row["dense_beat_number"] not in candidate_numbers:
            candidate_numbers.append(row["dense_beat_number"])
    candidate_numbers = sorted(candidate_numbers[:12])
    return [_sample_row(by_number[number], total) for number in candidate_numbers]


def _sample_row(prompt_row: dict[str, Any], total: int) -> dict[str, Any]:
    beat_number = prompt_row["dense_beat_number"]
    reasons = []
    if beat_number in CURRENT_REVIEW_BEATS:
        reasons.append("current review beat")
    if beat_number <= 2:
        reasons.append("early section")
    elif beat_number >= total - 1:
        reasons.append("final section")
    elif beat_number <= total // 3:
        reasons.append("middle section")
    else:
        reasons.append("late section")
    if prompt_row.get("context_used"):
        reasons.append("review-context prompt")
    concept = prompt_row.get("visual_concept_text") or ""
    return {
        "dense_beat_number": beat_number,
        "reason_for_inclusion": ", ".join(dict.fromkeys(reasons)),
        "visual_family_or_concept": concept,
        "review_context": bool(prompt_row.get("context_used")),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    missing = [row for row in report["images"] if row["status"] == "missing"]
    invalid = [row for row in report["images"] if row["status"] == "invalid"]
    warnings = [row for row in report["images"] if row["status"] in {"warning", "extra"}]
    review_context = [row for row in report["images"] if row["context_used"]]
    lines = [
        "# Dense Image Generation Report",
        "",
        "## Summary",
        "",
        f"- Expected dense images: {report['expected_image_count']}",
        f"- Present dense images: {report['present_image_count']}",
        f"- Missing dense images: {report['missing_image_count']}",
        f"- Invalid dense images: {report['invalid_image_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        "## Readiness",
        "",
        f"- Ready for dense render: `{report['ready_for_dense_render']}`",
    ]
    lines.extend(f"- {reason}" for reason in report["readiness_reasons"])
    lines.extend(["", "## Missing images", ""])
    lines.extend(_checklist_lines(missing, "Missing dense images", include_message=False))
    lines.extend(["", "## Invalid images", ""])
    lines.extend(_checklist_lines(invalid, "Invalid dense images", include_message=True))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_checklist_lines(warnings, "Warning dense images", include_message=True))
    lines.extend(["", "## Review-context image slots", ""])
    lines.extend(_checklist_lines(review_context, "Review-context image slots", include_message=False))
    lines.extend(["", "## Recommended 8-12 sample batch", ""])
    lines.append("| Dense Beat | Reason | Visual Family/Concept | Review Context |")
    lines.append("| ---: | --- | --- | --- |")
    for row in report["sample_batch_recommendation"]:
        lines.append(
            f"| {row['dense_beat_number']} | {_md_cell(row['reason_for_inclusion'])} | "
            f"{_md_cell(_preview(row['visual_family_or_concept'], 120))} | {row['review_context']} |"
        )
    lines.extend(["", "## Full dense image checklist", ""])
    lines.append("| Dense Beat | Expected Filename | Status | Size | Review | Context |")
    lines.append("| ---: | --- | --- | --- | --- | --- |")
    for row in report["images"]:
        size = f"{row['width']}x{row['height']}" if row["width"] and row["height"] else "-"
        beat = row["dense_beat_number"] if row["dense_beat_number"] is not None else "-"
        lines.append(
            f"| {beat} | `{row['expected_filename']}` | {row['status']} | {size} | "
            f"{row['prompt_review_recommendation'] or '-'} | {row['context_used']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _checklist_lines(rows: list[dict[str, Any]], empty_label: str, include_message: bool) -> list[str]:
    if not rows:
        return ["- None"]
    lines = []
    for row in rows:
        beat = row["dense_beat_number"] if row["dense_beat_number"] is not None else "extra"
        line = f"- Dense beat {beat}: `{row['expected_filename']}`"
        if include_message:
            line += f" - {'; '.join(row['validation_messages'])}"
        lines.append(line)
    return lines


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


def _load_optional_json_object(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json_object(path, label)


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
