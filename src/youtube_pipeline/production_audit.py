from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from PIL import Image, ImageStat, UnidentifiedImageError

from .beat_io import load_beats, load_transcript_segments
from .config import PipelineConfig
from .errors import PipelineError
from .generated_images import generated_image_filename, validate_generated_images
from .models import Beat
from .prompts import SCHEMA_VERSION, STYLE_PROFILE

AUDIT_SCHEMA_VERSION = 1
JSON_REPORT_NAME = "production_audit_report.json"
MARKDOWN_REPORT_NAME = "production_audit_report.md"

SEVERITY_ORDER = {
    "error": 0,
    "warning": 1,
    "info": 2,
    "recommendation": 3,
}

CONTINUATION_WORDS = {
    "and",
    "are",
    "as",
    "because",
    "but",
    "for",
    "if",
    "is",
    "or",
    "so",
    "that",
    "then",
    "though",
    "when",
    "while",
    "who",
    "with",
    "yet",
}
PUNCTUATION_ENDINGS = {".", "!", "?", '"', "'"}
VISUAL_DENSITY_TARGET_RANGE_MIN = 70
VISUAL_DENSITY_TARGET_RANGE_MAX = 90
VISUAL_DENSITY_PREFERRED_BEAT_SECONDS = 7.5
VISUAL_DENSITY_MIN_REVIEW_SECONDS = 8.0
VISUAL_DENSITY_REVIEW_LIMIT = 8


@dataclass(frozen=True)
class AuditResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path

    @property
    def has_core_errors(self) -> bool:
        return any(finding["severity"] == "error" for finding in self.report["findings"])


def run_production_audit(config: PipelineConfig, base_dir: Path) -> AuditResult:
    beats_path = config.outputs.data_dir / "beats.json"
    transcript_segments_path = config.outputs.data_dir / "transcript_segments.json"
    prompts_path = config.outputs.data_dir / "image_prompts.json"
    generated_image_dir = config.outputs.images_dir.parent / "generated_images"
    output_dir = config.outputs.final_video.parent

    findings: list[dict[str, Any]] = []
    sections = _empty_sections()
    beats = _load_beats_for_audit(beats_path, findings)
    prompt_payload = _load_optional_json(prompts_path, "MISSING_IMAGE_PROMPTS_JSON", "INVALID_IMAGE_PROMPTS_JSON", findings)
    prompt_records = _prompt_records(prompt_payload)
    prompt_lookup = _prompt_lookup(prompt_records)
    transcript_segment_count = _transcript_segment_count(transcript_segments_path, findings)

    if beats is None:
        report = _build_report(
            config=config,
            beats=[],
            transcript_segment_count=transcript_segment_count,
            generated_image_count_found=_generated_image_count(generated_image_dir),
            sections=sections,
            findings=findings,
            recommendations=["Create data/beats.json by running the dry-run pipeline before auditing production readiness."],
        )
        return _write_reports(report, config.outputs.data_dir, base_dir)

    sections["visual_density"] = _audit_visual_density(beats, config.beats.max_duration)
    sections["pacing"] = _audit_pacing(beats, prompt_lookup, findings)
    sections["generated_images"] = _audit_generated_images(beats, generated_image_dir, findings)
    sections["prompts"] = _audit_prompts(prompt_payload, prompt_records, generated_image_dir, findings)
    sections["metaphors"] = _audit_metaphors(prompt_records, findings)
    sections["brightness"] = _audit_brightness(beats, generated_image_dir, findings)
    sections["render_risk"] = _audit_render_risk(beats, config.video.fps, findings)
    sections["outputs"] = _audit_outputs(output_dir, findings)
    sections["artifact_safety"] = _audit_artifact_safety(base_dir)
    sections["source_text"] = _audit_source_text(beats, prompt_lookup, findings)
    _add_info_findings(
        beats=beats,
        fps=config.video.fps,
        transcript_segment_count=transcript_segment_count,
        generated_image_count_found=_generated_image_count(generated_image_dir),
        prompt_payload=prompt_payload,
        findings=findings,
    )

    recommendations = _recommendations(findings)
    report = _build_report(
        config=config,
        beats=beats,
        transcript_segment_count=transcript_segment_count,
        generated_image_count_found=_generated_image_count(generated_image_dir),
        sections=sections,
        findings=findings,
        recommendations=recommendations,
    )
    return _write_reports(report, config.outputs.data_dir, base_dir)


def print_audit_summary(result: AuditResult, base_dir: Path) -> None:
    findings = result.report["findings"]
    counts = Counter(finding["severity"] for finding in findings)
    print("Production audit complete")
    print(f"Readiness: {result.report['readiness_score']:.1f}/10 {result.report['readiness_label']}")
    print(
        "Findings: "
        f"{counts['error']} errors, {counts['warning']} warnings, "
        f"{counts['info']} info, {counts['recommendation']} recommendations"
    )
    print("Reports written:")
    print(f"- {_display_path(result.json_path, base_dir)}")
    print(f"- {_display_path(result.markdown_path, base_dir)}")


def _empty_sections() -> dict[str, dict[str, Any]]:
    return {
        "visual_density": {},
        "pacing": {},
        "generated_images": {},
        "prompts": {},
        "metaphors": {},
        "brightness": {},
        "render_risk": {},
        "outputs": {},
        "artifact_safety": {},
    }


def _audit_visual_density(beats: list[Beat], config_max_duration: float) -> dict[str, Any]:
    durations = [beat.duration_seconds for beat in beats]
    total_duration = round(sum(durations), 3)
    beat_count = len(beats)
    average = round(total_duration / beat_count, 3) if beat_count else 0.0
    median = round(_median(durations), 3) if durations else 0.0
    p90 = round(_percentile(durations, 90), 3) if durations else 0.0
    estimated_target = int(round(total_duration / VISUAL_DENSITY_PREFERRED_BEAT_SECONDS)) if total_duration else 0

    beats_over_8 = [beat for beat in beats if beat.duration_seconds > 8.0]
    beats_over_10 = [beat for beat in beats if beat.duration_seconds > 10.0]
    beats_over_12 = [beat for beat in beats if beat.duration_seconds > 12.0]
    beats_over_config_max = [beat for beat in beats if beat.duration_seconds > config_max_duration]

    findings = []
    if estimated_target and beat_count < max(1, int(math.floor(estimated_target * 0.9))):
        findings.append(
            _visual_density_finding(
                "VISUAL_DENSITY_USABLE_SPARSE",
                "Current visual pacing is usable but sparse relative to the future 70-90 image target.",
                "Advisory only: keep this workflow usable while adding optional dense-mode planning later.",
            )
        )
    if beats_over_config_max:
        findings.append(
            _visual_density_finding(
                "BEATS_OVER_CONFIG_MAX_DURATION",
                f"{len(beats_over_config_max)} beats exceed configured beats.max_duration of {config_max_duration:.1f}s.",
                "Diagnostic only: investigate why existing beat records exceed config before changing generation behavior.",
            )
        )
    if beats_over_10:
        findings.append(
            _visual_density_finding(
                "LONG_STATIC_STRETCH_REVIEW",
                f"{len(beats_over_10)} beats are over 10 seconds and may be worth review for future dense visual pacing.",
                "Review top examples before deciding where optional dense splitting should happen.",
            )
        )
    if estimated_target:
        findings.append(
            _visual_density_finding(
                "DENSE_TARGET_ESTIMATE",
                f"At {VISUAL_DENSITY_PREFERRED_BEAT_SECONDS:.1f}s per image, this runtime estimates about {estimated_target} visual beats.",
                "Use as a planning estimate, not a hard image-count requirement.",
            )
        )

    return {
        "beat_count": beat_count,
        "total_duration_seconds": total_duration,
        "average_beat_seconds": average,
        "median_beat_seconds": median,
        "p90_beat_seconds": p90,
        "longest_beats": _beat_examples(
            sorted(beats, key=lambda beat: (-beat.duration_seconds, beat.beat_number))[:VISUAL_DENSITY_REVIEW_LIMIT],
            "longest_beats",
        ),
        "shortest_beats": _beat_examples(
            sorted(beats, key=lambda beat: (beat.duration_seconds, beat.beat_number))[:VISUAL_DENSITY_REVIEW_LIMIT],
            "shortest_beats",
        ),
        "beats_over_8s": len(beats_over_8),
        "beats_over_10s": len(beats_over_10),
        "beats_over_12s": len(beats_over_12),
        "estimated_dense_target_count": estimated_target,
        "estimated_preferred_beat_seconds": VISUAL_DENSITY_PREFERRED_BEAT_SECONDS,
        "target_range_min": VISUAL_DENSITY_TARGET_RANGE_MIN,
        "target_range_max": VISUAL_DENSITY_TARGET_RANGE_MAX,
        "density_label": _visual_density_label(beat_count, estimated_target, average),
        "findings": findings,
        "review_priority_beats": _review_priority_beats(beats, config_max_duration),
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    midpoint = count // 2
    if count % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _visual_density_label(beat_count: int, estimated_target: int, average_beat_seconds: float) -> str:
    if beat_count == 0:
        return "no_beats"
    lower_bound = max(1, int(math.floor(estimated_target * 0.9))) if estimated_target else 0
    upper_bound = int(math.ceil(estimated_target * 1.15)) if estimated_target else 0
    if estimated_target and beat_count < lower_bound:
        return "usable_sparse"
    if estimated_target and beat_count > upper_bound and average_beat_seconds < 6.5:
        return "over_dense_review"
    if 6.5 <= average_beat_seconds <= 9.0:
        return "dense_target_ready"
    return "usable_standard"


def _visual_density_finding(code: str, message: str, recommendation: str) -> dict[str, Any]:
    return {
        "severity": "info",
        "code": code,
        "message": message,
        "recommendation": recommendation,
    }


def _review_priority_beats(beats: list[Beat], config_max_duration: float) -> list[dict[str, Any]]:
    candidates = []
    for beat in beats:
        reasons = []
        if beat.duration_seconds > config_max_duration:
            reasons.append("exceeds_config_max_duration")
        if beat.duration_seconds > 10.0:
            reasons.append("long_static_stretch_review")
        elif beat.duration_seconds > VISUAL_DENSITY_MIN_REVIEW_SECONDS:
            reasons.append("future_dense_mode_candidate")
        if not reasons:
            continue
        candidates.append((beat.duration_seconds, beat.beat_number, beat, ", ".join(reasons)))

    selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[:VISUAL_DENSITY_REVIEW_LIMIT]
    return [_beat_example(beat, reason) for _, _, beat, reason in selected]


def _beat_examples(beats: list[Beat], reason: str) -> list[dict[str, Any]]:
    return [_beat_example(beat, reason) for beat in beats]


def _beat_example(beat: Beat, reason: str) -> dict[str, Any]:
    return {
        "beat_number": beat.beat_number,
        "start_seconds": round(beat.start_seconds, 3),
        "end_seconds": round(beat.end_seconds, 3),
        "duration_seconds": round(beat.duration_seconds, 3),
        "reason": reason,
        "source_preview": _preview(beat.text_preview, 100),
    }


def _load_beats_for_audit(path: Path, findings: list[dict[str, Any]]) -> list[Beat] | None:
    if not path.exists():
        findings.append(_finding("error", "MISSING_BEATS_JSON", f"Missing core beats file: {path}"))
        return None
    try:
        return load_beats(path)
    except PipelineError as exc:
        findings.append(_finding("error", "INVALID_BEATS_JSON", f"Invalid core beats file: {path} ({exc})"))
        return None


def _load_optional_json(
    path: Path,
    missing_code: str,
    invalid_code: str,
    findings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not path.exists():
        findings.append(_finding("warning", missing_code, f"Optional audit input is missing: {path}"))
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        findings.append(_finding("warning", invalid_code, f"Optional audit input is unreadable: {path} ({exc})"))
        return None
    if not isinstance(payload, dict):
        findings.append(_finding("warning", invalid_code, f"Optional audit input must be a JSON object: {path}"))
        return None
    return payload


def _transcript_segment_count(path: Path, findings: list[dict[str, Any]]) -> int | None:
    if not path.exists():
        findings.append(_finding("info", "MISSING_TRANSCRIPT_SEGMENTS_JSON", f"Optional transcript segments file is missing: {path}"))
        return None
    try:
        return len(load_transcript_segments(path))
    except PipelineError as exc:
        findings.append(_finding("warning", "INVALID_TRANSCRIPT_SEGMENTS_JSON", f"Transcript segments are unreadable: {path} ({exc})"))
        return None


def _prompt_records(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    prompts = payload.get("prompts")
    if not isinstance(prompts, list):
        return []
    return [record for record in prompts if isinstance(record, dict)]


def _prompt_lookup(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for record in records:
        beat_number = record.get("beat_number")
        if isinstance(beat_number, int):
            lookup[beat_number] = record
    return lookup


def _audit_pacing(
    beats: list[Beat],
    prompt_lookup: dict[int, dict[str, Any]],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    first_30 = []
    over_8 = []
    over_10 = []
    over_12 = []
    for beat in beats:
        row = _beat_review_row(beat, prompt_lookup, "Review whether this beat needs extra visual variation.")
        if beat.start_seconds < 30 and beat.duration_seconds > 5:
            first_30.append(row)
            findings.append(
                _finding(
                    "warning",
                    "LONG_BEAT_FIRST_30S",
                    f"Beat {beat.beat_number} is over 5 seconds in the first 30 seconds.",
                    beat.beat_number,
                    "Review hook pacing and consider faster visual variation.",
                )
            )
        if beat.start_seconds >= 30 and beat.duration_seconds > 8:
            over_8.append(row)
            findings.append(
                _finding(
                    "recommendation",
                    "LONG_BEAT_BODY_REVIEW",
                    f"Beat {beat.beat_number} is over 8 seconds.",
                    beat.beat_number,
                    "Review whether this beat needs extra visual variation.",
                )
            )
        if beat.start_seconds >= 30 and beat.duration_seconds > 10:
            over_10.append(row)
            findings.append(
                _finding(
                    "warning",
                    "LONG_BEAT_BODY_HIGH_PRIORITY",
                    f"Beat {beat.beat_number} is over 10 seconds.",
                    beat.beat_number,
                    "Prioritize this beat for visual review before full production.",
                )
            )
        if beat.start_seconds >= 30 and beat.duration_seconds > 12:
            over_12.append(row)
            findings.append(
                _finding(
                    "warning",
                    "LONG_BEAT_BODY_PACING_RISK",
                    f"Beat {beat.beat_number} is over 12 seconds.",
                    beat.beat_number,
                    "Consider sub-beat planning later if the full video feels slow.",
                )
            )
    return {
        "first_30s_over_5s": sorted(first_30, key=lambda item: item["beat_number"]),
        "body_over_8s": sorted(over_8, key=lambda item: item["beat_number"]),
        "body_over_10s": sorted(over_10, key=lambda item: item["beat_number"]),
        "body_over_12s": sorted(over_12, key=lambda item: item["beat_number"]),
    }


def _beat_review_row(
    beat: Beat,
    prompt_lookup: dict[int, dict[str, Any]],
    recommendation: str,
) -> dict[str, Any]:
    prompt = prompt_lookup.get(beat.beat_number, {})
    keywords = prompt.get("visual_concept_keywords")
    source_text = prompt.get("source_text")
    return {
        "beat_number": beat.beat_number,
        "start": beat.start,
        "end": beat.end,
        "duration_seconds": round(beat.duration_seconds, 3),
        "source_text_preview": _preview(source_text if isinstance(source_text, str) else beat.text_preview),
        "visual_concept_keywords": keywords if isinstance(keywords, list) else [],
        "recommendation": recommendation,
    }


def _audit_generated_images(
    beats: list[Beat],
    image_dir: Path,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    if not image_dir.exists():
        findings.append(
            _finding(
                "warning",
                "MISSING_GENERATED_IMAGE",
                f"Generated image directory is missing: {image_dir}",
                recommendation="Generate or save one image per beat before rendering generated-image video.",
            )
        )
    report = validate_generated_images(beats, image_dir)
    issues = []
    for error in report.errors:
        code = _generated_image_code(error)
        issue = _generated_image_issue(error, code)
        issues.append(issue)
        findings.append(_finding("warning", code, error, issue.get("beat_number"), _image_recommendation(code)))
    for warning in report.warnings:
        code = _generated_image_code(warning)
        issue = _generated_image_issue(warning, code)
        issues.append(issue)
        findings.append(_finding("warning", code, warning, issue.get("beat_number"), _image_recommendation(code)))
    issues = sorted(issues, key=lambda item: (item.get("filename", ""), item.get("beat_number") or 0, item["message"]))
    return {
        "expected_count": len(beats),
        "found_count": _generated_image_count(image_dir),
        "ok": report.ok,
        "issues": issues,
    }


def _generated_image_code(message: str) -> str:
    if message.startswith("Missing generated image"):
        return "MISSING_GENERATED_IMAGE"
    if message.startswith("Extra generated image"):
        return "EXTRA_GENERATED_IMAGE"
    return "INVALID_GENERATED_IMAGE"


def _generated_image_issue(message: str, code: str) -> dict[str, Any]:
    beat_number = _beat_number_from_text(message)
    filename = _filename_from_text(message)
    issue: dict[str, Any] = {"code": code, "message": message}
    if beat_number is not None:
        issue["beat_number"] = beat_number
    if filename is not None:
        issue["filename"] = filename
    return issue


def _image_recommendation(code: str) -> str:
    if code == "MISSING_GENERATED_IMAGE":
        return "Save a generated PNG for this beat before generated-image rendering."
    if code == "EXTRA_GENERATED_IMAGE":
        return "Review whether this file belongs to an older run or should be removed manually."
    return "Regenerate or replace the image with a valid 16:9 PNG."


def _audit_prompts(
    payload: dict[str, Any] | None,
    records: list[dict[str, Any]],
    generated_image_dir: Path,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_images_exist = _generated_image_count(generated_image_dir) > 0
    if payload is None:
        if generated_images_exist:
            findings.append(
                _finding(
                    "warning",
                    "PROMPT_STYLE_MISMATCH",
                    "Generated images exist, but prompt metadata is missing.",
                    recommendation="Generated images may be from an older prompt style. Review visual consistency.",
                )
            )
        return {
            "present": False,
            "schema_version": None,
            "style_profile": None,
            "prompt_count": 0,
            "missing_visual_concept_field_count": 0,
        }

    schema_version = payload.get("schema_version")
    style_profile = payload.get("style_profile")
    if schema_version != SCHEMA_VERSION:
        findings.append(
            _finding(
                "warning",
                "PROMPT_SCHEMA_MISMATCH",
                f"Prompt schema_version is {schema_version}; expected {SCHEMA_VERSION}.",
                recommendation="Regenerate prompts before producing final images.",
            )
        )
    if style_profile != STYLE_PROFILE:
        findings.append(
            _finding(
                "warning",
                "PROMPT_STYLE_MISMATCH",
                f"Prompt style_profile is {style_profile}; expected {STYLE_PROFILE}.",
                recommendation="Generated images may be from an older prompt style. Review visual consistency.",
            )
        )

    missing_fields = 0
    for record in records:
        if "visual_concept_text" not in record or "visual_concept_keywords" not in record:
            missing_fields += 1
            findings.append(
                _finding(
                    "warning",
                    "MISSING_VISUAL_CONCEPT_FIELDS",
                    f"Prompt record for beat {record.get('beat_number', '?')} is missing visual concept fields.",
                    record.get("beat_number") if isinstance(record.get("beat_number"), int) else None,
                    "Regenerate prompts so visual concept debugging fields are present.",
                )
            )
    return {
        "present": True,
        "schema_version": schema_version,
        "style_profile": style_profile,
        "prompt_count": len(records),
        "missing_visual_concept_field_count": missing_fields,
    }


def _audit_metaphors(records: list[dict[str, Any]], findings: list[dict[str, Any]]) -> dict[str, Any]:
    keyword_counter: Counter[str] = Counter()
    consecutive_repeats = []
    fallback_count = 0
    previous_keywords: set[str] = set()
    for record in sorted(records, key=lambda item: item.get("beat_number", 0)):
        raw_keywords = record.get("visual_concept_keywords")
        keywords = [str(keyword) for keyword in raw_keywords] if isinstance(raw_keywords, list) else []
        if not keywords:
            fallback_count += 1
        keyword_counter.update(keywords)
        current_keywords = set(keywords)
        repeated = sorted(previous_keywords & current_keywords)
        if repeated:
            consecutive_repeats.append({"beat_number": record.get("beat_number"), "keywords": repeated})
        previous_keywords = current_keywords

    keyword_distribution = [
        {"keyword": keyword, "count": count}
        for keyword, count in sorted(keyword_counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    prompt_count = len(records)
    if prompt_count and fallback_count / prompt_count >= 0.35:
        findings.append(
            _finding(
                "warning",
                "METAPHOR_FALLBACK_OVERUSE",
                f"{fallback_count} of {prompt_count} prompts use fallback visual concepts.",
                recommendation="Review fallback beats and regenerate weak concepts before full production.",
            )
        )
    for item in keyword_distribution:
        if item["count"] >= 3 or (item["keyword"] == "time" and item["count"] >= 2):
            findings.append(
                _finding(
                    "warning",
                    "METAPHOR_REPETITION_RISK",
                    f"Metaphor keyword '{item['keyword']}' appears {item['count']} times.",
                    recommendation="Review repeated metaphor families for visual variety.",
                )
            )
    if consecutive_repeats:
        findings.append(
            _finding(
                "warning",
                "METAPHOR_REPETITION_RISK",
                "Consecutive beats reuse the same visual concept keywords.",
                recommendation="Review consecutive repeated metaphors for pacing and variety.",
            )
        )
    return {
        "keyword_distribution": keyword_distribution,
        "fallback_count": fallback_count,
        "consecutive_repeats": consecutive_repeats,
    }


def _audit_brightness(beats: list[Beat], image_dir: Path, findings: list[dict[str, Any]]) -> dict[str, Any]:
    images = []
    if not image_dir.exists():
        return {"images": images}
    for beat in beats:
        image_path = image_dir / generated_image_filename(beat)
        if not image_path.exists() or not image_path.is_file():
            continue
        try:
            with Image.open(image_path) as image:
                grayscale = image.convert("L")
                stat = ImageStat.Stat(grayscale)
                average = float(stat.mean[0])
                histogram = grayscale.histogram()
                total_pixels = grayscale.width * grayscale.height
                near_white_ratio = sum(histogram[245:256]) / total_pixels
                near_black_ratio = sum(histogram[0:16]) / total_pixels
        except (OSError, UnidentifiedImageError):
            continue

        metrics = {
            "beat_number": beat.beat_number,
            "filename": image_path.name,
            "average_brightness": round(average, 2),
            "near_white_ratio": round(near_white_ratio, 4),
            "near_black_ratio": round(near_black_ratio, 4),
        }
        images.append(metrics)
        if average < 30:
            findings.append(
                _finding(
                    "warning",
                    "IMAGE_TOO_DARK",
                    f"{image_path.name} average brightness is below 30.",
                    beat.beat_number,
                    "Review image contrast before final render.",
                )
            )
        if average > 210:
            findings.append(
                _finding(
                    "warning",
                    "IMAGE_TOO_BRIGHT",
                    f"{image_path.name} average brightness is above 210.",
                    beat.beat_number,
                    "Review whether this image is too bright for the V2.3 style.",
                )
            )
        if near_white_ratio > 0.35:
            findings.append(
                _finding(
                    "warning",
                    "POSSIBLE_WHITEBOARD_FRAME",
                    f"{image_path.name} has a high near-white pixel ratio.",
                    beat.beat_number,
                    "Review for possible whiteboard or clinical visual drift.",
                )
            )
    return {"images": sorted(images, key=lambda item: item["beat_number"])}


def _audit_render_risk(beats: list[Beat], fps: int, findings: list[dict[str, Any]]) -> dict[str, Any]:
    duration = sum(beat.duration_seconds for beat in beats)
    frame_count = int(math.ceil(duration * fps))
    warnings = []
    if duration >= 8 * 60:
        warnings.append("Video duration exceeds 8 minutes.")
    if frame_count >= 14400:
        warnings.append("Estimated frame count is high for kinetic MoviePy rendering.")
    if len(beats) >= 60:
        warnings.append("Beat count is large.")
    if warnings:
        findings.append(
            _finding(
                "warning",
                "KINETIC_RENDER_TIMEOUT_RISK",
                "Kinetic render may require a longer timeout for full-length production.",
                recommendation="Plan a longer render window and avoid running production renders as quick checks.",
            )
        )
    return {
        "total_duration_seconds": round(duration, 3),
        "fps": fps,
        "frame_count_estimate": frame_count,
        "beat_count": len(beats),
        "kinetic_mode_available": True,
        "warnings": warnings,
    }


def _audit_outputs(output_dir: Path, findings: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = []
    for filename in ("final_video.mp4", "final_video_generated.mp4", "final_video_generated_kinetic.mp4"):
        path = output_dir / filename
        exists = path.exists()
        outputs.append({"path": path.as_posix(), "exists": exists})
        if exists:
            findings.append(
                _finding(
                    "warning",
                    "OUTPUT_FILE_EXISTS",
                    f"Render output already exists and may be overwritten: {path}",
                    recommendation="Move or rename existing output before rendering if it should be preserved.",
                )
            )
    return {"fixed_outputs": outputs}


def _audit_artifact_safety(base_dir: Path) -> dict[str, Any]:
    gitignore_path = base_dir / ".gitignore"
    patterns = _gitignore_patterns(gitignore_path)
    paths = ["data/", "assets/generated_images/", "assets/generated_images_backups/", "output/"]
    return {
        "gitignore_present": gitignore_path.exists(),
        "paths": [
            {
                "path": path,
                "present": (base_dir / path).exists(),
                "appears_ignored": path in patterns,
            }
            for path in paths
        ],
    }


def _audit_source_text(
    beats: list[Beat],
    prompt_lookup: dict[int, dict[str, Any]],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    warnings = []
    for beat in beats:
        record = prompt_lookup.get(beat.beat_number, {})
        text = record.get("source_text") if isinstance(record.get("source_text"), str) else beat.text_preview
        reasons = _source_text_split_reasons(text)
        if not reasons:
            continue
        item = {
            "beat_number": beat.beat_number,
            "source_text_preview": _preview(text),
            "reasons": reasons,
        }
        warnings.append(item)
        findings.append(
            _finding(
                "warning",
                "SOURCE_TEXT_SPLIT_WARNING",
                f"Beat {beat.beat_number} may have an awkward source_text boundary.",
                beat.beat_number,
                "Review transcript beat boundaries before generating final images.",
            )
        )
    return {"warnings": sorted(warnings, key=lambda item: item["beat_number"])}


def _add_info_findings(
    beats: list[Beat],
    fps: int,
    transcript_segment_count: int | None,
    generated_image_count_found: int,
    prompt_payload: dict[str, Any] | None,
    findings: list[dict[str, Any]],
) -> None:
    total_duration = round(sum(beat.duration_seconds for beat in beats), 3)
    frame_count = int(math.ceil(total_duration * fps))
    findings.append(
        _finding(
            "info",
            "AUDIT_SUMMARY",
            f"Audit covers {len(beats)} beats, {total_duration:.3f}s, and about {frame_count} frames at {fps}fps.",
        )
    )
    findings.append(
        _finding(
            "info",
            "GENERATED_IMAGE_STATUS",
            f"Generated image files found: {generated_image_count_found}; expected: {len(beats)}.",
        )
    )
    if transcript_segment_count is not None:
        findings.append(
            _finding(
                "info",
                "TRANSCRIPT_SEGMENT_STATUS",
                f"Transcript segment count: {transcript_segment_count}.",
            )
        )
    if prompt_payload is not None:
        findings.append(
            _finding(
                "info",
                "PROMPT_STYLE_STATUS",
                f"Prompt schema: {prompt_payload.get('schema_version')}; style profile: {prompt_payload.get('style_profile')}.",
            )
        )


def _source_text_split_reasons(text: str) -> list[str]:
    compact = " ".join(text.split())
    if not compact:
        return []
    reasons = []
    first_word = re.sub(r"[^A-Za-z']", "", compact.split()[0]).lower()
    if first_word in CONTINUATION_WORDS:
        reasons.append("starts_with_continuation_word")
    if compact[-1] not in PUNCTUATION_ENDINGS:
        reasons.append("ends_without_punctuation")
    tail_words = compact.rstrip(".,!?").split()[-3:]
    if 0 < len(tail_words) <= 3 and compact[-1] not in PUNCTUATION_ENDINGS:
        reasons.append("ends_with_short_incomplete_phrase")
    return reasons


def _build_report(
    config: PipelineConfig,
    beats: list[Beat],
    transcript_segment_count: int | None,
    generated_image_count_found: int,
    sections: dict[str, dict[str, Any]],
    findings: list[dict[str, Any]],
    recommendations: list[str],
) -> dict[str, Any]:
    total_duration = round(sum(beat.duration_seconds for beat in beats), 3)
    frame_count = int(math.ceil(total_duration * config.video.fps))
    sorted_findings = _sort_findings(findings)
    score, score_reasons = _readiness_score(sorted_findings)
    label = _readiness_label(score, sorted_findings)
    report = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "readiness_score": score,
        "readiness_label": label,
        "summary": {
            "total_duration_seconds": total_duration,
            "fps": config.video.fps,
            "frame_count_estimate": frame_count,
            "beat_count": len(beats),
            "transcript_segment_count": transcript_segment_count,
            "generated_image_count_expected": len(beats),
            "generated_image_count_found": generated_image_count_found,
        },
        "sections": sections,
        "findings": sorted_findings,
        "recommendations": recommendations,
        "score_reasons": score_reasons,
    }
    return report


def _readiness_score(findings: list[dict[str, Any]]) -> tuple[float, list[str]]:
    if any(finding["severity"] == "error" for finding in findings):
        return 0.0, ["Core audit data is missing or invalid."]

    deductions = {
        "pacing": 0.0,
        "generated_images": 0.0,
        "prompts": 0.0,
        "metaphors": 0.0,
        "brightness": 0.0,
        "render_risk": 0.0,
        "artifact_completeness": 0.0,
    }
    for finding in findings:
        code = finding["code"]
        if code == "LONG_BEAT_FIRST_30S":
            deductions["pacing"] += 0.35
        elif code == "LONG_BEAT_BODY_REVIEW":
            deductions["pacing"] += 0.15
        elif code == "LONG_BEAT_BODY_HIGH_PRIORITY":
            deductions["pacing"] += 0.3
        elif code == "LONG_BEAT_BODY_PACING_RISK":
            deductions["pacing"] += 0.45
        elif code == "MISSING_GENERATED_IMAGE":
            deductions["generated_images"] += 0.2
        elif code == "INVALID_GENERATED_IMAGE":
            deductions["generated_images"] += 0.35
        elif code == "EXTRA_GENERATED_IMAGE":
            deductions["artifact_completeness"] += 0.1
        elif code in {"PROMPT_SCHEMA_MISMATCH", "PROMPT_STYLE_MISMATCH", "MISSING_VISUAL_CONCEPT_FIELDS"}:
            deductions["prompts"] += 0.5
        elif code in {"METAPHOR_FALLBACK_OVERUSE", "METAPHOR_REPETITION_RISK"}:
            deductions["metaphors"] += 0.35
        elif code in {"IMAGE_TOO_DARK", "IMAGE_TOO_BRIGHT", "POSSIBLE_WHITEBOARD_FRAME"}:
            deductions["brightness"] += 0.15
        elif code == "KINETIC_RENDER_TIMEOUT_RISK":
            deductions["render_risk"] += 0.6
        elif code in {"OUTPUT_FILE_EXISTS", "SOURCE_TEXT_SPLIT_WARNING"}:
            deductions["artifact_completeness"] += 0.1

    capped = {
        "pacing": min(deductions["pacing"], 2.5),
        "generated_images": min(deductions["generated_images"], 2.5),
        "prompts": min(deductions["prompts"], 1.5),
        "metaphors": min(deductions["metaphors"], 1.0),
        "brightness": min(deductions["brightness"], 1.0),
        "render_risk": min(deductions["render_risk"], 1.0),
        "artifact_completeness": min(deductions["artifact_completeness"], 1.0),
    }
    score = max(0.0, min(10.0, 10.0 - sum(capped.values())))
    reasons = [
        f"{name.replace('_', ' ')}: -{value:.1f}"
        for name, value in capped.items()
        if value > 0
    ]
    if not reasons:
        reasons.append("No major production-readiness deductions.")
    return round(score, 1), reasons


def _readiness_label(score: float, findings: list[dict[str, Any]]) -> str:
    if any(finding["severity"] == "error" for finding in findings):
        return "blocked"
    major_codes = {"LONG_BEAT_BODY_PACING_RISK", "INVALID_GENERATED_IMAGE", "PROMPT_STYLE_MISMATCH", "KINETIC_RENDER_TIMEOUT_RISK"}
    if score < 6.0 or any(finding["code"] in major_codes for finding in findings):
        return "production_risky"
    if score < 8.5 or any(finding["severity"] == "warning" for finding in findings):
        return "needs_review"
    return "prototype_ready"


def _recommendations(findings: list[dict[str, Any]]) -> list[str]:
    priority = [
        ("MISSING_GENERATED_IMAGE", "Add missing generated images before rendering generated-image video."),
        ("INVALID_GENERATED_IMAGE", "Replace invalid generated images with valid 16:9 PNG files."),
        ("LONG_BEAT_BODY_PACING_RISK", "Review beats over 12 seconds for pacing risk."),
        ("LONG_BEAT_BODY_HIGH_PRIORITY", "Review beats over 10 seconds before full production."),
        ("LONG_BEAT_FIRST_30S", "Tighten first-30-second visual pacing before a full-length release."),
        ("PROMPT_STYLE_MISMATCH", "Regenerate prompts or review images for possible style mixing."),
        ("PROMPT_SCHEMA_MISMATCH", "Regenerate prompts with the current V2.3 schema before image generation."),
        ("METAPHOR_FALLBACK_OVERUSE", "Review fallback concepts and strengthen weak prompt concepts."),
        ("METAPHOR_REPETITION_RISK", "Review repeated metaphor keywords for visual variety."),
        ("IMAGE_TOO_DARK", "Review very dark generated images for readability."),
        ("IMAGE_TOO_BRIGHT", "Review overly bright generated images for style drift."),
        ("POSSIBLE_WHITEBOARD_FRAME", "Review near-white images for possible whiteboard or clinical drift."),
        ("KINETIC_RENDER_TIMEOUT_RISK", "Reserve a longer render window for full-length kinetic exports."),
        ("OUTPUT_FILE_EXISTS", "Move or rename existing output videos before rendering if they should be preserved."),
        ("SOURCE_TEXT_SPLIT_WARNING", "Review possible awkward transcript beat boundaries."),
    ]
    codes = {finding["code"] for finding in findings}
    return [message for code, message in priority if code in codes]


def _write_reports(report: dict[str, Any], data_dir: Path, base_dir: Path) -> AuditResult:
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / JSON_REPORT_NAME
    markdown_path = data_dir / MARKDOWN_REPORT_NAME
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return AuditResult(report=report, json_path=json_path, markdown_path=markdown_path)


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    findings = report["findings"]
    warnings = [finding for finding in findings if finding["severity"] == "warning"]
    lines = [
        "# Production Audit Report",
        "",
        f"Readiness: {report['readiness_score']:.1f}/10 `{report['readiness_label']}`",
        "",
        "## Score Notes",
        "",
    ]
    lines.extend(f"- {reason}" for reason in report.get("score_reasons", []))
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Duration: {summary['total_duration_seconds']:.3f}s",
            f"- FPS: {summary['fps']}",
            f"- Estimated frames: {summary['frame_count_estimate']}",
            f"- Beats: {summary['beat_count']}",
            f"- Transcript segments: {summary['transcript_segment_count']}",
            f"- Generated images: {summary['generated_image_count_found']} found / {summary['generated_image_count_expected']} expected",
            "",
            "## Visual Density",
            "",
        ]
    )
    lines.extend(_visual_density_markdown(report["sections"].get("visual_density", {})))
    lines.extend(
        [
            "## Top Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- `{finding['code']}`: {finding['message']}" for finding in warnings[:10])
    else:
        lines.append("- None")
    lines.extend(["", "## Long Beats", ""])
    lines.extend(_long_beat_markdown(report["sections"].get("pacing", {})))
    lines.extend(["", "## Generated Images", ""])
    generated = report["sections"].get("generated_images", {})
    lines.append(f"- Status: {'ok' if generated.get('ok') else 'needs review'}")
    lines.append(f"- Found: {generated.get('found_count', 0)} / {generated.get('expected_count', 0)}")
    lines.extend(["", "## Prompt / Style", ""])
    prompts = report["sections"].get("prompts", {})
    lines.append(f"- Present: {prompts.get('present')}")
    lines.append(f"- Schema version: {prompts.get('schema_version')}")
    lines.append(f"- Style profile: {prompts.get('style_profile')}")
    lines.extend(["", "## Metaphors", ""])
    metaphor_lines = [
        f"- {item['keyword']}: {item['count']}"
        for item in report["sections"].get("metaphors", {}).get("keyword_distribution", [])
    ]
    lines.extend(metaphor_lines or ["- No keyword distribution available"])
    lines.extend(["", "## Brightness", ""])
    brightness_warnings = [finding for finding in warnings if finding["code"] in {"IMAGE_TOO_DARK", "IMAGE_TOO_BRIGHT", "POSSIBLE_WHITEBOARD_FRAME"}]
    lines.extend(f"- `{finding['code']}`: {finding['message']}" for finding in brightness_warnings[:10])
    if not brightness_warnings:
        lines.append("- No brightness warnings")
    lines.extend(["", "## Render Risk", ""])
    render = report["sections"].get("render_risk", {})
    lines.append(f"- Estimated frame count: {render.get('frame_count_estimate', summary['frame_count_estimate'])}")
    render_warnings = render.get("warnings", [])
    lines.extend(f"- {warning}" for warning in render_warnings)
    if not render_warnings:
        lines.append("- No render-risk warnings")
    lines.extend(["", "## Overwrite Warnings", ""])
    output_warnings = [finding for finding in warnings if finding["code"] == "OUTPUT_FILE_EXISTS"]
    lines.extend(f"- {finding['message']}" for finding in output_warnings)
    if not output_warnings:
        lines.append("- No fixed output files currently exist")
    lines.extend(["", "## Next Recommended Actions", ""])
    lines.extend(f"- {recommendation}" for recommendation in report["recommendations"])
    if not report["recommendations"]:
        lines.append("- No priority recommendations")
    lines.append("")
    return "\n".join(lines)


def _visual_density_markdown(visual_density: dict[str, Any]) -> list[str]:
    if not visual_density:
        return ["- No visual-density diagnostics available", ""]

    lines = [
        f"- Density label: `{visual_density.get('density_label')}`",
        f"- Beats: {visual_density.get('beat_count', 0)}",
        f"- Average beat: {visual_density.get('average_beat_seconds', 0):.3f}s",
        f"- Median beat: {visual_density.get('median_beat_seconds', 0):.3f}s",
        f"- P90 beat: {visual_density.get('p90_beat_seconds', 0):.3f}s",
        f"- Long beats: {visual_density.get('beats_over_8s', 0)} over 8s, "
        f"{visual_density.get('beats_over_10s', 0)} over 10s, "
        f"{visual_density.get('beats_over_12s', 0)} over 12s",
        f"- Dense target estimate: {visual_density.get('estimated_dense_target_count', 0)} beats at "
        f"{visual_density.get('estimated_preferred_beat_seconds', 0):.1f}s preferred average "
        f"(future target range {visual_density.get('target_range_min', 0)}-{visual_density.get('target_range_max', 0)})",
    ]
    findings = visual_density.get("findings", [])
    if findings:
        lines.append("- Diagnostic notes:")
        lines.extend(f"  - `{finding['code']}`: {finding['message']}" for finding in findings[:VISUAL_DENSITY_REVIEW_LIMIT])
    else:
        lines.append("- Diagnostic notes: none")

    priority = visual_density.get("review_priority_beats", [])
    lines.extend(["", "### Review Priority Beats", ""])
    if not priority:
        lines.append("- None")
    else:
        lines.append("| Beat | Start | End | Duration | Reason | Source Preview |")
        lines.append("| --- | ---: | ---: | ---: | --- | --- |")
        for item in priority[:VISUAL_DENSITY_REVIEW_LIMIT]:
            lines.append(
                f"| {item['beat_number']} | {item['start_seconds']:.3f} | {item['end_seconds']:.3f} | "
                f"{item['duration_seconds']:.3f}s | {item['reason']} | {_escape_markdown_table_text(item['source_preview'])} |"
            )
    lines.append("")
    return lines


def _escape_markdown_table_text(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _long_beat_markdown(pacing: dict[str, Any]) -> list[str]:
    rows = []
    for key, title in (
        ("first_30s_over_5s", "First 30s over 5s"),
        ("body_over_8s", "Body over 8s"),
        ("body_over_10s", "Body over 10s"),
        ("body_over_12s", "Body over 12s"),
    ):
        rows.append(f"### {title}")
        rows.append("")
        items = pacing.get(key, [])
        if not items:
            rows.append("- None")
            rows.append("")
            continue
        rows.append("| Beat | Start | End | Duration | Keywords | Recommendation |")
        rows.append("| --- | --- | --- | ---: | --- | --- |")
        for item in items:
            keywords = ", ".join(str(keyword) for keyword in item.get("visual_concept_keywords", [])) or "-"
            rows.append(
                f"| {item['beat_number']} | {item['start']} | {item['end']} | "
                f"{item['duration_seconds']:.3f}s | {keywords} | {item['recommendation']} |"
            )
        rows.append("")
    return rows


def _sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        findings,
        key=lambda finding: (
            SEVERITY_ORDER.get(finding["severity"], 99),
            finding["code"],
            finding.get("beat_number", 999999),
            finding["message"],
        ),
    )


def _finding(
    severity: str,
    code: str,
    message: str,
    beat_number: int | None = None,
    recommendation: str | None = None,
) -> dict[str, Any]:
    finding: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if beat_number is not None:
        finding["beat_number"] = beat_number
    if recommendation is not None:
        finding["recommendation"] = recommendation
    return finding


def _generated_image_count(image_dir: Path) -> int:
    if not image_dir.exists():
        return 0
    return sum(1 for path in image_dir.iterdir() if path.is_file())


def _gitignore_patterns(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _beat_number_from_text(text: str) -> int | None:
    match = re.search(r"beat (\d{3})", text)
    if match:
        return int(match.group(1))
    return None


def _filename_from_text(text: str) -> str | None:
    match = re.search(r"(beat_\d{3}\.png)", text)
    if match:
        return match.group(1)
    return None


def _preview(text: str, max_chars: int = 100) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
