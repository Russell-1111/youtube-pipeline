from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from .beat_io import load_beats
from .config import PipelineConfig
from .dense_images import DENSE_IMAGE_DIR_NAME, dense_image_filename
from .errors import InputFileError, PipelineError
from .models import Beat
from .render import get_audio_duration, render_video_kinetic
from .render_ffmpeg import FFmpegRenderResult, render_video_kinetic_ffmpeg

DENSE_BEAT_COUNT = 70
PREVIEW_BEATS_NAME = "beats_dense_preview.json"
IMAGE_REPORT_NAME = "dense_image_generation_report.json"
QA_REPORT_NAME = "dense_image_visual_qa.json"
JSON_REPORT_NAME = "dense_render_report.json"
MARKDOWN_REPORT_NAME = "dense_render_report.md"
OUTPUT_NAME = "final_dense_preview.mp4"

FFmpegRenderer = Callable[..., FFmpegRenderResult]
MoviePyRenderer = Callable[[list[Beat], Path, Path, int, int, int], None]


@dataclass(frozen=True)
class DenseRenderResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path
    output_path: Path


def run_dense_preview_render(
    config: PipelineConfig,
    base_dir: Path,
    *,
    ffmpeg_renderer: FFmpegRenderer | None = None,
    fallback_renderer: MoviePyRenderer | None = None,
) -> DenseRenderResult:
    ffmpeg_renderer = ffmpeg_renderer or render_video_kinetic_ffmpeg
    fallback_renderer = fallback_renderer or render_video_kinetic
    data_dir = config.outputs.data_dir
    beats_path = data_dir / PREVIEW_BEATS_NAME
    image_report_path = data_dir / IMAGE_REPORT_NAME
    qa_report_path = data_dir / QA_REPORT_NAME
    json_path = data_dir / JSON_REPORT_NAME
    markdown_path = data_dir / MARKDOWN_REPORT_NAME
    image_dir = config.outputs.images_dir.parent / DENSE_IMAGE_DIR_NAME
    output_path = base_dir / "output" / OUTPUT_NAME
    work_dir = output_path.parent / "benchmarks" / "final_dense_preview_segments"

    _validate_output_path(output_path, base_dir)
    _validate_voiceover(config.inputs.voiceover)
    beats = load_beats(beats_path)
    image_report = _load_json_object(image_report_path, "dense image generation report")
    qa_report = _load_json_object(qa_report_path, "dense visual QA report")
    dense_beats = _validate_and_map_dense_inputs(
        beats=beats,
        image_report=image_report,
        qa_report=qa_report,
        image_dir=image_dir,
        base_dir=base_dir,
    )

    renderer_name = "ffmpeg"
    render_error: str | None = None
    audio_duration = get_audio_duration(config.inputs.voiceover)
    try:
        ffmpeg_result = ffmpeg_renderer(
            dense_beats,
            config.inputs.voiceover,
            output_path,
            config.video.fps,
            config.video.width,
            config.video.height,
            work_dir=work_dir,
            audio_duration_seconds=audio_duration,
        )
        video_duration = ffmpeg_result.video_duration_seconds
        motion_note = ffmpeg_result.motion_equivalence_note
    except PipelineError as exc:
        render_error = str(exc)
        renderer_name = "moviepy"
        fallback_renderer(
            dense_beats,
            config.inputs.voiceover,
            output_path,
            config.video.fps,
            config.video.width,
            config.video.height,
        )
        video_duration = audio_duration
        motion_note = f"FFmpeg renderer failed and MoviePy kinetic fallback was used: {render_error}"

    report = _build_report(
        dense_beats=dense_beats,
        image_report=image_report,
        qa_report=qa_report,
        beats_path=beats_path,
        image_report_path=image_report_path,
        qa_report_path=qa_report_path,
        image_dir=image_dir,
        output_path=output_path,
        json_path=json_path,
        markdown_path=markdown_path,
        renderer_name=renderer_name,
        ffmpeg_error=render_error,
        video_duration_seconds=video_duration,
        audio_duration_seconds=audio_duration,
        fps=config.video.fps,
        width=config.video.width,
        height=config.video.height,
        motion_note=motion_note,
        base_dir=base_dir,
    )
    markdown = _markdown_report(report)
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return DenseRenderResult(report=report, json_path=json_path, markdown_path=markdown_path, output_path=output_path)


def print_dense_render_summary(result: DenseRenderResult, base_dir: Path) -> None:
    report = result.report
    print("Dense preview render complete")
    print(f"Renderer: {report['renderer']}")
    print(f"Dense beats rendered: {report['dense_beat_count']}")
    print(f"Final dense preview: {_display_path(result.output_path, base_dir)}")
    print(f"Video duration seconds: {report['video']['duration_seconds']}")
    print(f"Resolution/FPS: {report['video']['width']}x{report['video']['height']} @ {report['video']['fps']}")
    print("Reports written:")
    print(f"- {_display_path(result.json_path, base_dir)}")
    print(f"- {_display_path(result.markdown_path, base_dir)}")


def _validate_and_map_dense_inputs(
    *,
    beats: list[Beat],
    image_report: dict[str, Any],
    qa_report: dict[str, Any],
    image_dir: Path,
    base_dir: Path,
) -> list[Beat]:
    _require(len(beats) == DENSE_BEAT_COUNT, f"Dense beat count must be {DENSE_BEAT_COUNT}; found {len(beats)}.")
    _require(image_report.get("ready_for_dense_render") is True, "Dense image report is not ready for dense render.")
    _require(image_report.get("expected_image_count") == DENSE_BEAT_COUNT, "Dense image expected count must match dense beat count.")
    _require(image_report.get("present_image_count") == DENSE_BEAT_COUNT, "Dense image present count must match dense beat count.")
    _require(image_report.get("missing_image_count") == 0, "Dense image report has missing images.")
    _require(image_report.get("invalid_image_count") == 0, "Dense image report has invalid images.")
    _require(image_report.get("warning_count") == 0, "Dense image report has warnings.")
    _require(qa_report.get("ready_for_dense_render_planning") is True, "Dense visual QA is not ready for dense render planning.")
    _require(qa_report.get("fail_count") == 0, "Dense visual QA has failures.")
    _require(qa_report.get("total_images_reviewed") == DENSE_BEAT_COUNT, "Dense visual QA reviewed count must match dense beat count.")

    image_rows = image_report.get("images")
    _require(isinstance(image_rows, list), "Dense image report must contain an images list.")
    expected_from_report = _valid_expected_images(image_rows)
    _require(set(expected_from_report) == set(range(1, DENSE_BEAT_COUNT + 1)), "Dense image report does not cover every dense beat.")

    qa_rows = qa_report.get("images")
    _require(isinstance(qa_rows, list), "Dense visual QA report must contain an images list.")
    qa_numbers = {row.get("dense_beat_number") for row in qa_rows if isinstance(row, dict)}
    _require(qa_numbers == set(range(1, DENSE_BEAT_COUNT + 1)), "Dense visual QA report does not cover every dense beat.")

    mapped: list[Beat] = []
    for expected_number, beat in enumerate(beats, start=1):
        _require(beat.beat_number == expected_number, f"Dense beat numbering must be sequential; expected {expected_number}, found {beat.beat_number}.")
        image_path = image_dir / dense_image_filename(expected_number)
        _require(image_path.exists() and image_path.is_file(), f"Missing dense image for beat {expected_number:03}: {image_path}")
        _require(expected_from_report[expected_number] == _display_path(image_path, base_dir), f"Dense image report path mismatch for beat {expected_number:03}.")
        mapped.append(replace(beat, image_path=image_path.as_posix()))
    return mapped


def _valid_expected_images(image_rows: list[Any]) -> dict[int, str]:
    by_number: dict[int, str] = {}
    for row in image_rows:
        if not isinstance(row, dict):
            continue
        beat_number = row.get("dense_beat_number")
        if not isinstance(beat_number, int):
            continue
        expected_filename = row.get("expected_filename")
        expected_path = row.get("expected_path")
        if row.get("status") != "valid":
            continue
        if expected_filename != dense_image_filename(beat_number):
            continue
        if not isinstance(expected_path, str):
            continue
        by_number[beat_number] = expected_path
    return by_number


def _build_report(
    *,
    dense_beats: list[Beat],
    image_report: dict[str, Any],
    qa_report: dict[str, Any],
    beats_path: Path,
    image_report_path: Path,
    qa_report_path: Path,
    image_dir: Path,
    output_path: Path,
    json_path: Path,
    markdown_path: Path,
    renderer_name: str,
    ffmpeg_error: str | None,
    video_duration_seconds: float,
    audio_duration_seconds: float,
    fps: int,
    width: int,
    height: int,
    motion_note: str,
    base_dir: Path,
) -> dict[str, Any]:
    return {
        "report_schema_version": 1,
        "created_at": _utc_now(),
        "renderer": renderer_name,
        "ffmpeg_error": ffmpeg_error,
        "dense_beat_count": len(dense_beats),
        "source_paths": {
            "dense_preview_beats": _display_path(beats_path, base_dir),
            "dense_image_generation_report": _display_path(image_report_path, base_dir),
            "dense_visual_qa": _display_path(qa_report_path, base_dir),
            "dense_image_directory": _display_path(image_dir, base_dir),
        },
        "output_paths": {
            "video": _display_path(output_path, base_dir),
            "json": _display_path(json_path, base_dir),
            "markdown": _display_path(markdown_path, base_dir),
        },
        "video": {
            "duration_seconds": round(video_duration_seconds, 3),
            "audio_duration_seconds": round(audio_duration_seconds, 3),
            "width": width,
            "height": height,
            "fps": fps,
            "motion_note": motion_note,
        },
        "safety_gates": {
            "dense_beat_count": len(dense_beats),
            "ready_for_dense_render": image_report.get("ready_for_dense_render"),
            "expected_image_count": image_report.get("expected_image_count"),
            "present_image_count": image_report.get("present_image_count"),
            "missing_image_count": image_report.get("missing_image_count"),
            "invalid_image_count": image_report.get("invalid_image_count"),
            "warning_count": image_report.get("warning_count"),
            "ready_for_dense_render_planning": qa_report.get("ready_for_dense_render_planning"),
            "qa_fail_count": qa_report.get("fail_count"),
            "qa_reviewed_count": qa_report.get("total_images_reviewed"),
            "output_path_verified": _display_path(output_path, base_dir) == "output/final_dense_preview.mp4",
        },
        "dense_images": [
            {
                "dense_beat_number": beat.beat_number,
                "image_path": _display_path(Path(beat.image_path), base_dir),
            }
            for beat in dense_beats
        ],
    }


def _markdown_report(report: dict[str, Any]) -> str:
    gates = report["safety_gates"]
    lines = [
        "# Dense Render Report",
        "",
        "## Summary",
        "",
        f"- Renderer: `{report['renderer']}`",
        f"- Dense beats rendered: {report['dense_beat_count']}",
        f"- Output video: `{report['output_paths']['video']}`",
        f"- Duration seconds: {report['video']['duration_seconds']}",
        f"- Resolution/FPS: {report['video']['width']}x{report['video']['height']} @ {report['video']['fps']}",
        "",
        "## Safety gates",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in gates.items())
    lines.extend(["", "## Dense image mapping", ""])
    lines.append("| Dense Beat | Image Path |")
    lines.append("| ---: | --- |")
    for row in report["dense_images"]:
        lines.append(f"| {row['dense_beat_number']} | `{row['image_path']}` |")
    lines.append("")
    return "\n".join(lines)


def _validate_output_path(output_path: Path, base_dir: Path) -> None:
    _require(_display_path(output_path, base_dir) == "output/final_dense_preview.mp4", "Dense render output path must be output/final_dense_preview.mp4.")


def _validate_voiceover(voiceover: Path) -> None:
    if not voiceover.exists():
        raise InputFileError(f"Missing voiceover file: {voiceover}")


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
