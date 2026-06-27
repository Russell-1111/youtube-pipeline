from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path
from typing import Any

from .beat_io import load_beats
from .config import PipelineConfig
from .dense_images import DENSE_IMAGE_DIR_NAME, dense_image_filename
from .dense_render import DENSE_BEAT_COUNT
from .errors import InputFileError, PipelineError

PREVIEW_BEATS_NAME = "beats_dense_preview.json"
PREVIEW_PROMPTS_NAME = "image_prompts_dense_preview.json"
DENSE_BEAT_REVIEW_NAME = "dense_beat_review.json"
DENSE_IMAGE_REPORT_NAME = "dense_image_generation_report.json"
DENSE_VISUAL_QA_NAME = "dense_image_visual_qa.json"
DENSE_RENDER_REPORT_NAME = "dense_render_report.json"
DENSE_PLAYBACK_QA_NAME = "dense_render_playback_qa.json"
JSON_REPORT_NAME = "dense_handoff_report.json"
MARKDOWN_REPORT_NAME = "dense_handoff_report.md"
PREVIEW_VIDEO_PATH = Path("output/final_dense_preview.mp4")

PROTECTED_PATHS = (
    "data/beats.json",
    "data/image_prompts.json",
    "data/transcript_segments.json",
    "assets/generated_images/",
    "output/final_video.mp4",
    "output/final_video_generated.mp4",
    "output/final_video_generated_kinetic.mp4",
    "output/final_video_generated_kinetic_ffmpeg.mp4",
)


@dataclass(frozen=True)
class DenseHandoffResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path

    @property
    def ready(self) -> bool:
        return self.report["status"] == "ready"


def run_dense_handoff_preparation(config: PipelineConfig, base_dir: Path) -> DenseHandoffResult:
    data_dir = config.outputs.data_dir
    json_path = data_dir / JSON_REPORT_NAME
    markdown_path = data_dir / MARKDOWN_REPORT_NAME
    image_dir = config.outputs.images_dir.parent / DENSE_IMAGE_DIR_NAME
    preview_video_path = base_dir / PREVIEW_VIDEO_PATH

    paths = {
        "dense_preview_beats": data_dir / PREVIEW_BEATS_NAME,
        "dense_image_prompts": data_dir / PREVIEW_PROMPTS_NAME,
        "dense_beat_review": data_dir / DENSE_BEAT_REVIEW_NAME,
        "dense_image_generation_report": data_dir / DENSE_IMAGE_REPORT_NAME,
        "dense_visual_qa": data_dir / DENSE_VISUAL_QA_NAME,
        "dense_render_report": data_dir / DENSE_RENDER_REPORT_NAME,
        "dense_playback_qa": data_dir / DENSE_PLAYBACK_QA_NAME,
        "dense_image_directory": image_dir,
        "dense_preview_video": preview_video_path,
    }

    artifacts = _load_artifacts(paths)
    git_state = _git_state(base_dir)
    protected_state = _protected_state(base_dir)
    report = _build_report(
        artifacts=artifacts,
        paths=paths,
        json_path=json_path,
        markdown_path=markdown_path,
        git_state=git_state,
        protected_state=protected_state,
        base_dir=base_dir,
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return DenseHandoffResult(report=report, json_path=json_path, markdown_path=markdown_path)


def print_dense_handoff_summary(result: DenseHandoffResult, base_dir: Path) -> None:
    report = result.report
    print("Dense production handoff dry run complete")
    print(f"Status: {report['status']}")
    print(f"Recommendation: {report['recommendation']}")
    print(f"Blocked reasons: {len(report['blocked_reasons'])}")
    print(f"Dense beats: {report['validation_counts']['dense_beat_count']}")
    print(f"Dense prompts: {report['validation_counts']['dense_prompt_count']}")
    print(f"Dense images present: {report['validation_counts']['dense_image_present_count']}")
    print("Reports written:")
    print(f"- {_display_path(result.json_path, base_dir)}")
    print(f"- {_display_path(result.markdown_path, base_dir)}")


def _load_artifacts(paths: dict[str, Path]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {"load_errors": []}
    try:
        artifacts["dense_preview_beats"] = load_beats(paths["dense_preview_beats"])
    except PipelineError as exc:
        artifacts["dense_preview_beats"] = []
        artifacts["load_errors"].append(str(exc))

    for key, label in (
        ("dense_image_prompts", "dense image prompts"),
        ("dense_beat_review", "dense beat review"),
        ("dense_image_generation_report", "dense image generation report"),
        ("dense_visual_qa", "dense visual QA report"),
        ("dense_render_report", "dense render report"),
        ("dense_playback_qa", "dense playback QA report"),
    ):
        try:
            artifacts[key] = _load_json_object(paths[key], label)
        except PipelineError as exc:
            artifacts[key] = {}
            artifacts["load_errors"].append(str(exc))
    return artifacts


def _build_report(
    *,
    artifacts: dict[str, Any],
    paths: dict[str, Path],
    json_path: Path,
    markdown_path: Path,
    git_state: dict[str, Any],
    protected_state: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    beats = artifacts["dense_preview_beats"]
    prompts = _list_field(artifacts["dense_image_prompts"], "prompts")
    image_report = artifacts["dense_image_generation_report"]
    visual_qa = artifacts["dense_visual_qa"]
    render_report = artifacts["dense_render_report"]
    playback_qa = artifacts["dense_playback_qa"]
    visual_qa_rows = _list_field(visual_qa, "images")
    render_images = _list_field(render_report, "dense_images")
    actual_dense_images = _dense_image_files(paths["dense_image_directory"])

    counts = {
        "dense_beat_count": len(beats),
        "dense_prompt_count": len(prompts),
        "dense_image_expected_count": image_report.get("expected_image_count"),
        "dense_image_present_count": image_report.get("present_image_count"),
        "dense_image_actual_file_count": len(actual_dense_images),
        "dense_image_missing_count": image_report.get("missing_image_count"),
        "dense_image_invalid_count": image_report.get("invalid_image_count"),
        "dense_image_warning_count": image_report.get("warning_count"),
        "visual_qa_reviewed_count": visual_qa.get("total_images_reviewed"),
        "visual_qa_pass_count": visual_qa.get("pass_count"),
        "visual_qa_caution_count": visual_qa.get("caution_count"),
        "visual_qa_fail_count": visual_qa.get("fail_count"),
        "playback_qa_fail_count": _playback_fail_count(playback_qa),
        "frame_level_pass_count": _nested_get(playback_qa, "frame_level_summary", "pass_count"),
        "frame_level_caution_count": _nested_get(playback_qa, "frame_level_summary", "caution_count"),
        "frame_level_fail_count": _nested_get(playback_qa, "frame_level_summary", "fail_count"),
        "render_dense_image_mapping_count": len(render_images),
    }
    caution_beats = _caution_beats(visual_qa_rows, playback_qa)
    gates = _gate_results(
        beats=beats,
        prompts=prompts,
        image_report=image_report,
        visual_qa=visual_qa,
        render_report=render_report,
        playback_qa=playback_qa,
        actual_dense_images=actual_dense_images,
        protected_state=protected_state,
        paths=paths,
        base_dir=base_dir,
    )
    blocked = [
        gate["message"]
        for gate in gates
        if gate["status"] == "fail"
    ]
    blocked.extend(artifacts["load_errors"])
    status = "ready" if not blocked else "blocked"

    return {
        "report_schema_version": 1,
        "created_at": _utc_now(),
        "status": status,
        "recommendation": "dense_ready_for_production_handoff" if status == "ready" else "dense_blocked_for_production_handoff",
        "dry_run": True,
        "no_files_copied_or_overwritten": True,
        "note": "Dry run only: no files were copied, moved, renamed, or overwritten.",
        "source_paths": {key: _display_path(path, base_dir) for key, path in paths.items()},
        "output_paths": {
            "json": _display_path(json_path, base_dir),
            "markdown": _display_path(markdown_path, base_dir),
        },
        "proposed_future_production_files": [
            {
                "source": "data/beats_dense_preview.json",
                "target": "data/beats.json",
            },
            {
                "source": "data/image_prompts_dense_preview.json",
                "target": "data/image_prompts.json",
            },
            {
                "source": "assets/generated_images_dense_preview/dense_beat_###.png",
                "target": "assets/generated_images/beat_###.png",
            },
            {
                "source": "output/final_dense_preview.mp4",
                "target": "production candidate video",
            },
        ],
        "next_command_recommendation": "python -m youtube_pipeline --apply-dense-handoff",
        "next_command_note": "Recommendation only; overwrite/apply behavior is intentionally not implemented in this dry-run command.",
        "validation_counts": counts,
        "caution_beats": caution_beats,
        "safety_gates": gates,
        "blocked_reasons": blocked,
        "git_state": git_state,
        "protected_standard_files": protected_state,
    }


def _gate_results(
    *,
    beats: list[Any],
    prompts: list[Any],
    image_report: dict[str, Any],
    visual_qa: dict[str, Any],
    render_report: dict[str, Any],
    playback_qa: dict[str, Any],
    actual_dense_images: list[Path],
    protected_state: dict[str, Any],
    paths: dict[str, Path],
    base_dir: Path,
) -> list[dict[str, Any]]:
    image_numbers = _actual_dense_image_numbers(actual_dense_images)
    render_video_path = _nested_get(render_report, "output_paths", "video")
    render_output_verified = _nested_get(render_report, "safety_gates", "output_path_verified")
    playback_video_path = playback_qa.get("video_path")
    gates = [
        _gate("dense_beat_count_is_70", len(beats) == DENSE_BEAT_COUNT, f"Dense beat count must be {DENSE_BEAT_COUNT}; found {len(beats)}."),
        _gate("dense_prompt_count_is_70", len(prompts) == DENSE_BEAT_COUNT, f"Dense prompt count must be {DENSE_BEAT_COUNT}; found {len(prompts)}."),
        _gate("dense_image_report_ready", image_report.get("ready_for_dense_render") is True, "Dense image report is not ready."),
        _gate("dense_visual_qa_ready", visual_qa.get("ready_for_dense_render_planning") is True, "Dense visual QA is not ready."),
        _gate(
            "dense_render_report_verifies_output_path",
            render_video_path == "output/final_dense_preview.mp4" and render_output_verified is True,
            "Dense render report does not verify output/final_dense_preview.mp4.",
        ),
        _gate("dense_playback_qa_passes", playback_qa.get("qa_result") == "pass", "Dense playback QA is not pass."),
        _gate(
            "dense_playback_qa_has_no_failures",
            _playback_fail_count(playback_qa) == 0,
            f"Dense playback QA has {_playback_fail_count(playback_qa)} failures.",
        ),
        _gate(
            "dense_preview_video_exists",
            paths["dense_preview_video"].exists() and paths["dense_preview_video"].is_file(),
            "Missing dense preview video: output/final_dense_preview.mp4.",
        ),
        _gate(
            "dense_preview_video_path_matches_playback_qa",
            playback_video_path == "output/final_dense_preview.mp4",
            "Dense playback QA video path is not output/final_dense_preview.mp4.",
        ),
        _gate(
            "dense_image_count_matches_dense_beat_count",
            image_report.get("present_image_count") == len(beats) == len(image_numbers),
            "Dense image count does not match dense beat count.",
        ),
        _gate("dense_image_missing_count_zero", image_report.get("missing_image_count") == 0, "Dense image report has missing images."),
        _gate("dense_image_invalid_count_zero", image_report.get("invalid_image_count") == 0, "Dense image report has invalid images."),
        _gate("dense_visual_qa_fail_count_zero", visual_qa.get("fail_count") == 0, "Dense visual QA has failures."),
        _gate(
            "all_dense_images_are_present_and_sequential",
            image_numbers == list(range(1, DENSE_BEAT_COUNT + 1)),
            "Dense preview image directory does not contain exactly dense_beat_001.png through dense_beat_070.png.",
        ),
        _gate(
            "protected_standard_files_untouched",
            protected_state.get("clean") is not False,
            "Protected standard files or outputs have local diffs.",
        ),
        _gate(
            "render_report_output_matches_existing_video",
            render_video_path == _display_path(paths["dense_preview_video"], base_dir),
            "Dense render report video path does not match the dense preview video path.",
        ),
    ]
    return gates


def _gate(name: str, ok: bool, message: str) -> dict[str, Any]:
    return {"name": name, "status": "pass" if ok else "fail", "message": "pass" if ok else message}


def _playback_fail_count(playback_qa: dict[str, Any]) -> int | None:
    fail_count = _nested_get(playback_qa, "frame_level_summary", "fail_count")
    if isinstance(fail_count, int):
        issue_count = len(playback_qa.get("issues", [])) if isinstance(playback_qa.get("issues"), list) else 0
        criteria = playback_qa.get("criteria_results")
        criteria_fail_count = 0
        if isinstance(criteria, dict):
            criteria_fail_count = sum(1 for value in criteria.values() if value == "fail")
        return fail_count + issue_count + criteria_fail_count
    return None


def _caution_beats(visual_qa_rows: list[Any], playback_qa: dict[str, Any]) -> dict[str, list[int]]:
    visual = sorted(
        row["dense_beat_number"]
        for row in visual_qa_rows
        if isinstance(row, dict) and row.get("qa_status") == "caution" and isinstance(row.get("dense_beat_number"), int)
    )
    playback_cautions = _nested_get(playback_qa, "source_report_checks", "visual_qa_caution_beats")
    return {
        "visual_qa": visual,
        "playback_source_report": playback_cautions if isinstance(playback_cautions, list) else [],
    }


def _dense_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(path for path in image_dir.glob("dense_beat_*.png") if path.is_file())


def _actual_dense_image_numbers(paths: list[Path]) -> list[int]:
    numbers = []
    for path in paths:
        stem = path.stem
        try:
            numbers.append(int(stem.removeprefix("dense_beat_")))
        except ValueError:
            continue
    return sorted(numbers)


def _list_field(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _protected_state(base_dir: Path) -> dict[str, Any]:
    return _git_diff_state(base_dir, PROTECTED_PATHS)


def _git_state(base_dir: Path) -> dict[str, Any]:
    command = ["git", "status", "--short"]
    result = _run_git(command, base_dir)
    if result["available"] is not True:
        return {"available": False, "clean": None, "warning": result["error"]}
    lines = [line for line in result["stdout"].splitlines() if line.strip()]
    return {
        "available": True,
        "clean": not lines,
        "status_short": lines,
        "warning": None if not lines else "Working tree was not clean before dense handoff report generation.",
    }


def _git_diff_state(base_dir: Path, paths: tuple[str, ...]) -> dict[str, Any]:
    command = ["git", "diff", "--", *paths]
    result = _run_git(command, base_dir)
    if result["available"] is not True:
        return {"available": False, "clean": None, "warning": result["error"], "checked_paths": list(paths)}
    diff = result["stdout"]
    return {
        "available": True,
        "clean": diff == "",
        "checked_paths": list(paths),
        "diff_line_count": len(diff.splitlines()),
        "warning": None if diff == "" else "Protected standard files or outputs had local diffs.",
    }


def _run_git(command: list[str], cwd: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    except (OSError, ValueError) as exc:
        return {"available": False, "error": f"Could not run {' '.join(command)}: {exc}"}
    if completed.returncode != 0:
        return {"available": False, "error": completed.stderr.strip() or completed.stdout.strip() or f"{' '.join(command)} failed."}
    return {"available": True, "stdout": completed.stdout}


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


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dense Production Handoff Dry Run",
        "",
        "## Summary",
        "",
        f"- Status: `{report['status']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Dry run: `{report['dry_run']}`",
        f"- No files copied or overwritten: `{report['no_files_copied_or_overwritten']}`",
        f"- Next command recommendation: `{report['next_command_recommendation']}`",
        f"- Note: {report['next_command_note']}",
        "",
        "## Proposed future production files",
        "",
    ]
    for row in report["proposed_future_production_files"]:
        lines.append(f"- `{row['source']}` -> `{row['target']}`")

    lines.extend(["", "## Validation counts", ""])
    for key, value in report["validation_counts"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Caution beats", ""])
    visual = report["caution_beats"]["visual_qa"]
    playback = report["caution_beats"]["playback_source_report"]
    lines.append(f"- Visual QA: `{', '.join(str(item) for item in visual) if visual else 'none'}`")
    lines.append(f"- Playback source report: `{', '.join(str(item) for item in playback) if playback else 'none'}`")

    lines.extend(["", "## Safety gates", ""])
    for gate in report["safety_gates"]:
        lines.append(f"- {gate['name']}: `{gate['status']}` - {gate['message']}")

    lines.extend(["", "## Blocked reasons", ""])
    if report["blocked_reasons"]:
        lines.extend(f"- {reason}" for reason in report["blocked_reasons"])
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Git and protected files",
            "",
            f"- Git status clean: `{report['git_state']['clean']}`",
            f"- Protected standard files clean: `{report['protected_standard_files']['clean']}`",
            "",
            "Dry-run note: no production files, standard image files, standard videos, or dense preview assets were copied or overwritten.",
            "",
        ]
    )
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
