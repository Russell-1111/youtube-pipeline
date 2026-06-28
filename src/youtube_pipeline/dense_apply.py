from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .beat_io import load_beats
from .config import PipelineConfig
from .dense_handoff import (
    PREVIEW_BEATS_NAME,
    PREVIEW_PROMPTS_NAME,
    _display_path,
    _git_state,
    run_dense_handoff_preparation,
)
from .dense_images import DENSE_IMAGE_DIR_NAME
from .dense_render import DENSE_BEAT_COUNT
from .errors import PipelineError

JSON_REPORT_NAME = "dense_apply_report.json"
MARKDOWN_REPORT_NAME = "dense_apply_report.md"


@dataclass(frozen=True)
class DenseApplyResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path

    @property
    def ok(self) -> bool:
        return self.report["status"] == "ok"


def run_dense_apply(
    config: PipelineConfig,
    base_dir: Path,
    *,
    dry_run: bool,
    confirm_production_overwrite: bool,
) -> DenseApplyResult:
    data_dir = config.outputs.data_dir
    json_path = data_dir / JSON_REPORT_NAME
    markdown_path = data_dir / MARKDOWN_REPORT_NAME
    data_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run and not confirm_production_overwrite:
        report = _refusal_report(
            mode="refused",
            reason=(
                "Refusing dense apply: pass either --dry-run to preview the copy plan "
                "or --confirm-production-overwrite to run the production overwrite."
            ),
            base_dir=base_dir,
            json_path=json_path,
            markdown_path=markdown_path,
        )
        _write_reports(report, json_path, markdown_path)
        return DenseApplyResult(report=report, json_path=json_path, markdown_path=markdown_path)

    handoff = run_dense_handoff_preparation(config, base_dir)
    mode = "dry_run" if dry_run else "applied"
    copy_plan = _copy_plan(config, base_dir)
    gate_failures = _apply_gate_failures(handoff.report, base_dir, require_clean_git=confirm_production_overwrite)

    if gate_failures:
        report = _build_report(
            mode=mode,
            status="blocked",
            base_dir=base_dir,
            json_path=json_path,
            markdown_path=markdown_path,
            handoff_report=handoff.report,
            copy_plan=copy_plan,
            blocked_reasons=gate_failures,
            backup_path=None,
            copied_files=[],
            before_hashes={},
            after_hashes={},
            verification=None,
        )
        _write_reports(report, json_path, markdown_path)
        return DenseApplyResult(report=report, json_path=json_path, markdown_path=markdown_path)

    if dry_run:
        report = _build_report(
            mode="dry_run",
            status="ok",
            base_dir=base_dir,
            json_path=json_path,
            markdown_path=markdown_path,
            handoff_report=handoff.report,
            copy_plan=copy_plan,
            blocked_reasons=[],
            backup_path=None,
            copied_files=[],
            before_hashes=_protected_hashes(config),
            after_hashes=_protected_hashes(config),
            verification={"result": "dry_run_no_files_modified"},
        )
        _write_reports(report, json_path, markdown_path)
        return DenseApplyResult(report=report, json_path=json_path, markdown_path=markdown_path)

    backup_path = base_dir / "backups" / f"dense_apply_{_timestamp()}"
    before_hashes = _protected_hashes(config)
    _create_backup(config, base_dir, backup_path, copy_plan)
    copied_files = _copy_dense_outputs(copy_plan, base_dir)
    after_hashes = _protected_hashes(config)
    verification = _verify_production_outputs(config)
    report = _build_report(
        mode="applied",
        status="ok" if verification["result"] == "pass" else "blocked",
        base_dir=base_dir,
        json_path=json_path,
        markdown_path=markdown_path,
        handoff_report=handoff.report,
        copy_plan=copy_plan,
        blocked_reasons=[] if verification["result"] == "pass" else verification["errors"],
        backup_path=backup_path,
        copied_files=copied_files,
        before_hashes=before_hashes,
        after_hashes=after_hashes,
        verification=verification,
    )
    _write_reports(report, json_path, markdown_path)
    _write_backup_manifest(report, backup_path, base_dir)
    return DenseApplyResult(report=report, json_path=json_path, markdown_path=markdown_path)


def print_dense_apply_summary(result: DenseApplyResult, base_dir: Path) -> None:
    report = result.report
    print("Dense apply command complete")
    print(f"Mode: {report['mode']}")
    print(f"Status: {report['status']}")
    print(f"Blocked reasons: {len(report['blocked_reasons'])}")
    print(f"Would copy files: {report['would_copy_files_count']}")
    print(f"Copied files: {report['copied_files_count']}")
    if report.get("backup_path"):
        print(f"Backup path: {report['backup_path']}")
    print("Reports written:")
    print(f"- {_display_path(result.json_path, base_dir)}")
    print(f"- {_display_path(result.markdown_path, base_dir)}")


def _refusal_report(*, mode: str, reason: str, base_dir: Path, json_path: Path, markdown_path: Path) -> dict[str, Any]:
    return {
        "report_schema_version": 1,
        "created_at": _utc_now(),
        "mode": mode,
        "status": "blocked",
        "blocked_reasons": [reason],
        "required_flags": ["--dry-run", "--confirm-production-overwrite"],
        "output_paths": {"json": _display_path(json_path, base_dir), "markdown": _display_path(markdown_path, base_dir)},
        "would_copy": [],
        "would_copy_files_count": 0,
        "copied_files": [],
        "copied_files_count": 0,
        "backup_path": None,
        "verification": None,
    }


def _build_report(
    *,
    mode: str,
    status: str,
    base_dir: Path,
    json_path: Path,
    markdown_path: Path,
    handoff_report: dict[str, Any],
    copy_plan: list[dict[str, str]],
    blocked_reasons: list[str],
    backup_path: Path | None,
    copied_files: list[dict[str, str]],
    before_hashes: dict[str, str | None],
    after_hashes: dict[str, str | None],
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    counts = handoff_report.get("validation_counts", {})
    return {
        "report_schema_version": 1,
        "created_at": _utc_now(),
        "mode": mode,
        "status": status,
        "handoff_status": handoff_report.get("status"),
        "handoff_recommendation": handoff_report.get("recommendation"),
        "blocked_reasons": blocked_reasons,
        "output_paths": {"json": _display_path(json_path, base_dir), "markdown": _display_path(markdown_path, base_dir)},
        "would_copy": copy_plan,
        "would_copy_files_count": len(copy_plan),
        "copied_files": copied_files,
        "copied_files_count": len(copied_files),
        "backup_path": _display_path(backup_path, base_dir) if backup_path else None,
        "production_counts": {
            "beats": counts.get("dense_beat_count") if mode == "dry_run" else _safe_len(base_dir / "data" / "beats.json"),
            "prompts": counts.get("dense_prompt_count") if mode == "dry_run" else _safe_prompt_count(base_dir / "data" / "image_prompts.json"),
            "images": counts.get("dense_image_actual_file_count") if mode == "dry_run" else len(list((base_dir / "assets" / "generated_images").glob("beat_*.png"))),
        },
        "before_hashes": before_hashes,
        "after_hashes": after_hashes,
        "verification": verification,
        "restore_instructions": _restore_instructions(backup_path, base_dir) if backup_path else None,
    }


def _apply_gate_failures(handoff_report: dict[str, Any], base_dir: Path, *, require_clean_git: bool) -> list[str]:
    failures = []
    counts = handoff_report.get("validation_counts", {})
    if handoff_report.get("status") != "ready":
        failures.append("--prepare-dense-handoff status must be ready.")
    if handoff_report.get("recommendation") != "dense_ready_for_production_handoff":
        failures.append("--prepare-dense-handoff recommendation must be dense_ready_for_production_handoff.")
    if require_clean_git:
        git_state = _git_state(base_dir)
        if git_state.get("clean") is not True:
            failures.append("Git status must be clean before applying dense handoff.")
    expected_counts = {
        "dense_beat_count": DENSE_BEAT_COUNT,
        "dense_prompt_count": DENSE_BEAT_COUNT,
        "dense_image_actual_file_count": DENSE_BEAT_COUNT,
        "dense_image_missing_count": 0,
        "dense_image_invalid_count": 0,
        "visual_qa_fail_count": 0,
        "playback_qa_fail_count": 0,
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            failures.append(f"{key} must be {expected}; found {counts.get(key)}.")
    if not (base_dir / "output" / "final_dense_preview.mp4").is_file():
        failures.append("output/final_dense_preview.mp4 must exist.")
    return failures


def _copy_plan(config: PipelineConfig, base_dir: Path) -> list[dict[str, str]]:
    data_dir = config.outputs.data_dir
    source_image_dir = config.outputs.images_dir.parent / DENSE_IMAGE_DIR_NAME
    target_image_dir = config.outputs.images_dir.parent / "generated_images"
    rows = [
        {"source": _display_path(data_dir / PREVIEW_BEATS_NAME, base_dir), "target": "data/beats.json"},
        {"source": _display_path(data_dir / PREVIEW_PROMPTS_NAME, base_dir), "target": "data/image_prompts.json"},
    ]
    for number in range(1, DENSE_BEAT_COUNT + 1):
        rows.append(
            {
                "source": _display_path(source_image_dir / f"dense_beat_{number:03}.png", base_dir),
                "target": _display_path(target_image_dir / f"beat_{number:03}.png", base_dir),
            }
        )
    return rows


def _create_backup(config: PipelineConfig, base_dir: Path, backup_path: Path, copy_plan: list[dict[str, str]]) -> None:
    backup_path.mkdir(parents=True, exist_ok=False)
    for relative in ("data/beats.json", "data/image_prompts.json"):
        source = base_dir / relative
        if source.exists():
            target = backup_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    image_dir = config.outputs.images_dir.parent / "generated_images"
    if image_dir.exists():
        shutil.copytree(image_dir, backup_path / "assets" / "generated_images")
    (backup_path / "copy_plan.json").write_text(json.dumps(copy_plan, indent=2) + "\n", encoding="utf-8")


def _copy_dense_outputs(copy_plan: list[dict[str, str]], base_dir: Path) -> list[dict[str, str]]:
    copied = []
    for row in copy_plan:
        source = base_dir / row["source"]
        target = base_dir / row["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(row)
    return copied


def _verify_production_outputs(config: PipelineConfig) -> dict[str, Any]:
    errors = []
    beats = load_beats(config.outputs.data_dir / "beats.json")
    prompts = _load_prompt_count(config.outputs.data_dir / "image_prompts.json")
    images = sorted((config.outputs.images_dir.parent / "generated_images").glob("beat_*.png"))
    if len(beats) != DENSE_BEAT_COUNT:
        errors.append(f"Production beats count must be {DENSE_BEAT_COUNT}; found {len(beats)}.")
    if prompts != DENSE_BEAT_COUNT:
        errors.append(f"Production prompts count must be {DENSE_BEAT_COUNT}; found {prompts}.")
    if len(images) != DENSE_BEAT_COUNT:
        errors.append(f"Production image count must be {DENSE_BEAT_COUNT}; found {len(images)}.")
    return {
        "result": "pass" if not errors else "fail",
        "errors": errors,
        "production_beats_count": len(beats),
        "production_prompts_count": prompts,
        "production_image_count": len(images),
    }


def _protected_hashes(config: PipelineConfig) -> dict[str, str | None]:
    paths = [
        config.outputs.data_dir / "beats.json",
        config.outputs.data_dir / "image_prompts.json",
    ]
    hashes = {_display_path(path, config.outputs.data_dir.parent): _hash_file(path) for path in paths}
    image_dir = config.outputs.images_dir.parent / "generated_images"
    for path in sorted(image_dir.glob("beat_*.png")) if image_dir.exists() else []:
        hashes[_display_path(path, config.outputs.data_dir.parent)] = _hash_file(path)
    return hashes


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_len(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return len(payload) if isinstance(payload, list) else None


def _safe_prompt_count(path: Path) -> int | None:
    try:
        return _load_prompt_count(path)
    except PipelineError:
        return None


def _load_prompt_count(path: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Could not read production prompts from {path}: {exc}") from exc
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(prompts, list):
        raise PipelineError(f"Invalid prompts JSON: expected object with prompts list in {path}")
    return len(prompts)


def _write_reports(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")


def _write_backup_manifest(report: dict[str, Any], backup_path: Path, base_dir: Path) -> None:
    manifest_json = backup_path / "backup_manifest.json"
    manifest_md = backup_path / "backup_manifest.md"
    payload = {
        "created_at": report["created_at"],
        "backup_path": _display_path(backup_path, base_dir),
        "restore_instructions": report["restore_instructions"],
        "before_hashes": report["before_hashes"],
        "after_hashes": report["after_hashes"],
    }
    manifest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_md.write_text(_markdown_backup_manifest(payload), encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dense Apply Report",
        "",
        "## Summary",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Status: `{report['status']}`",
        f"- Handoff status: `{report.get('handoff_status')}`",
        f"- Handoff recommendation: `{report.get('handoff_recommendation')}`",
        f"- Would copy files: `{report['would_copy_files_count']}`",
        f"- Copied files: `{report['copied_files_count']}`",
        f"- Backup path: `{report.get('backup_path') or 'none'}`",
        "",
        "## Blocked reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in report["blocked_reasons"]) if report["blocked_reasons"] else lines.append("- None")
    lines.extend(["", "## Copy plan", ""])
    for row in report["would_copy"]:
        lines.append(f"- `{row['source']}` -> `{row['target']}`")
    lines.extend(["", "## Verification", "", f"- Result: `{(report.get('verification') or {}).get('result')}`", ""])
    return "\n".join(lines)


def _markdown_backup_manifest(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Dense Apply Backup Manifest",
            "",
            f"- Created at: `{payload['created_at']}`",
            f"- Backup path: `{payload['backup_path']}`",
            "",
            "## Restore instructions",
            "",
            payload["restore_instructions"],
            "",
        ]
    )


def _restore_instructions(backup_path: Path, base_dir: Path) -> str:
    backup = _display_path(backup_path, base_dir)
    return (
        f"Copy `{backup}/data/beats.json` back to `data/beats.json`, "
        f"`{backup}/data/image_prompts.json` back to `data/image_prompts.json`, "
        f"and `{backup}/assets/generated_images/` back to `assets/generated_images/`."
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
