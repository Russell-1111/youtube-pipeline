import json
from pathlib import Path

from youtube_pipeline.__main__ import main

from test_dense_handoff import file_hash, write_dense_handoff_inputs, write_protected_files


def test_apply_dense_handoff_refuses_without_mode_flag(tmp_path, capsys):
    write_dense_handoff_inputs(tmp_path)

    result = main(["--config", str(tmp_path / "config.yaml"), "--apply-dense-handoff"])

    captured = capsys.readouterr()
    report = json.loads((tmp_path / "data" / "dense_apply_report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert report["mode"] == "refused"
    assert "--dry-run" in report["blocked_reasons"][0]
    assert "--confirm-production-overwrite" in report["blocked_reasons"][0]
    assert "Mode: refused" in captured.out


def test_apply_dense_handoff_dry_run_succeeds_and_writes_apply_reports_only_for_production(tmp_path):
    write_dense_handoff_inputs(tmp_path)
    protected_paths = write_protected_files(tmp_path)
    before = {path: file_hash(path) for path in protected_paths}
    before_files = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()}

    result = main(["--config", str(tmp_path / "config.yaml"), "--apply-dense-handoff", "--dry-run"])

    after = {path: file_hash(path) for path in protected_paths}
    after_files = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()}
    report = json.loads((tmp_path / "data" / "dense_apply_report.json").read_text(encoding="utf-8"))
    assert result == 0
    assert report["mode"] == "dry_run"
    assert report["status"] == "ok"
    assert report["would_copy_files_count"] == 72
    assert report["copied_files_count"] == 0
    assert before == after
    assert "data/beats.json" in report["before_hashes"]
    assert "data/image_prompts.json" in report["after_hashes"]
    assert after_files - before_files == {
        "data/dense_handoff_report.json",
        "data/dense_handoff_report.md",
        "data/dense_apply_report.json",
        "data/dense_apply_report.md",
    }


def test_apply_dense_handoff_dry_run_does_not_modify_standard_image_folder(tmp_path):
    write_dense_handoff_inputs(tmp_path)
    standard_dir = tmp_path / "assets" / "generated_images"
    standard_dir.mkdir(parents=True)
    standard_image = standard_dir / "beat_001.png"
    standard_image.write_bytes(b"standard image")
    before = file_hash(standard_image)

    result = main(["--config", str(tmp_path / "config.yaml"), "--apply-dense-handoff", "--dry-run"])

    assert result == 0
    assert file_hash(standard_image) == before
    assert not (standard_dir / "beat_070.png").exists()


def test_apply_dense_handoff_real_apply_refuses_dirty_git(tmp_path, monkeypatch):
    write_dense_handoff_inputs(tmp_path)
    monkeypatch.setattr(
        "youtube_pipeline.dense_apply._git_state",
        lambda base_dir: {"available": True, "clean": False, "status_short": [" M README.md"]},
    )

    result = main(["--config", str(tmp_path / "config.yaml"), "--apply-dense-handoff", "--confirm-production-overwrite"])

    report = json.loads((tmp_path / "data" / "dense_apply_report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert "Git status must be clean before applying dense handoff." in report["blocked_reasons"]
    assert report["copied_files_count"] == 0


def test_apply_dense_handoff_real_apply_refuses_not_ready_handoff(tmp_path, monkeypatch):
    write_dense_handoff_inputs(tmp_path, render_report=False)
    monkeypatch.setattr("youtube_pipeline.dense_apply._git_state", lambda base_dir: {"available": True, "clean": True})

    result = main(["--config", str(tmp_path / "config.yaml"), "--apply-dense-handoff", "--confirm-production-overwrite"])

    report = json.loads((tmp_path / "data" / "dense_apply_report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert "--prepare-dense-handoff status must be ready." in report["blocked_reasons"]
    assert report["copied_files_count"] == 0


def test_apply_dense_handoff_real_apply_refuses_dense_image_count_mismatch(tmp_path, monkeypatch):
    write_dense_handoff_inputs(tmp_path)
    (tmp_path / "assets" / "generated_images_dense_preview" / "dense_beat_070.png").unlink()
    monkeypatch.setattr("youtube_pipeline.dense_apply._git_state", lambda base_dir: {"available": True, "clean": True})

    result = main(["--config", str(tmp_path / "config.yaml"), "--apply-dense-handoff", "--confirm-production-overwrite"])

    report = json.loads((tmp_path / "data" / "dense_apply_report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert any("dense_image_actual_file_count must be 70" in reason for reason in report["blocked_reasons"])
    assert report["copied_files_count"] == 0


def test_apply_dense_handoff_real_apply_refuses_playback_qa_failed(tmp_path, monkeypatch):
    write_dense_handoff_inputs(tmp_path, playback_result="fail", playback_fail_count=1)
    monkeypatch.setattr("youtube_pipeline.dense_apply._git_state", lambda base_dir: {"available": True, "clean": True})

    result = main(["--config", str(tmp_path / "config.yaml"), "--apply-dense-handoff", "--confirm-production-overwrite"])

    report = json.loads((tmp_path / "data" / "dense_apply_report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert any("playback_qa_fail_count must be 0" in reason for reason in report["blocked_reasons"])
    assert report["copied_files_count"] == 0


def test_apply_dense_handoff_real_apply_creates_backup_before_copying(tmp_path, monkeypatch):
    write_dense_handoff_inputs(tmp_path)
    write_protected_files(tmp_path)
    monkeypatch.setattr("youtube_pipeline.dense_apply._git_state", lambda base_dir: {"available": True, "clean": True})

    result = main(["--config", str(tmp_path / "config.yaml"), "--apply-dense-handoff", "--confirm-production-overwrite"])

    report = json.loads((tmp_path / "data" / "dense_apply_report.json").read_text(encoding="utf-8"))
    backup_path = tmp_path / report["backup_path"]
    assert result == 0
    assert report["mode"] == "applied"
    assert report["copied_files_count"] == 72
    assert (backup_path / "data" / "beats.json").exists()
    assert (backup_path / "data" / "image_prompts.json").exists()
    assert (backup_path / "assets" / "generated_images").exists()
    assert (backup_path / "backup_manifest.json").exists()
    assert (backup_path / "backup_manifest.md").exists()


def test_apply_dense_handoff_real_apply_maps_dense_images_to_standard_names(tmp_path, monkeypatch):
    write_dense_handoff_inputs(tmp_path)
    write_protected_files(tmp_path)
    source_dir = tmp_path / "assets" / "generated_images_dense_preview"
    for number in range(1, 71):
        (source_dir / f"dense_beat_{number:03}.png").write_bytes(f"dense {number}".encode("utf-8"))
    monkeypatch.setattr("youtube_pipeline.dense_apply._git_state", lambda base_dir: {"available": True, "clean": True})

    result = main(["--config", str(tmp_path / "config.yaml"), "--apply-dense-handoff", "--confirm-production-overwrite"])

    standard_dir = tmp_path / "assets" / "generated_images"
    assert result == 0
    assert (standard_dir / "beat_001.png").read_bytes() == b"dense 1"
    assert (standard_dir / "beat_070.png").read_bytes() == b"dense 70"
    assert not (standard_dir / "dense_beat_001.png").exists()


def test_apply_dense_handoff_flags_are_mutually_exclusive(tmp_path):
    write_dense_handoff_inputs(tmp_path)

    try:
        main(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--apply-dense-handoff",
                "--dry-run",
                "--confirm-production-overwrite",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive dense apply flags")


def test_apply_dense_handoff_is_mutually_exclusive_with_standard_modes(tmp_path):
    write_dense_handoff_inputs(tmp_path)

    try:
        main(["--config", str(tmp_path / "config.yaml"), "--apply-dense-handoff", "--generate-prompts"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive apply mode")
