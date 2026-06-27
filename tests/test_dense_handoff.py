import hashlib
import json
from pathlib import Path

from youtube_pipeline.__main__ import main
from youtube_pipeline.manifest import write_beats
from youtube_pipeline.models import Beat


def write_config(path: Path) -> None:
    path.write_text(
        """
inputs:
  voiceover: input/voiceover.mp3
  transcript: input/transcript.srt

outputs:
  data_dir: data
  images_dir: assets/images
  contact_sheet: assets/contact_sheets/contact_sheet.png
  final_video: output/final_video.mp4

video:
  width: 1920
  height: 1080
  fps: 30

beats:
  min_duration: 6
  target_duration: 9
  max_duration: 12
  min_gap_beat_duration: 1.5
  min_intro_beat_duration: 1.0
  max_preview_chars: 80

timing:
  duration_mismatch_tolerance: 1.0
""".lstrip(),
        encoding="utf-8",
    )


def beat(number: int) -> Beat:
    start = float((number - 1) * 6)
    end = start + 6.0
    return Beat(
        beat_number=number,
        beat_type="normal",
        start=f"00:00:{int(start) % 60:02},000",
        end=f"00:00:{int(end) % 60:02},000",
        start_seconds=start,
        end_seconds=end,
        duration_seconds=6.0,
        text_preview=f"Dense beat {number}",
        segment_indexes=[number],
        image_path=f"assets/images/beat_{number:03}.png",
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dense_handoff_inputs(
    tmp_path: Path,
    *,
    render_report: bool = True,
    playback_result: str = "pass",
    playback_fail_count: int = 0,
    preview_video: bool = True,
) -> None:
    write_config(tmp_path / "config.yaml")
    data_dir = tmp_path / "data"
    image_dir = tmp_path / "assets" / "generated_images_dense_preview"
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    write_beats(data_dir / "beats_dense_preview.json", [beat(number) for number in range(1, 71)])
    (data_dir / "image_prompts_dense_preview.json").write_text(
        json.dumps(
            {
                "total_dense_beats": 70,
                "prompt_count": 70,
                "prompts": [{"dense_beat_number": number, "final_image_prompt": f"Prompt {number}"} for number in range(1, 71)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (data_dir / "dense_beat_review.json").write_text(
        json.dumps(
            {
                "total_dense_beats": 70,
                "readiness": "ready",
                "beats": [{"dense_beat_number": number, "recommendation": "approve"} for number in range(1, 71)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for number in range(1, 71):
        (image_dir / f"dense_beat_{number:03}.png").write_bytes(b"png")
    image_rows = [
        {
            "dense_beat_number": number,
            "expected_filename": f"dense_beat_{number:03}.png",
            "expected_path": f"assets/generated_images_dense_preview/dense_beat_{number:03}.png",
            "status": "valid",
        }
        for number in range(1, 71)
    ]
    (data_dir / "dense_image_generation_report.json").write_text(
        json.dumps(
            {
                "ready_for_dense_render": True,
                "expected_image_count": 70,
                "present_image_count": 70,
                "missing_image_count": 0,
                "invalid_image_count": 0,
                "warning_count": 0,
                "images": image_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (data_dir / "dense_image_visual_qa.json").write_text(
        json.dumps(
            {
                "ready_for_dense_render_planning": True,
                "total_images_reviewed": 70,
                "pass_count": 69,
                "caution_count": 1,
                "fail_count": 0,
                "images": [
                    {
                        "dense_beat_number": number,
                        "filename": f"dense_beat_{number:03}.png",
                        "qa_status": "caution" if number == 8 else "pass",
                    }
                    for number in range(1, 71)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if render_report:
        (data_dir / "dense_render_report.json").write_text(
            json.dumps(
                {
                    "dense_beat_count": 70,
                    "output_paths": {"video": "output/final_dense_preview.mp4"},
                    "video": {"duration_seconds": 592.9, "audio_duration_seconds": 592.9, "width": 1920, "height": 1080, "fps": 30},
                    "safety_gates": {"output_path_verified": True},
                    "dense_images": [{"dense_beat_number": number} for number in range(1, 71)],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    (data_dir / "dense_render_playback_qa.json").write_text(
        json.dumps(
            {
                "video_path": "output/final_dense_preview.mp4",
                "qa_result": playback_result,
                "metadata": {"duration_seconds": 592.9, "width": 1920, "height": 1080, "fps": 30, "audio_video_duration_delta_seconds": 0.0},
                "source_report_checks": {"visual_qa_caution_beats": [8]},
                "frame_level_summary": {"pass_count": 80, "caution_count": 0, "fail_count": playback_fail_count},
                "criteria_results": {"all_70_dense_images_in_order": "pass"},
                "issues": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if preview_video:
        video_path = tmp_path / "output" / "final_dense_preview.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"preview video")


def write_protected_files(tmp_path: Path) -> list[Path]:
    paths = [
        tmp_path / "data" / "beats.json",
        tmp_path / "data" / "image_prompts.json",
        tmp_path / "data" / "transcript_segments.json",
        tmp_path / "assets" / "generated_images" / "beat_001.png",
        tmp_path / "output" / "final_video.mp4",
        tmp_path / "output" / "final_video_generated.mp4",
        tmp_path / "output" / "final_video_generated_kinetic.mp4",
        tmp_path / "output" / "final_video_generated_kinetic_ffmpeg.mp4",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"protected {path.name}".encode("utf-8"))
    return paths


def test_dense_handoff_succeeds_when_all_dense_reports_are_ready(tmp_path):
    write_dense_handoff_inputs(tmp_path)

    result = main(["--config", str(tmp_path / "config.yaml"), "--prepare-dense-handoff"])

    report = json.loads((tmp_path / "data" / "dense_handoff_report.json").read_text(encoding="utf-8"))
    assert result == 0
    assert report["status"] == "ready"
    assert report["recommendation"] == "dense_ready_for_production_handoff"
    assert report["no_files_copied_or_overwritten"] is True
    assert report["validation_counts"]["dense_beat_count"] == 70
    assert report["validation_counts"]["dense_prompt_count"] == 70
    assert report["validation_counts"]["dense_image_actual_file_count"] == 70
    assert report["caution_beats"]["visual_qa"] == [8]
    assert (tmp_path / "data" / "dense_handoff_report.md").exists()


def test_dense_handoff_refuses_missing_dense_render_report(tmp_path):
    write_dense_handoff_inputs(tmp_path, render_report=False)

    result = main(["--config", str(tmp_path / "config.yaml"), "--prepare-dense-handoff"])

    report = json.loads((tmp_path / "data" / "dense_handoff_report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert report["status"] == "blocked"
    assert any("Missing dense render report file" in reason for reason in report["blocked_reasons"])


def test_dense_handoff_refuses_failed_playback_qa(tmp_path):
    write_dense_handoff_inputs(tmp_path, playback_result="fail", playback_fail_count=1)

    result = main(["--config", str(tmp_path / "config.yaml"), "--prepare-dense-handoff"])

    report = json.loads((tmp_path / "data" / "dense_handoff_report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert report["status"] == "blocked"
    assert "Dense playback QA is not pass." in report["blocked_reasons"]
    assert "Dense playback QA has 1 failures." in report["blocked_reasons"]


def test_dense_handoff_refuses_missing_dense_preview_video(tmp_path):
    write_dense_handoff_inputs(tmp_path, preview_video=False)

    result = main(["--config", str(tmp_path / "config.yaml"), "--prepare-dense-handoff"])

    report = json.loads((tmp_path / "data" / "dense_handoff_report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert report["status"] == "blocked"
    assert "Missing dense preview video: output/final_dense_preview.mp4." in report["blocked_reasons"]


def test_dense_handoff_writes_only_dense_handoff_reports(tmp_path):
    write_dense_handoff_inputs(tmp_path)
    before = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()}

    result = main(["--config", str(tmp_path / "config.yaml"), "--prepare-dense-handoff"])

    after = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()}
    assert result == 0
    assert after - before == {"data/dense_handoff_report.json", "data/dense_handoff_report.md"}


def test_dense_handoff_does_not_modify_protected_production_files(tmp_path):
    write_dense_handoff_inputs(tmp_path)
    protected_paths = write_protected_files(tmp_path)
    before = {path: file_hash(path) for path in protected_paths}

    result = main(["--config", str(tmp_path / "config.yaml"), "--prepare-dense-handoff"])

    after = {path: file_hash(path) for path in protected_paths}
    assert result == 0
    assert before == after

