import hashlib
import json
from pathlib import Path

from PIL import Image

from youtube_pipeline.__main__ import main
from youtube_pipeline.manifest import write_beats
from youtube_pipeline.models import Beat
from youtube_pipeline.render_ffmpeg import FFmpegRenderResult


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


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1920, 1080), "white").save(path)


def write_dense_render_inputs(
    tmp_path: Path,
    *,
    write_image_report: bool = True,
    ready_for_dense_render: bool = True,
    fail_count: int = 0,
    voiceover: bool = True,
) -> None:
    write_config(tmp_path / "config.yaml")
    data_dir = tmp_path / "data"
    image_dir = tmp_path / "assets" / "generated_images_dense_preview"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_beats(data_dir / "beats_dense_preview.json", [beat(number) for number in range(1, 71)])
    for number in range(1, 71):
        write_png(image_dir / f"dense_beat_{number:03}.png")
    if voiceover:
        voiceover_path = tmp_path / "input" / "voiceover.mp3"
        voiceover_path.parent.mkdir(parents=True, exist_ok=True)
        voiceover_path.write_bytes(b"placeholder")
    if write_image_report:
        image_report = {
            "ready_for_dense_render": ready_for_dense_render,
            "expected_image_count": 70,
            "present_image_count": 70,
            "missing_image_count": 0,
            "invalid_image_count": 0,
            "warning_count": 0,
            "images": [
                {
                    "dense_beat_number": number,
                    "expected_filename": f"dense_beat_{number:03}.png",
                    "expected_path": f"assets/generated_images_dense_preview/dense_beat_{number:03}.png",
                    "status": "valid",
                }
                for number in range(1, 71)
            ],
        }
        (data_dir / "dense_image_generation_report.json").write_text(json.dumps(image_report, indent=2), encoding="utf-8")
    qa_report = {
        "ready_for_dense_render_planning": True,
        "total_images_reviewed": 70,
        "fail_count": fail_count,
        "images": [
            {
                "dense_beat_number": number,
                "filename": f"dense_beat_{number:03}.png",
                "qa_status": "fail" if number <= fail_count else "pass",
            }
            for number in range(1, 71)
        ],
    }
    (data_dir / "dense_image_visual_qa.json").write_text(json.dumps(qa_report, indent=2), encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_render_dense_preview_is_mutually_exclusive(tmp_path):
    write_config(tmp_path / "config.yaml")

    try:
        main(["--config", str(tmp_path / "config.yaml"), "--render-dense-preview", "--generate-prompts"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive dense render mode")


def test_dense_render_refuses_missing_dense_image_report(tmp_path, monkeypatch, capsys):
    write_dense_render_inputs(tmp_path, write_image_report=False)
    monkeypatch.setattr("youtube_pipeline.dense_render.render_video_kinetic_ffmpeg", lambda *args, **kwargs: None)

    result = main(["--config", str(tmp_path / "config.yaml"), "--render-dense-preview"])

    captured = capsys.readouterr()
    assert result == 1
    assert "Missing dense image generation report file" in captured.err
    assert not (tmp_path / "output" / "final_dense_preview.mp4").exists()


def test_dense_render_refuses_not_ready_dense_image_report(tmp_path, monkeypatch, capsys):
    write_dense_render_inputs(tmp_path, ready_for_dense_render=False)
    monkeypatch.setattr("youtube_pipeline.dense_render.render_video_kinetic_ffmpeg", lambda *args, **kwargs: None)

    result = main(["--config", str(tmp_path / "config.yaml"), "--render-dense-preview"])

    captured = capsys.readouterr()
    assert result == 1
    assert "Dense image report is not ready for dense render" in captured.err
    assert not (tmp_path / "output" / "final_dense_preview.mp4").exists()


def test_dense_render_refuses_qa_fail_count(tmp_path, monkeypatch, capsys):
    write_dense_render_inputs(tmp_path, fail_count=1)
    monkeypatch.setattr("youtube_pipeline.dense_render.render_video_kinetic_ffmpeg", lambda *args, **kwargs: None)

    result = main(["--config", str(tmp_path / "config.yaml"), "--render-dense-preview"])

    captured = capsys.readouterr()
    assert result == 1
    assert "Dense visual QA has failures" in captured.err
    assert not (tmp_path / "output" / "final_dense_preview.mp4").exists()


def test_dense_render_refuses_missing_voiceover(tmp_path, monkeypatch, capsys):
    write_dense_render_inputs(tmp_path, voiceover=False)
    monkeypatch.setattr("youtube_pipeline.dense_render.render_video_kinetic_ffmpeg", lambda *args, **kwargs: None)

    result = main(["--config", str(tmp_path / "config.yaml"), "--render-dense-preview"])

    captured = capsys.readouterr()
    assert result == 1
    assert "Missing voiceover file" in captured.err
    assert not (tmp_path / "output" / "final_dense_preview.mp4").exists()


def test_dense_render_maps_dense_beats_and_calls_renderer_with_safe_output(tmp_path, monkeypatch):
    write_dense_render_inputs(tmp_path)
    calls = []

    def fake_get_audio_duration(audio_path):
        return 420.0

    def fake_renderer(beats, audio_path, output_path, fps, width, height, **kwargs):
        calls.append((beats, audio_path, output_path, fps, width, height, kwargs))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")
        return FFmpegRenderResult(
            output_path=output_path,
            segment_dir=kwargs["work_dir"],
            concat_video_path=kwargs["work_dir"] / "video_concat.mp4",
            video_duration_seconds=420.0,
            beat_count=len(beats),
            motion_equivalence_note="test renderer",
        )

    monkeypatch.setattr("youtube_pipeline.dense_render.get_audio_duration", fake_get_audio_duration)
    monkeypatch.setattr("youtube_pipeline.dense_render.render_video_kinetic_ffmpeg", fake_renderer)

    result = main(["--config", str(tmp_path / "config.yaml"), "--render-dense-preview"])

    assert result == 0
    assert len(calls) == 1
    beats, audio_path, output_path, fps, width, height, kwargs = calls[0]
    assert len(beats) == 70
    assert beats[0].image_path == (tmp_path / "assets" / "generated_images_dense_preview" / "dense_beat_001.png").as_posix()
    assert beats[-1].image_path == (tmp_path / "assets" / "generated_images_dense_preview" / "dense_beat_070.png").as_posix()
    assert audio_path == tmp_path / "input" / "voiceover.mp3"
    assert output_path == tmp_path / "output" / "final_dense_preview.mp4"
    assert fps == 30
    assert width == 1920
    assert height == 1080
    assert kwargs["audio_duration_seconds"] == 420.0
    assert kwargs["work_dir"] == tmp_path / "output" / "benchmarks" / "final_dense_preview_segments"
    report = json.loads((tmp_path / "data" / "dense_render_report.json").read_text(encoding="utf-8"))
    assert report["renderer"] == "ffmpeg"
    assert report["output_paths"]["video"] == "output/final_dense_preview.mp4"
    assert report["dense_images"][0]["image_path"] == "assets/generated_images_dense_preview/dense_beat_001.png"
    assert (tmp_path / "data" / "dense_render_report.md").exists()


def test_dense_render_does_not_touch_protected_or_standard_paths(tmp_path, monkeypatch):
    write_dense_render_inputs(tmp_path)
    protected_paths = [
        tmp_path / "data" / "beats.json",
        tmp_path / "data" / "image_prompts.json",
        tmp_path / "data" / "transcript_segments.json",
        tmp_path / "output" / "final_video.mp4",
        tmp_path / "output" / "final_video_generated.mp4",
        tmp_path / "output" / "final_video_generated_kinetic.mp4",
        tmp_path / "output" / "final_video_generated_kinetic_ffmpeg.mp4",
        tmp_path / "assets" / "generated_images" / "beat_001.png",
    ]
    for path in protected_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"protected {path.name}".encode("utf-8"))
    before = {path: file_hash(path) for path in protected_paths}

    def fake_renderer(beats, audio_path, output_path, fps, width, height, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")
        return FFmpegRenderResult(output_path, kwargs["work_dir"], kwargs["work_dir"] / "video_concat.mp4", 420.0, len(beats), "test")

    monkeypatch.setattr("youtube_pipeline.dense_render.get_audio_duration", lambda audio_path: 420.0)
    monkeypatch.setattr("youtube_pipeline.dense_render.render_video_kinetic_ffmpeg", fake_renderer)

    result = main(["--config", str(tmp_path / "config.yaml"), "--render-dense-preview"])

    after = {path: file_hash(path) for path in protected_paths}
    assert result == 0
    assert before == after
    assert (tmp_path / "output" / "final_dense_preview.mp4").exists()
