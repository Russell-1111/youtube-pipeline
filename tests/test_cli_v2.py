import hashlib
import json
from pathlib import Path

from PIL import Image

from youtube_pipeline.__main__ import main
from youtube_pipeline.manifest import write_beats, write_transcript_segments
from youtube_pipeline.models import Beat, TranscriptSegment
from youtube_pipeline.render_benchmark import RenderBenchmarkReport
from youtube_pipeline.time_utils import seconds_to_timestamp


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


def beat(tmp_path: Path) -> Beat:
    return Beat(
        beat_number=1,
        beat_type="normal",
        start="00:00:00,000",
        end="00:00:06,000",
        start_seconds=0.0,
        end_seconds=6.0,
        duration_seconds=6.0,
        text_preview="Preview",
        segment_indexes=[1],
        image_path=(tmp_path / "assets" / "images" / "beat_001.png").as_posix(),
    )


def segment(index: int, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        index=index,
        start=seconds_to_timestamp(start),
        end=seconds_to_timestamp(end),
        start_seconds=start,
        end_seconds=end,
        duration_seconds=end - start,
        text=text,
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_validate_generated_images_fails_before_images_exist(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_beats(tmp_path / "data" / "beats.json", [beat(tmp_path)])

    result = main(["--config", str(config_path), "--validate-generated-images"])

    captured = capsys.readouterr()
    assert result == 1
    assert "Missing generated image for beat 001" in captured.err


def test_cli_use_generated_images_selects_generated_paths_and_output(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_beats(tmp_path / "data" / "beats.json", [beat(tmp_path)])
    voiceover = tmp_path / "input" / "voiceover.mp3"
    voiceover.parent.mkdir(parents=True)
    voiceover.write_bytes(b"placeholder")
    image_path = tmp_path / "assets" / "generated_images" / "beat_001.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (1920, 1080), "white").save(image_path)

    calls = []

    def fake_render_video(beats, audio_path, output_path, fps):
        calls.append((beats, audio_path, output_path, fps))

    monkeypatch.setattr("youtube_pipeline.__main__.render_video", fake_render_video)

    result = main(["--config", str(config_path), "--use-generated-images"])

    assert result == 0
    assert len(calls) == 1
    beats, audio_path, output_path, fps = calls[0]
    assert beats[0].image_path == image_path.as_posix()
    assert audio_path == voiceover
    assert output_path == tmp_path / "output" / "final_video_generated.mp4"
    assert fps == 30


def test_cli_use_generated_images_kinetic_selects_generated_paths_and_output(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_beats(tmp_path / "data" / "beats.json", [beat(tmp_path)])
    voiceover = tmp_path / "input" / "voiceover.mp3"
    voiceover.parent.mkdir(parents=True)
    voiceover.write_bytes(b"placeholder")
    image_path = tmp_path / "assets" / "generated_images" / "beat_001.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (1920, 1080), "white").save(image_path)

    calls = []

    def fake_render_video_kinetic(beats, audio_path, output_path, fps, width, height):
        calls.append((beats, audio_path, output_path, fps, width, height))

    monkeypatch.setattr("youtube_pipeline.__main__.render_video_kinetic", fake_render_video_kinetic)

    result = main(["--config", str(config_path), "--use-generated-images-kinetic"])

    assert result == 0
    assert len(calls) == 1
    beats, audio_path, output_path, fps, width, height = calls[0]
    assert beats[0].image_path == image_path.as_posix()
    assert audio_path == voiceover
    assert output_path == tmp_path / "output" / "final_video_generated_kinetic.mp4"
    assert fps == 30
    assert width == 1920
    assert height == 1080


def test_cli_use_generated_images_kinetic_fails_validation_before_render(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_beats(tmp_path / "data" / "beats.json", [beat(tmp_path)])

    def fake_render_video_kinetic(*args):
        raise AssertionError("kinetic render should not run when validation fails")

    monkeypatch.setattr("youtube_pipeline.__main__.render_video_kinetic", fake_render_video_kinetic)

    result = main(["--config", str(config_path), "--use-generated-images-kinetic"])

    captured = capsys.readouterr()
    assert result == 1
    assert "Missing generated image for beat 001" in captured.err


def test_cli_use_generated_images_kinetic_ffmpeg_selects_generated_paths_and_output(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_beats(tmp_path / "data" / "beats.json", [beat(tmp_path)])
    voiceover = tmp_path / "input" / "voiceover.mp3"
    voiceover.parent.mkdir(parents=True)
    voiceover.write_bytes(b"placeholder")
    image_path = tmp_path / "assets" / "generated_images" / "beat_001.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (1920, 1080), "white").save(image_path)

    calls = []

    def fake_render_video_kinetic_ffmpeg(beats, audio_path, output_path, fps, width, height, **kwargs):
        calls.append((beats, audio_path, output_path, fps, width, height, kwargs))

        class Result:
            motion_equivalence_note = "experimental"

        return Result()

    monkeypatch.setattr("youtube_pipeline.__main__.get_audio_duration", lambda audio_path: 6.0)
    monkeypatch.setattr("youtube_pipeline.__main__.render_video_kinetic_ffmpeg", fake_render_video_kinetic_ffmpeg)

    result = main(["--config", str(config_path), "--use-generated-images-kinetic-ffmpeg"])

    assert result == 0
    assert len(calls) == 1
    beats, audio_path, output_path, fps, width, height, kwargs = calls[0]
    assert beats[0].image_path == image_path.as_posix()
    assert audio_path == voiceover
    assert output_path == tmp_path / "output" / "final_video_generated_kinetic_ffmpeg.mp4"
    assert fps == 30
    assert width == 1920
    assert height == 1080
    assert kwargs["audio_duration_seconds"] == 6.0
    assert kwargs["work_dir"] == tmp_path / "output" / "benchmarks" / "final_video_generated_kinetic_ffmpeg_segments"


def test_cli_benchmark_kinetic_render_writes_under_benchmarks(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_beats(tmp_path / "data" / "beats.json", [beat(tmp_path)])
    voiceover = tmp_path / "input" / "voiceover.mp3"
    voiceover.parent.mkdir(parents=True)
    voiceover.write_bytes(b"placeholder")
    image_path = tmp_path / "assets" / "generated_images" / "beat_001.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (1920, 1080), "white").save(image_path)

    calls = []

    def fake_run_kinetic_benchmark(**kwargs):
        calls.append(kwargs)
        return RenderBenchmarkReport(
            renderer_name=kwargs["renderer_name"],
            elapsed_seconds=1.2,
            beat_count=1,
            video_duration_seconds=6.0,
            audio_duration_seconds=6.0,
            output_path=(kwargs["output_dir"] / "ffmpeg_sample_1beats.mp4").as_posix(),
            output_size_bytes=123,
            width=1920,
            height=1080,
            fps=30.0,
            audio_present=True,
            duration_delta_seconds=0.0,
            success=True,
            warnings=[],
            errors=[],
        )

    monkeypatch.setattr("youtube_pipeline.__main__.run_kinetic_benchmark", fake_run_kinetic_benchmark)

    result = main(
        [
            "--config",
            str(config_path),
            "--benchmark-kinetic-render",
            "--benchmark-renderer",
            "ffmpeg",
            "--sample-beats",
            "1",
        ]
    )

    assert result == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["renderer_name"] == "ffmpeg"
    assert call["output_dir"] == tmp_path / "output" / "benchmarks"
    assert call["sample_beats"] == 1
    assert call["full"] is False


def test_cli_explicit_modes_are_mutually_exclusive(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    try:
        main(["--config", str(config_path), "--dry-run", "--generate-prompts"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive modes")


def test_cli_plan_dense_beats_is_mutually_exclusive(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    try:
        main(["--config", str(config_path), "--plan-dense-beats", "--generate-prompts"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive dense planning mode")


def test_cli_review_dense_beats_is_mutually_exclusive(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    try:
        main(["--config", str(config_path), "--review-dense-beats", "--generate-prompts"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive dense review mode")


def test_cli_generate_dense_prompts_is_mutually_exclusive(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    try:
        main(["--config", str(config_path), "--generate-dense-prompts", "--generate-prompts"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive dense prompt mode")


def test_cli_prepare_dense_images_is_mutually_exclusive(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    try:
        main(["--config", str(config_path), "--prepare-dense-images", "--generate-prompts"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive dense image preparation mode")


def test_cli_prepare_dense_handoff_is_mutually_exclusive(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    try:
        main(["--config", str(config_path), "--prepare-dense-handoff", "--generate-prompts"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive dense handoff mode")


def test_cli_generate_prompts_behavior_remains_standard_only(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    data_dir = tmp_path / "data"
    write_beats(data_dir / "beats.json", [beat(tmp_path)])
    write_transcript_segments(data_dir / "transcript_segments.json", [segment(1, 0, 6, "Standard source text.")])

    result = main(["--config", str(config_path), "--generate-prompts"])

    assert result == 0
    assert (data_dir / "image_prompts.json").exists()
    assert not (data_dir / "image_prompts_dense_preview.json").exists()
    assert not (data_dir / "image_prompts_dense_preview.md").exists()


def test_cli_help_shows_dense_planning_as_preview_only(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("Expected argparse help to exit")

    captured = capsys.readouterr()
    assert "--plan-dense-beats" in captured.out
    assert "Preview optional dense beat planning reports" in captured.out
    assert "overwriting data/beats.json" in captured.out


def test_cli_help_shows_dense_review_as_review_only(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("Expected argparse help to exit")

    captured = capsys.readouterr()
    assert "--review-dense-beats" in captured.out
    assert "Review dense preview beats" in captured.out
    assert "production files" in captured.out


def test_cli_plan_dense_beats_writes_preview_without_overwriting_protected_inputs(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    data_dir = tmp_path / "data"
    beats_path = data_dir / "beats.json"
    prompts_path = data_dir / "image_prompts.json"
    transcript_path = data_dir / "transcript_segments.json"
    write_beats(
        beats_path,
        [
            Beat(
                beat_number=1,
                beat_type="normal",
                start="00:00:00,000",
                end="00:00:12,000",
                start_seconds=0.0,
                end_seconds=12.0,
                duration_seconds=12.0,
                text_preview="First complete idea. Second complete idea.",
                segment_indexes=[1, 2],
                image_path=(tmp_path / "assets" / "images" / "beat_001.png").as_posix(),
            )
        ],
    )
    write_transcript_segments(
        transcript_path,
        [
            segment(1, 0, 6, "First complete idea."),
            segment(2, 6, 12, "Second complete idea."),
        ],
    )
    prompts_path.write_text(json.dumps({"schema_version": 2, "prompts": []}, indent=2), encoding="utf-8")
    before = {
        "beats": file_hash(beats_path),
        "prompts": file_hash(prompts_path),
        "transcript": file_hash(transcript_path),
    }

    result = main(["--config", str(config_path), "--plan-dense-beats"])

    after = {
        "beats": file_hash(beats_path),
        "prompts": file_hash(prompts_path),
        "transcript": file_hash(transcript_path),
    }
    assert result == 0
    assert before == after
    assert (data_dir / "dense_beat_plan.json").exists()
    assert (data_dir / "dense_beat_plan.md").exists()
    assert (data_dir / "beats_dense_preview.json").exists()
    report = json.loads((data_dir / "dense_beat_plan.json").read_text(encoding="utf-8"))
    markdown = (data_dir / "dense_beat_plan.md").read_text(encoding="utf-8")
    assert report["planner_strategy"] == "hybrid_dense_rebuild"
    assert report["applied_splits"] == []
    assert report["dense_group_records"]
    assert "`hybrid_dense_rebuild`" in markdown


def test_cli_review_dense_beats_writes_report_without_overwriting_protected_inputs(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    data_dir = tmp_path / "data"
    beats_path = data_dir / "beats.json"
    prompts_path = data_dir / "image_prompts.json"
    transcript_path = data_dir / "transcript_segments.json"
    preview_path = data_dir / "beats_dense_preview.json"
    dense_plan_path = data_dir / "dense_beat_plan.json"
    source_beat = beat(tmp_path)
    write_beats(beats_path, [source_beat])
    write_beats(preview_path, [source_beat])
    write_transcript_segments(transcript_path, [segment(1, 0, 6, "123 456")])
    prompts_path.write_text(json.dumps({"schema_version": 2, "prompts": []}, indent=2), encoding="utf-8")
    dense_plan_path.write_text(
        json.dumps(
            {
                "planner_version": 1,
                "planner_strategy": "hybrid_dense_rebuild",
                "warnings": [],
                "dense_group_records": [
                    {
                        "dense_beat_number": 1,
                        "beat_type": "normal",
                        "source_standard_beat_number": 1,
                        "start_seconds": 0.0,
                        "end_seconds": 6.0,
                        "duration_seconds": 6.0,
                        "segment_indexes": [1],
                        "grouping_score": 0.0,
                        "boundary_confidence": "high",
                        "reason": "dense_rebuild_preferred_boundary",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    before = {
        "beats": file_hash(beats_path),
        "prompts": file_hash(prompts_path),
        "transcript": file_hash(transcript_path),
    }

    result = main(["--config", str(config_path), "--review-dense-beats"])

    after = {
        "beats": file_hash(beats_path),
        "prompts": file_hash(prompts_path),
        "transcript": file_hash(transcript_path),
    }
    assert result == 0
    assert before == after
    assert (data_dir / "dense_beat_review.json").exists()
    assert (data_dir / "dense_beat_review.md").exists()
    report = json.loads((data_dir / "dense_beat_review.json").read_text(encoding="utf-8"))
    markdown = (data_dir / "dense_beat_review.md").read_text(encoding="utf-8")
    assert report["readiness"] == "not_ready"
    assert report["recommendation_counts"]["blocked"] == 1
    assert "Dense Beat Review" in markdown
    assert "NO_MEANINGFUL_SOURCE_TEXT" in markdown


def test_cli_review_dense_beats_missing_preview_fails_without_writing_report(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    result = main(["--config", str(config_path), "--review-dense-beats"])

    captured = capsys.readouterr()
    assert result == 1
    assert "Missing beats file" in captured.err
    assert not (tmp_path / "data" / "dense_beat_review.json").exists()
    assert not (tmp_path / "data" / "dense_beat_review.md").exists()


def test_cli_review_dense_beats_malformed_preview_fails_without_writing_report(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "beats_dense_preview.json").write_text("{not valid json", encoding="utf-8")

    result = main(["--config", str(config_path), "--review-dense-beats"])

    captured = capsys.readouterr()
    assert result == 1
    assert "Invalid beats JSON" in captured.err
    assert not (data_dir / "dense_beat_review.json").exists()
    assert not (data_dir / "dense_beat_review.md").exists()


def test_cli_dry_run_does_not_create_dense_preview_artifacts(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "voiceover.mp3").write_bytes(b"placeholder")
    (input_dir / "transcript.srt").write_text(
        """
1
00:00:00,000 --> 00:00:06,000
Complete sentence.
""".lstrip(),
        encoding="utf-8",
    )

    result = main(["--config", str(config_path), "--dry-run"])

    assert result == 0
    assert not (tmp_path / "data" / "dense_beat_plan.json").exists()
    assert not (tmp_path / "data" / "dense_beat_plan.md").exists()
    assert not (tmp_path / "data" / "beats_dense_preview.json").exists()


def test_cli_generated_image_modes_are_mutually_exclusive(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    try:
        main(["--config", str(config_path), "--use-generated-images", "--use-generated-images-kinetic"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive generated image modes")


def test_cli_ffmpeg_generated_image_mode_is_mutually_exclusive(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    try:
        main(
            [
                "--config",
                str(config_path),
                "--use-generated-images-kinetic",
                "--use-generated-images-kinetic-ffmpeg",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive kinetic modes")
