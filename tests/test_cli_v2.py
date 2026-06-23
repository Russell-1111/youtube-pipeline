from pathlib import Path

from PIL import Image

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


def test_cli_explicit_modes_are_mutually_exclusive(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    try:
        main(["--config", str(config_path), "--dry-run", "--generate-prompts"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive modes")


def test_cli_generated_image_modes_are_mutually_exclusive(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    try:
        main(["--config", str(config_path), "--use-generated-images", "--use-generated-images-kinetic"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive generated image modes")
