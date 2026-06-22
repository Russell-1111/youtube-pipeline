from dataclasses import replace
from pathlib import Path

from PIL import Image

from youtube_pipeline.generated_images import (
    beats_with_generated_image_paths,
    load_and_validate_generated_images,
    validate_generated_images,
)
from youtube_pipeline.manifest import write_beats
from youtube_pipeline.models import Beat


def beat(tmp_path: Path, number: int = 1) -> Beat:
    return Beat(
        beat_number=number,
        beat_type="normal",
        start="00:00:00,000",
        end="00:00:06,000",
        start_seconds=0.0,
        end_seconds=6.0,
        duration_seconds=6.0,
        text_preview="Preview",
        segment_indexes=[number],
        image_path=(tmp_path / "assets" / "images" / f"beat_{number:03}.png").as_posix(),
    )


def write_png(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def test_validation_passes_when_all_required_images_exist(tmp_path):
    beats = [beat(tmp_path, 1), beat(tmp_path, 2)]
    image_dir = tmp_path / "assets" / "generated_images"
    write_png(image_dir / "beat_001.png", (1920, 1080))
    write_png(image_dir / "beat_002.png", (1920, 1080))

    report = validate_generated_images(beats, image_dir)

    assert report.ok
    assert report.errors == []
    assert report.warnings == []


def test_validation_fails_clearly_when_images_are_missing(tmp_path):
    report = validate_generated_images([beat(tmp_path, 1), beat(tmp_path, 2)], tmp_path / "assets" / "generated_images")

    assert not report.ok
    assert len(report.errors) == 2
    assert "Missing generated image for beat 001" in report.errors[0]
    assert "Missing generated image for beat 002" in report.errors[1]


def test_validation_fails_for_non_png_image(tmp_path):
    image_dir = tmp_path / "assets" / "generated_images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (1920, 1080), "white").save(image_dir / "beat_001.png", format="JPEG")

    report = validate_generated_images([beat(tmp_path)], image_dir)

    assert not report.ok
    assert "must be PNG" in report.errors[0]


def test_validation_fails_for_invalid_dimensions_and_aspect_ratio(tmp_path):
    image_dir = tmp_path / "assets" / "generated_images"
    write_png(image_dir / "beat_001.png", (1280, 800))
    write_png(image_dir / "beat_002.png", (640, 360))

    report = validate_generated_images([beat(tmp_path, 1), beat(tmp_path, 2)], image_dir)

    assert not report.ok
    assert any("must be exact 16:9" in error for error in report.errors)
    assert any("too small" in error for error in report.errors)


def test_extra_images_warn_but_do_not_fail(tmp_path):
    image_dir = tmp_path / "assets" / "generated_images"
    write_png(image_dir / "beat_001.png", (1280, 720))
    write_png(image_dir / "beat_999.png", (1920, 1080))

    report = validate_generated_images([beat(tmp_path)], image_dir)

    assert report.ok
    assert any("preferred size" in warning for warning in report.warnings)
    assert any("Extra generated image file ignored" in warning for warning in report.warnings)


def test_generated_image_render_paths_replace_placeholder_paths(tmp_path):
    original = beat(tmp_path)
    generated = beats_with_generated_image_paths([original], tmp_path / "assets" / "generated_images")

    assert generated[0] == replace(original, image_path=(tmp_path / "assets" / "generated_images" / "beat_001.png").as_posix())
    assert original.image_path.endswith("assets/images/beat_001.png")


def test_load_and_validate_generated_images_uses_beats_json(tmp_path):
    beats_path = tmp_path / "data" / "beats.json"
    image_dir = tmp_path / "assets" / "generated_images"
    write_beats(beats_path, [beat(tmp_path)])
    write_png(image_dir / "beat_001.png", (1920, 1080))

    beats, report = load_and_validate_generated_images(beats_path, image_dir)

    assert report.ok
    assert [item.beat_number for item in beats] == [1]
