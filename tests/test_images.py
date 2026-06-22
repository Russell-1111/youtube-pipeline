from pathlib import Path

from PIL import Image

from youtube_pipeline.contact_sheet import generate_contact_sheet
from youtube_pipeline.images import generate_placeholder_images
from youtube_pipeline.models import Beat


def beat(tmp_path: Path, number: int, beat_type: str, preview: str) -> Beat:
    return Beat(
        beat_number=number,
        beat_type=beat_type,
        start="00:00:00,000",
        end="00:00:06,000",
        start_seconds=0.0,
        end_seconds=6.0,
        duration_seconds=6.0,
        text_preview=preview,
        segment_indexes=[] if beat_type in {"intro", "gap"} else [number],
        image_path=(tmp_path / f"beat_{number:03}.png").as_posix(),
    )


def test_whiteboard_placeholder_images_generate(tmp_path):
    beats = [
        beat(tmp_path, 1, "intro", "Intro / silence"),
        beat(tmp_path, 2, "normal", "Learn the history and timeline"),
        beat(tmp_path, 3, "gap", "Pause / silence"),
    ]

    generate_placeholder_images(beats, 640, 360)

    for item in beats:
        output_path = Path(item.image_path)
        assert output_path.exists()
        with Image.open(output_path) as image:
            assert image.size == (640, 360)
            assert image.mode == "RGB"
            assert image.getpixel((10, 10)) == (255, 255, 255)


def test_placeholder_generation_removes_only_stale_beat_images(tmp_path):
    stale = tmp_path / "beat_999.png"
    unrelated = tmp_path / "logo.png"
    stale.write_bytes(b"stale")
    unrelated.write_bytes(b"keep")
    beats = [beat(tmp_path, 1, "normal", "Keep only current beat image")]

    generate_placeholder_images(beats, 640, 360)

    assert not stale.exists()
    assert unrelated.read_bytes() == b"keep"
    assert (tmp_path / "beat_001.png").exists()


def test_whiteboard_contact_sheet_generates(tmp_path):
    beats = [
        beat(tmp_path, 1, "normal", "Switch the button off"),
        beat(tmp_path, 2, "normal", "Sleep in bed at night"),
    ]
    output_path = tmp_path / "contact_sheet.png"

    generate_contact_sheet(beats, output_path)

    assert output_path.exists()
    with Image.open(output_path) as image:
        assert image.width > 640
        assert image.height >= 360
        assert image.mode == "RGB"
