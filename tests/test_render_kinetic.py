from pathlib import Path

import numpy as np
from PIL import Image

from youtube_pipeline.render import create_kinetic_image_clip, kinetic_motion_profile_for_beat


def write_test_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1920, 1080), color).save(path)


def test_kinetic_clip_sampled_frames_keep_target_dimensions(tmp_path):
    image_path = tmp_path / "beat_001.png"
    write_test_image(image_path, (20, 120, 220))

    clip = create_kinetic_image_clip(
        image_path,
        duration=2.0,
        beat_number=1,
        width=1920,
        height=1080,
        background_color=(247, 13, 199),
    )
    try:
        for timestamp in (0.0, 1.0, 1.99):
            frame = clip.get_frame(timestamp)
            assert frame.shape[0] == 1080
            assert frame.shape[1] == 1920
    finally:
        clip.close()


def test_kinetic_clip_does_not_expose_background_color(tmp_path):
    image_path = tmp_path / "beat_002.png"
    write_test_image(image_path, (20, 120, 220))
    background_color = np.array([247, 13, 199], dtype=np.uint8)

    clip = create_kinetic_image_clip(
        image_path,
        duration=2.0,
        beat_number=2,
        width=1920,
        height=1080,
        background_color=tuple(int(channel) for channel in background_color),
    )
    try:
        for timestamp in (0.0, 1.0, 1.99):
            frame = clip.get_frame(timestamp)
            background_pixels = np.all(frame[:, :, :3] == background_color, axis=2)
            assert not np.any(background_pixels)
    finally:
        clip.close()


def test_kinetic_motion_profile_selection_is_deterministic():
    assert kinetic_motion_profile_for_beat(1).name == "push_in"
    assert kinetic_motion_profile_for_beat(2).name == "pan_right"
    assert kinetic_motion_profile_for_beat(3).name == "push_out"
    assert kinetic_motion_profile_for_beat(8) == kinetic_motion_profile_for_beat(1)
