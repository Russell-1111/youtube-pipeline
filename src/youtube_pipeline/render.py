from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import Beat


@dataclass(frozen=True)
class KineticMotionProfile:
    name: str
    start_scale: float
    end_scale: float
    start_x_fraction: float
    end_x_fraction: float
    start_y_fraction: float
    end_y_fraction: float


KINETIC_MOTION_PROFILES: tuple[KineticMotionProfile, ...] = (
    KineticMotionProfile("push_in", 1.03, 1.06, 0.5, 0.5, 0.5, 0.5),
    KineticMotionProfile("pan_right", 1.05, 1.05, 0.2, 0.8, 0.5, 0.5),
    KineticMotionProfile("push_out", 1.06, 1.03, 0.5, 0.5, 0.5, 0.5),
    KineticMotionProfile("pan_left", 1.05, 1.05, 0.8, 0.2, 0.5, 0.5),
    KineticMotionProfile("push_in_up", 1.03, 1.06, 0.5, 0.5, 0.6, 0.4),
    KineticMotionProfile("drift_down", 1.04, 1.06, 0.5, 0.5, 0.35, 0.65),
    KineticMotionProfile("diagonal_drift", 1.04, 1.06, 0.35, 0.65, 0.6, 0.4),
)


def get_audio_duration(audio_path: Path) -> float:
    from moviepy import AudioFileClip

    with AudioFileClip(str(audio_path)) as audio:
        return float(audio.duration)


def render_video(beats: list[Beat], audio_path: Path, output_path: Path, fps: int) -> None:
    from moviepy import AudioFileClip, ImageClip, concatenate_videoclips

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clips = [ImageClip(str(Path(beat.image_path))).with_duration(beat.duration_seconds) for beat in beats]
    try:
        video = concatenate_videoclips(clips, method="compose")
        audio = AudioFileClip(str(audio_path))
        try:
            final = video.with_audio(audio)
            final.write_videofile(
                str(output_path),
                fps=fps,
                codec="libx264",
                audio_codec="aac",
                preset="medium",
            )
        finally:
            audio.close()
            video.close()
            if "final" in locals():
                final.close()
    finally:
        for clip in clips:
            clip.close()


def render_video_kinetic(
    beats: list[Beat],
    audio_path: Path,
    output_path: Path,
    fps: int,
    width: int,
    height: int,
) -> None:
    from moviepy import AudioFileClip, concatenate_videoclips

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clips = [
        create_kinetic_image_clip(
            Path(beat.image_path),
            duration=beat.duration_seconds,
            beat_number=beat.beat_number,
            width=width,
            height=height,
        )
        for beat in beats
    ]
    try:
        video = concatenate_videoclips(clips, method="compose")
        audio = AudioFileClip(str(audio_path))
        try:
            final = video.with_audio(audio)
            final.write_videofile(
                str(output_path),
                fps=fps,
                codec="libx264",
                audio_codec="aac",
                preset="medium",
            )
        finally:
            audio.close()
            video.close()
            if "final" in locals():
                final.close()
    finally:
        for clip in clips:
            clip.close()


def create_kinetic_image_clip(
    image_path: Path,
    duration: float,
    beat_number: int,
    width: int,
    height: int,
    background_color: tuple[int, int, int] | None = None,
):
    from moviepy import CompositeVideoClip, ImageClip

    profile = kinetic_motion_profile_for_beat(beat_number)
    base_clip = ImageClip(str(image_path)).resized((width, height)).with_duration(duration)
    moving_clip = base_clip.resized(lambda t: _scale_at(profile, t, duration)).with_position(
        lambda t: _position_at(profile, t, duration, width, height)
    )
    return CompositeVideoClip([moving_clip], size=(width, height), bg_color=background_color).with_duration(duration)


def kinetic_motion_profile_for_beat(beat_number: int) -> KineticMotionProfile:
    return KINETIC_MOTION_PROFILES[(beat_number - 1) % len(KINETIC_MOTION_PROFILES)]


def _scale_at(profile: KineticMotionProfile, t: float, duration: float) -> float:
    progress = _progress(t, duration)
    return profile.start_scale + (profile.end_scale - profile.start_scale) * progress


def _position_at(
    profile: KineticMotionProfile,
    t: float,
    duration: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    progress = _progress(t, duration)
    scale = _scale_at(profile, t, duration)
    x_fraction = profile.start_x_fraction + (profile.end_x_fraction - profile.start_x_fraction) * progress
    y_fraction = profile.start_y_fraction + (profile.end_y_fraction - profile.start_y_fraction) * progress
    max_x_offset = width * scale - width
    max_y_offset = height * scale - height
    return (-max_x_offset * x_fraction, -max_y_offset * y_fraction)


def _progress(t: float, duration: float) -> float:
    if duration <= 0:
        return 1.0
    return min(max(t / duration, 0.0), 1.0)
