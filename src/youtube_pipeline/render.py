from __future__ import annotations

from pathlib import Path

from .models import Beat


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
