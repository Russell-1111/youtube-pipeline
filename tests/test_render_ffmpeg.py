from pathlib import Path

import pytest

from youtube_pipeline.errors import PipelineError
from youtube_pipeline.models import Beat
from youtube_pipeline.render_ffmpeg import beats_for_stress_test, render_video_kinetic_ffmpeg


def beat(tmp_path: Path, beat_number: int = 1, duration: float = 2.0) -> Beat:
    image_path = tmp_path / "assets" / "generated_images" / f"beat_{beat_number:03}.png"
    return Beat(
        beat_number=beat_number,
        beat_type="normal",
        start="00:00:00,000",
        end="00:00:02,000",
        start_seconds=0.0,
        end_seconds=duration,
        duration_seconds=duration,
        text_preview="Preview",
        segment_indexes=[1],
        image_path=image_path.as_posix(),
    )


def test_render_video_kinetic_ffmpeg_uses_per_beat_segment_commands(tmp_path, monkeypatch):
    commands = []

    monkeypatch.setattr("youtube_pipeline.render_ffmpeg._ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr("youtube_pipeline.render_ffmpeg._run_ffmpeg", lambda command: commands.append(command))

    result = render_video_kinetic_ffmpeg(
        [beat(tmp_path)],
        tmp_path / "voiceover.mp3",
        tmp_path / "output" / "ffmpeg.mp4",
        fps=30,
        width=1920,
        height=1080,
        work_dir=tmp_path / "segments",
        audio_duration_seconds=2.0,
    )

    assert result.output_path == tmp_path / "output" / "ffmpeg.mp4"
    assert result.beat_count == 1
    assert result.video_duration_seconds == pytest.approx(2.0)
    assert len(commands) == 3
    assert "zoompan" in commands[0][commands[0].index("-vf") + 1]
    assert commands[1][:6] == ["ffmpeg", "-y", "-f", "concat", "-safe", "0"]
    assert "-map" in commands[2]
    assert (tmp_path / "segments" / "concat.txt").exists()


def test_render_video_kinetic_ffmpeg_extends_last_segment_to_audio_duration(tmp_path, monkeypatch):
    commands = []

    monkeypatch.setattr("youtube_pipeline.render_ffmpeg._ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr("youtube_pipeline.render_ffmpeg._run_ffmpeg", lambda command: commands.append(command))

    result = render_video_kinetic_ffmpeg(
        [beat(tmp_path, duration=2.0)],
        tmp_path / "voiceover.mp3",
        tmp_path / "output" / "ffmpeg.mp4",
        fps=30,
        width=1920,
        height=1080,
        work_dir=tmp_path / "segments",
        audio_duration_seconds=2.5,
    )

    assert result.video_duration_seconds == pytest.approx(2.5)
    assert commands[0][commands[0].index("-frames:v") + 1] == "75"


def test_beats_for_stress_test_cycles_images_and_sets_target_duration(tmp_path):
    source = [beat(tmp_path, 1, 2.0), beat(tmp_path, 2, 2.0)]

    stress = beats_for_stress_test(source, beat_count=5, duration_seconds=50.0)

    assert len(stress) == 5
    assert stress[0].image_path == source[0].image_path
    assert stress[1].image_path == source[1].image_path
    assert stress[2].image_path == source[0].image_path
    assert stress[-1].end_seconds == pytest.approx(50.0)
    assert sum(item.duration_seconds for item in stress) == pytest.approx(50.0)


def test_beats_for_stress_test_rejects_empty_source():
    with pytest.raises(PipelineError):
        beats_for_stress_test([], beat_count=70, duration_seconds=540.0)
