from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import imageio_ffmpeg

from .errors import PipelineError
from .models import Beat
from .render import kinetic_motion_profile_for_beat


@dataclass(frozen=True)
class FFmpegRenderResult:
    output_path: Path
    segment_dir: Path
    concat_video_path: Path
    video_duration_seconds: float
    beat_count: int
    motion_equivalence_note: str


def render_video_kinetic_ffmpeg(
    beats: list[Beat],
    audio_path: Path,
    output_path: Path,
    fps: int,
    width: int,
    height: int,
    *,
    work_dir: Path | None = None,
    audio_duration_seconds: float | None = None,
) -> FFmpegRenderResult:
    if not beats:
        raise PipelineError("Cannot render FFmpeg kinetic video without beats.")

    ffmpeg = _ffmpeg_exe()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment_dir = work_dir or output_path.parent / f"{output_path.stem}_segments"
    if segment_dir.exists():
        shutil.rmtree(segment_dir)
    segment_dir.mkdir(parents=True, exist_ok=True)

    beat_frames = _frame_counts(beats, fps, audio_duration_seconds)
    segment_paths: list[Path] = []
    for beat, frames in zip(beats, beat_frames, strict=True):
        segment_path = segment_dir / f"beat_{beat.beat_number:03}.mp4"
        _render_segment(ffmpeg, beat, segment_path, frames, fps, width, height)
        segment_paths.append(segment_path)

    concat_list = segment_dir / "concat.txt"
    _write_concat_list(concat_list, segment_paths)
    concat_video_path = segment_dir / "video_concat.mp4"
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(concat_video_path),
        ]
    )

    video_duration = sum(beat_frames) / fps
    _mux_audio(ffmpeg, concat_video_path, audio_path, output_path, video_duration)
    return FFmpegRenderResult(
        output_path=output_path,
        segment_dir=segment_dir,
        concat_video_path=concat_video_path,
        video_duration_seconds=video_duration,
        beat_count=len(beats),
        motion_equivalence_note=(
            "Experimental FFmpeg zoompan renderer reuses MoviePy motion profile values, "
            "but visual equivalence still requires frame review."
        ),
    )


def beats_for_stress_test(source_beats: list[Beat], beat_count: int, duration_seconds: float) -> list[Beat]:
    if not source_beats:
        raise PipelineError("Cannot build stress-test beats without source beats.")
    if beat_count <= 0:
        raise PipelineError("Stress beat count must be positive.")
    if duration_seconds <= 0:
        raise PipelineError("Stress duration must be positive.")

    duration_per_beat = duration_seconds / beat_count
    stress_beats: list[Beat] = []
    for index in range(beat_count):
        source = source_beats[index % len(source_beats)]
        start = index * duration_per_beat
        end = duration_seconds if index == beat_count - 1 else (index + 1) * duration_per_beat
        stress_beats.append(
            replace(
                source,
                beat_number=index + 1,
                start_seconds=start,
                end_seconds=end,
                duration_seconds=end - start,
                start=_seconds_to_timestamp(start),
                end=_seconds_to_timestamp(end),
            )
        )
    return stress_beats


def _ffmpeg_exe() -> str:
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - defensive dependency failure path
        raise PipelineError(f"Could not locate FFmpeg through imageio-ffmpeg: {exc}") from exc


def _frame_counts(beats: list[Beat], fps: int, target_duration_seconds: float | None) -> list[int]:
    counts = [max(1, int(round(beat.duration_seconds * fps))) for beat in beats]
    if target_duration_seconds is not None:
        target_frames = max(1, int(math.ceil(target_duration_seconds * fps)))
        difference = target_frames - sum(counts)
        if difference > 0:
            counts[-1] += difference
    return counts


def _render_segment(ffmpeg: str, beat: Beat, output_path: Path, frames: int, fps: int, width: int, height: int) -> None:
    profile = kinetic_motion_profile_for_beat(beat.beat_number)
    denominator = max(frames - 1, 1)
    zoom = _linear_expr(profile.start_scale, profile.end_scale, denominator)
    x_fraction = _linear_expr(profile.start_x_fraction, profile.end_x_fraction, denominator)
    y_fraction = _linear_expr(profile.start_y_fraction, profile.end_y_fraction, denominator)
    x = f"(iw-iw/zoom)*({x_fraction})"
    y = f"(ih-ih/zoom)*({y_fraction})"
    vf = (
        f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}:s={width}x{height}:fps={fps},"
        "format=yuv420p"
    )
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-loop",
            "1",
            "-i",
            str(Path(beat.image_path)),
            "-vf",
            vf,
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )


def _mux_audio(ffmpeg: str, video_path: Path, audio_path: Path, output_path: Path, duration_seconds: float) -> None:
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-t",
            f"{duration_seconds:.6f}",
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )


def _linear_expr(start: float, end: float, denominator: int) -> str:
    delta = end - start
    if abs(delta) < 0.000001:
        return f"{start:.8f}"
    return f"{start:.8f}+({delta:.8f})*on/{denominator}"


def _write_concat_list(path: Path, segment_paths: list[Path]) -> None:
    lines = []
    for segment_path in segment_paths:
        escaped = segment_path.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise PipelineError(f"FFmpeg command failed with exit code {result.returncode}: {stderr}")


def _seconds_to_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"
