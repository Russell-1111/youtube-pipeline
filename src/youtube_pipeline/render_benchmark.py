from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from moviepy import AudioFileClip, VideoFileClip

from .errors import PipelineError
from .models import Beat
from .render import render_video_kinetic
from .render_ffmpeg import beats_for_stress_test, render_video_kinetic_ffmpeg


@dataclass(frozen=True)
class RenderBenchmarkReport:
    renderer_name: str
    elapsed_seconds: float
    beat_count: int
    video_duration_seconds: float | None
    audio_duration_seconds: float | None
    output_path: str
    output_size_bytes: int | None
    width: int | None
    height: int | None
    fps: float | None
    audio_present: bool
    duration_delta_seconds: float | None
    success: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_kinetic_benchmark(
    *,
    renderer_name: str,
    beats: list[Beat],
    audio_path: Path,
    output_dir: Path,
    fps: int,
    width: int,
    height: int,
    sample_beats: int | None = 5,
    full: bool = False,
    stress_beats: int | None = None,
    stress_duration: float | None = None,
) -> RenderBenchmarkReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_beats = _select_beats(beats, sample_beats, full, stress_beats, stress_duration)
    audio_duration = _audio_duration(audio_path)
    target_audio_duration = sum(beat.duration_seconds for beat in selected_beats)
    suffix = _output_suffix(full, sample_beats, stress_beats)
    output_path = output_dir / f"{renderer_name}_{suffix}.mp4"

    start = time.perf_counter()
    warnings: list[str] = []
    errors: list[str] = []
    try:
        if renderer_name == "ffmpeg":
            render_video_kinetic_ffmpeg(
                selected_beats,
                audio_path,
                output_path,
                fps,
                width,
                height,
                work_dir=output_dir / f"{renderer_name}_{suffix}_segments",
                audio_duration_seconds=target_audio_duration,
            )
            warnings.append(
                "Experimental FFmpeg motion reuses MoviePy profile values but still requires visual frame review."
            )
        elif renderer_name == "moviepy":
            sample_audio = _write_sample_audio(audio_path, output_dir / f"{renderer_name}_{suffix}_audio.m4a", target_audio_duration)
            render_video_kinetic(selected_beats, sample_audio, output_path, fps, width, height)
        else:
            raise PipelineError(f"Unsupported benchmark renderer: {renderer_name}")
        elapsed = time.perf_counter() - start
        report = inspect_video_metadata(
            renderer_name=renderer_name,
            elapsed_seconds=elapsed,
            beat_count=len(selected_beats),
            expected_audio_duration=target_audio_duration,
            output_path=output_path,
            warnings=warnings,
            errors=errors,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        errors.append(str(exc))
        report = RenderBenchmarkReport(
            renderer_name=renderer_name,
            elapsed_seconds=round(elapsed, 3),
            beat_count=len(selected_beats),
            video_duration_seconds=None,
            audio_duration_seconds=round(audio_duration, 3),
            output_path=output_path.as_posix(),
            output_size_bytes=output_path.stat().st_size if output_path.exists() else None,
            width=None,
            height=None,
            fps=None,
            audio_present=False,
            duration_delta_seconds=None,
            success=False,
            warnings=warnings,
            errors=errors,
        )

    report_path = output_dir / "render_benchmark_report.json"
    write_benchmark_report(report_path, report)
    return report


def inspect_video_metadata(
    *,
    renderer_name: str,
    elapsed_seconds: float,
    beat_count: int,
    expected_audio_duration: float,
    output_path: Path,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> RenderBenchmarkReport:
    warnings = list(warnings or [])
    errors = list(errors or [])
    with VideoFileClip(str(output_path)) as clip:
        audio_present = clip.audio is not None
        audio_duration = float(clip.audio.duration) if clip.audio is not None else None
        video_duration = float(clip.duration)
        duration_delta = None if audio_duration is None else abs(video_duration - audio_duration)
        if not audio_present:
            errors.append("Rendered output has no audio stream.")
        if duration_delta is not None and duration_delta > 0.25:
            warnings.append(f"Audio/video duration delta is {duration_delta:.3f}s.")
        return RenderBenchmarkReport(
            renderer_name=renderer_name,
            elapsed_seconds=round(elapsed_seconds, 3),
            beat_count=beat_count,
            video_duration_seconds=round(video_duration, 3),
            audio_duration_seconds=None if audio_duration is None else round(audio_duration, 3),
            output_path=output_path.as_posix(),
            output_size_bytes=output_path.stat().st_size,
            width=int(clip.size[0]),
            height=int(clip.size[1]),
            fps=float(clip.fps),
            audio_present=audio_present,
            duration_delta_seconds=None if duration_delta is None else round(duration_delta, 3),
            success=not errors,
            warnings=warnings,
            errors=errors,
        )


def write_benchmark_report(path: Path, report: RenderBenchmarkReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _select_beats(
    beats: list[Beat],
    sample_beats: int | None,
    full: bool,
    stress_beats: int | None,
    stress_duration: float | None,
) -> list[Beat]:
    if stress_beats is not None:
        if stress_duration is None:
            raise PipelineError("--stress-duration is required with --stress-beats.")
        return beats_for_stress_test(beats, stress_beats, stress_duration)
    if full:
        return beats
    count = sample_beats or 5
    if count <= 0:
        raise PipelineError("--sample-beats must be positive.")
    return beats[:count]


def _output_suffix(full: bool, sample_beats: int | None, stress_beats: int | None) -> str:
    if stress_beats is not None:
        return f"stress_{stress_beats}beats"
    if full:
        return "full"
    return f"sample_{sample_beats or 5}beats"


def _audio_duration(audio_path: Path) -> float:
    with AudioFileClip(str(audio_path)) as audio:
        return float(audio.duration)


def _write_sample_audio(audio_path: Path, output_path: Path, duration_seconds: float) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with AudioFileClip(str(audio_path)) as audio:
        sample = audio.subclipped(0, min(duration_seconds, float(audio.duration)))
        try:
            sample.write_audiofile(str(output_path), codec="aac")
        finally:
            sample.close()
    return output_path
