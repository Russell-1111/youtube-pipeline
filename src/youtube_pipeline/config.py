from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError


@dataclass(frozen=True)
class InputConfig:
    voiceover: Path
    transcript: Path


@dataclass(frozen=True)
class OutputConfig:
    data_dir: Path
    images_dir: Path
    contact_sheet: Path
    final_video: Path


@dataclass(frozen=True)
class VideoConfig:
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class BeatConfig:
    min_duration: float
    target_duration: float
    max_duration: float
    min_gap_beat_duration: float
    min_intro_beat_duration: float
    max_preview_chars: int


@dataclass(frozen=True)
class TimingConfig:
    duration_mismatch_tolerance: float


@dataclass(frozen=True)
class PipelineConfig:
    inputs: InputConfig
    outputs: OutputConfig
    video: VideoConfig
    beats: BeatConfig
    timing: TimingConfig


def load_config(config_path: Path) -> PipelineConfig:
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML config: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Config must be a YAML mapping.")

    base_dir = config_path.resolve().parent
    return PipelineConfig(
        inputs=InputConfig(
            voiceover=_path(raw, "inputs", "voiceover", base_dir),
            transcript=_path(raw, "inputs", "transcript", base_dir),
        ),
        outputs=OutputConfig(
            data_dir=_path(raw, "outputs", "data_dir", base_dir),
            images_dir=_path(raw, "outputs", "images_dir", base_dir),
            contact_sheet=_path(raw, "outputs", "contact_sheet", base_dir),
            final_video=_path(raw, "outputs", "final_video", base_dir),
        ),
        video=VideoConfig(
            width=_positive_int(raw, "video", "width"),
            height=_positive_int(raw, "video", "height"),
            fps=_positive_int(raw, "video", "fps"),
        ),
        beats=BeatConfig(
            min_duration=_positive_number(raw, "beats", "min_duration"),
            target_duration=_positive_number(raw, "beats", "target_duration"),
            max_duration=_positive_number(raw, "beats", "max_duration"),
            min_gap_beat_duration=_positive_number(
                raw, "beats", "min_gap_beat_duration", default=1.5, allow_zero=True
            ),
            min_intro_beat_duration=_positive_number(
                raw, "beats", "min_intro_beat_duration", default=1.0, allow_zero=True
            ),
            max_preview_chars=_positive_int(raw, "beats", "max_preview_chars", default=80),
        ),
        timing=TimingConfig(
            duration_mismatch_tolerance=_positive_number(raw, "timing", "duration_mismatch_tolerance", allow_zero=True),
        ),
    )


def validate_config(config: PipelineConfig) -> None:
    if config.video.width <= 0 or config.video.height <= 0 or config.video.fps <= 0:
        raise ConfigError("Video width, height, and fps must be positive.")
    if not (0 < config.beats.min_duration <= config.beats.target_duration <= config.beats.max_duration):
        raise ConfigError("Beat durations must satisfy min_duration <= target_duration <= max_duration.")
    if config.beats.max_preview_chars < 10:
        raise ConfigError("beats.max_preview_chars must be at least 10.")


def ensure_output_dirs(config: PipelineConfig) -> None:
    config.outputs.data_dir.mkdir(parents=True, exist_ok=True)
    config.outputs.images_dir.mkdir(parents=True, exist_ok=True)
    config.outputs.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    config.outputs.final_video.parent.mkdir(parents=True, exist_ok=True)


def _section(raw: dict[str, Any], section: str) -> dict[str, Any]:
    value = raw.get(section)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid config section: {section}")
    return value


def _path(raw: dict[str, Any], section: str, key: str, base_dir: Path) -> Path:
    value = _section(raw, section).get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing or invalid path config: {section}.{key}")
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _positive_number(
    raw: dict[str, Any],
    section: str,
    key: str,
    allow_zero: bool = False,
    default: float | None = None,
) -> float:
    value = _section(raw, section).get(key, default)
    if not isinstance(value, (int, float)):
        raise ConfigError(f"Missing or invalid numeric config: {section}.{key}")
    value = float(value)
    if allow_zero:
        valid = value >= 0
    else:
        valid = value > 0
    if not valid:
        raise ConfigError(f"Config value must be positive: {section}.{key}")
    return value


def _positive_int(raw: dict[str, Any], section: str, key: str, default: int | None = None) -> int:
    value = _positive_number(raw, section, key, default=default)
    if int(value) != value:
        raise ConfigError(f"Config value must be an integer: {section}.{key}")
    return int(value)
