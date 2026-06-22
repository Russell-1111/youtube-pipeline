from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .beat_io import load_beats
from .errors import PipelineError
from .models import Beat

MIN_WIDTH = 1280
MIN_HEIGHT = 720
PREFERRED_WIDTH = 1920
PREFERRED_HEIGHT = 1080


@dataclass(frozen=True)
class GeneratedImageValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_generated_images(beats: list[Beat], image_dir: Path) -> GeneratedImageValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    expected_names = {generated_image_filename(beat) for beat in beats}

    for beat in beats:
        image_path = image_dir / generated_image_filename(beat)
        if not image_path.exists():
            errors.append(f"Missing generated image for beat {beat.beat_number:03}: {image_path}")
            continue
        if not image_path.is_file():
            errors.append(f"Generated image path is not a file for beat {beat.beat_number:03}: {image_path}")
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
                image_format = image.format
        except (OSError, UnidentifiedImageError) as exc:
            errors.append(f"Could not open generated image for beat {beat.beat_number:03}: {image_path} ({exc})")
            continue

        if image_format != "PNG":
            errors.append(f"Generated image must be PNG for beat {beat.beat_number:03}: {image_path} is {image_format}")
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            errors.append(
                f"Generated image is too small for beat {beat.beat_number:03}: "
                f"{image_path} is {width}x{height}, minimum is {MIN_WIDTH}x{MIN_HEIGHT}"
            )
        if width * 9 != height * 16:
            errors.append(
                f"Generated image must be exact 16:9 for beat {beat.beat_number:03}: "
                f"{image_path} is {width}x{height}"
            )
        elif (width, height) != (PREFERRED_WIDTH, PREFERRED_HEIGHT):
            warnings.append(
                f"Generated image for beat {beat.beat_number:03} is {width}x{height}; "
                f"preferred size is {PREFERRED_WIDTH}x{PREFERRED_HEIGHT}: {image_path}"
            )

    if image_dir.exists():
        for extra_path in sorted(image_dir.iterdir()):
            if extra_path.is_file() and extra_path.name not in expected_names:
                warnings.append(f"Extra generated image file ignored: {extra_path}")

    return GeneratedImageValidationReport(errors=errors, warnings=warnings)


def load_and_validate_generated_images(beats_path: Path, image_dir: Path) -> tuple[list[Beat], GeneratedImageValidationReport]:
    beats = load_beats(beats_path)
    return beats, validate_generated_images(beats, image_dir)


def beats_with_generated_image_paths(beats: list[Beat], image_dir: Path) -> list[Beat]:
    return [
        replace(beat, image_path=(image_dir / generated_image_filename(beat)).as_posix())
        for beat in beats
    ]


def require_valid_generated_images(beats_path: Path, image_dir: Path) -> list[Beat]:
    beats, report = load_and_validate_generated_images(beats_path, image_dir)
    if not report.ok:
        raise PipelineError(_format_validation_failure(report))
    return beats_with_generated_image_paths(beats, image_dir)


def generated_image_filename(beat: Beat) -> str:
    return f"beat_{beat.beat_number:03}.png"


def _format_validation_failure(report: GeneratedImageValidationReport) -> str:
    lines = ["Generated image validation failed:"]
    lines.extend(f"- {error}" for error in report.errors)
    if report.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in report.warnings)
    return "\n".join(lines)
