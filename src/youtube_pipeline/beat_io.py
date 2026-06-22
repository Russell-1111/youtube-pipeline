from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import InputFileError, PipelineError
from .models import Beat, TranscriptSegment


def load_beats(path: Path) -> list[Beat]:
    if not path.exists():
        raise InputFileError(f"Missing beats file: {path}. Run: python -m youtube_pipeline --dry-run")

    payload = _read_json(path, "beats")
    if not isinstance(payload, list):
        raise PipelineError(f"Invalid beats JSON: expected a list in {path}")

    beats: list[Beat] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise PipelineError(f"Invalid beat record at index {index} in {path}: expected an object")
        beats.append(_beat_from_dict(item, path, index))
    return beats


def load_transcript_segments(path: Path) -> list[TranscriptSegment]:
    if not path.exists():
        raise InputFileError(f"Missing transcript segments file: {path}")

    payload = _read_json(path, "transcript segments")
    if not isinstance(payload, list):
        raise PipelineError(f"Invalid transcript segments JSON: expected a list in {path}")

    segments: list[TranscriptSegment] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise PipelineError(f"Invalid transcript segment at index {index} in {path}: expected an object")
        segments.append(_segment_from_dict(item, path, index))
    return segments


def try_load_transcript_segments(path: Path) -> list[TranscriptSegment] | None:
    try:
        return load_transcript_segments(path)
    except PipelineError:
        return None


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid {label} JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise PipelineError(f"Could not read {label} JSON from {path}: {exc}") from exc


def _beat_from_dict(item: dict[str, Any], path: Path, index: int) -> Beat:
    try:
        return Beat(
            beat_number=_int(item, "beat_number"),
            beat_type=_str(item, "beat_type"),
            start=_str(item, "start"),
            end=_str(item, "end"),
            start_seconds=_float(item, "start_seconds"),
            end_seconds=_float(item, "end_seconds"),
            duration_seconds=_float(item, "duration_seconds"),
            text_preview=_str(item, "text_preview"),
            segment_indexes=_int_list(item, "segment_indexes"),
            image_path=_str(item, "image_path"),
        )
    except PipelineError as exc:
        raise PipelineError(f"Invalid beat record at index {index} in {path}: {exc}") from exc


def _segment_from_dict(item: dict[str, Any], path: Path, index: int) -> TranscriptSegment:
    try:
        return TranscriptSegment(
            index=_int(item, "index"),
            start=_str(item, "start"),
            end=_str(item, "end"),
            start_seconds=_float(item, "start_seconds"),
            end_seconds=_float(item, "end_seconds"),
            duration_seconds=_float(item, "duration_seconds"),
            text=_str(item, "text"),
        )
    except PipelineError as exc:
        raise PipelineError(f"Invalid transcript segment at index {index} in {path}: {exc}") from exc


def _str(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str):
        raise PipelineError(f"missing or invalid string field: {key}")
    return value


def _int(item: dict[str, Any], key: str) -> int:
    value = item.get(key)
    if not isinstance(value, int):
        raise PipelineError(f"missing or invalid integer field: {key}")
    return value


def _float(item: dict[str, Any], key: str) -> float:
    value = item.get(key)
    if not isinstance(value, (int, float)):
        raise PipelineError(f"missing or invalid numeric field: {key}")
    return float(value)


def _int_list(item: dict[str, Any], key: str) -> list[int]:
    value = item.get(key)
    if not isinstance(value, list) or not all(isinstance(entry, int) for entry in value):
        raise PipelineError(f"missing or invalid integer list field: {key}")
    return value
