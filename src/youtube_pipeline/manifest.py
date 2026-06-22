from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import Beat, TranscriptSegment


def write_transcript_segments(path: Path, segments: list[TranscriptSegment]) -> None:
    _write_json(path, [segment.to_dict() for segment in segments])


def write_beats(path: Path, beats: list[Beat]) -> None:
    _write_json(path, [beat.to_dict() for beat in beats])


def write_manifest(path: Path, beats: list[Beat]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "beat_number",
                "beat_type",
                "start",
                "end",
                "duration_seconds",
                "image_path",
                "text_preview",
            ],
        )
        writer.writeheader()
        for beat in beats:
            writer.writerow(
                {
                    "beat_number": beat.beat_number,
                    "beat_type": beat.beat_type,
                    "start": beat.start,
                    "end": beat.end,
                    "duration_seconds": f"{beat.duration_seconds:.3f}",
                    "image_path": beat.image_path,
                    "text_preview": beat.text_preview,
                }
            )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
