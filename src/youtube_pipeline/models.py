from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TranscriptSegment:
    index: int
    start: str
    end: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Beat:
    beat_number: int
    beat_type: str
    start: str
    end: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    text_preview: str
    segment_indexes: list[int]
    image_path: str

    def to_dict(self) -> dict:
        return asdict(self)
