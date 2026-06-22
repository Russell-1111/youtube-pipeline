from __future__ import annotations

import re
from pathlib import Path

from .errors import SRTParseError
from .models import TranscriptSegment

TIMESTAMP_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?$"
)


def parse_srt_file(path: Path) -> list[TranscriptSegment]:
    return parse_srt(path.read_text(encoding="utf-8-sig"))


def parse_srt(content: str) -> list[TranscriptSegment]:
    blocks = re.split(r"\r?\n\s*\r?\n", content.strip())
    segments: list[TranscriptSegment] = []

    for block_number, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        if len(lines) < 2:
            raise SRTParseError(f"SRT block {block_number} is incomplete.")

        timestamp_line_index = 1 if lines[0].isdigit() else 0
        if timestamp_line_index >= len(lines):
            raise SRTParseError(f"SRT block {block_number} is missing a timestamp line.")

        match = TIMESTAMP_RE.match(lines[timestamp_line_index])
        if not match:
            raise SRTParseError(f"Invalid timestamp line in SRT block {block_number}: {lines[timestamp_line_index]}")

        start = _parse_timestamp(match.group("start"))
        end = _parse_timestamp(match.group("end"))
        if end <= start:
            raise SRTParseError(f"SRT block {block_number} end timestamp must be after start timestamp.")

        text_lines = lines[timestamp_line_index + 1 :]
        if not text_lines:
            raise SRTParseError(f"SRT block {block_number} has no caption text.")

        if segments and start < segments[-1].end_seconds:
            raise SRTParseError(
                f"SRT block {block_number} starts before the previous segment ends; transcript order is non-monotonic."
            )

        segment_index = len(segments) + 1
        segments.append(
            TranscriptSegment(
                index=segment_index,
                start=_normalize_timestamp(match.group("start")),
                end=_normalize_timestamp(match.group("end")),
                start_seconds=start,
                end_seconds=end,
                duration_seconds=end - start,
                text=" ".join(text_lines),
            )
        )

    if not segments:
        raise SRTParseError("Transcript is empty after parsing.")

    return segments


def _parse_timestamp(timestamp: str) -> float:
    normalized = timestamp.replace(".", ",")
    hours_text, minutes_text, rest = normalized.split(":")
    seconds_text, millis_text = rest.split(",")
    hours = int(hours_text)
    minutes = int(minutes_text)
    seconds = int(seconds_text)
    millis = int(millis_text)
    if minutes >= 60 or seconds >= 60:
        raise SRTParseError(f"Invalid timestamp value: {timestamp}")
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def _normalize_timestamp(timestamp: str) -> str:
    return timestamp.replace(".", ",")
