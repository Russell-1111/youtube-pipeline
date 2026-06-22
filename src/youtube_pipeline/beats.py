from __future__ import annotations

from pathlib import Path
import re

from .config import BeatConfig
from .errors import TimingError
from .models import Beat, TranscriptSegment
from .time_utils import seconds_to_timestamp

EPSILON = 0.001


def build_beats(
    segments: list[TranscriptSegment],
    beat_config: BeatConfig,
    images_dir: Path,
    audio_duration: float | None = None,
    duration_mismatch_tolerance: float = 1.0,
) -> list[Beat]:
    if not segments:
        raise TimingError("Cannot build beats from an empty transcript.")

    pending: list[dict] = []

    first_start = segments[0].start_seconds
    absorb_intro = 0.0 < first_start < beat_config.min_intro_beat_duration
    if first_start >= beat_config.min_intro_beat_duration:
        pending.append(
            _pending_beat(
                beat_type="intro",
                start_seconds=0.0,
                end_seconds=first_start,
                text_preview="Intro / silence",
                segment_indexes=[],
            )
        )

    current: list[TranscriptSegment] = []
    current_visual_start: float | None = 0.0 if absorb_intro else None
    previous_end = 0.0 if absorb_intro else first_start

    for segment in segments:
        if current:
            gap = segment.start_seconds - current[-1].end_seconds
            if gap >= beat_config.min_gap_beat_duration:
                pending.extend(_normal_pending(current, current_visual_start, beat_config.max_preview_chars))
                pending.append(
                    _pending_beat(
                        beat_type="gap",
                        start_seconds=current[-1].end_seconds,
                        end_seconds=segment.start_seconds,
                        text_preview="Pause / silence",
                        segment_indexes=[],
                    )
                )
                current = [segment]
                current_visual_start = segment.start_seconds
                previous_end = segment.end_seconds
                continue

            proposed_start = current_visual_start if current_visual_start is not None else current[0].start_seconds
            proposed_end = segment.end_seconds
            proposed_duration = proposed_end - proposed_start
            current_duration = current[-1].end_seconds - proposed_start

            if proposed_duration > beat_config.max_duration and current_duration >= beat_config.min_duration:
                if gap >= beat_config.min_gap_beat_duration:
                    normal_end = current[-1].end_seconds
                elif gap > EPSILON:
                    normal_end = segment.start_seconds
                else:
                    normal_end = current[-1].end_seconds
                pending.extend(_normal_pending(current, current_visual_start, beat_config.max_preview_chars, normal_end))
                if gap >= beat_config.min_gap_beat_duration:
                    pending.append(
                        _pending_beat(
                            beat_type="gap",
                            start_seconds=current[-1].end_seconds,
                            end_seconds=segment.start_seconds,
                            text_preview="Pause / silence",
                            segment_indexes=[],
                        )
                    )
                current = [segment]
                current_visual_start = segment.start_seconds
            else:
                current.append(segment)
        else:
            gap = segment.start_seconds - previous_end
            if gap >= beat_config.min_gap_beat_duration:
                pending.append(
                    _pending_beat(
                        beat_type="gap",
                        start_seconds=previous_end,
                        end_seconds=segment.start_seconds,
                        text_preview="Pause / silence",
                        segment_indexes=[],
                    )
                )
                current_visual_start = segment.start_seconds
            elif gap > EPSILON:
                current_visual_start = previous_end
            elif current_visual_start is None:
                current_visual_start = segment.start_seconds
            current = [segment]

        previous_end = segment.end_seconds

    if current:
        pending.extend(_normal_pending(current, current_visual_start, beat_config.max_preview_chars))

    if audio_duration is not None:
        _apply_audio_duration(pending, audio_duration, duration_mismatch_tolerance)

    return _finalize_beats(pending, images_dir)


def _normal_pending(
    segments: list[TranscriptSegment],
    start_seconds: float | None,
    max_preview_chars: int,
    end_seconds: float | None = None,
) -> list[dict]:
    if not segments:
        return []
    visual_start = segments[0].start_seconds if start_seconds is None else start_seconds
    visual_end = segments[-1].end_seconds if end_seconds is None else end_seconds
    return [
        _pending_beat(
            beat_type="normal",
            start_seconds=visual_start,
            end_seconds=visual_end,
            text_preview=_preview(_clean_preview_text(" ".join(segment.text for segment in segments)), max_preview_chars),
            segment_indexes=[segment.index for segment in segments],
        )
    ]


def _apply_audio_duration(pending: list[dict], audio_duration: float, tolerance: float) -> None:
    if audio_duration <= 0:
        raise TimingError(f"Audio duration must be positive; got {audio_duration:.3f}s.")
    if not pending:
        raise TimingError("No beats available for duration reconciliation.")

    timeline_end = pending[-1]["end_seconds"]
    difference = audio_duration - timeline_end
    if abs(difference) > tolerance:
        raise TimingError(
            "Transcript/audio duration mismatch exceeds tolerance: "
            f"transcript={timeline_end:.3f}s audio={audio_duration:.3f}s difference={difference:.3f}s"
        )
    if difference > EPSILON:
        pending[-1]["end_seconds"] = audio_duration


def _pending_beat(
    beat_type: str,
    start_seconds: float,
    end_seconds: float,
    text_preview: str,
    segment_indexes: list[int],
) -> dict:
    if end_seconds <= start_seconds:
        raise TimingError(f"Invalid beat duration: {start_seconds:.3f}s to {end_seconds:.3f}s.")
    return {
        "beat_type": beat_type,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "text_preview": text_preview,
        "segment_indexes": segment_indexes,
    }


def _finalize_beats(pending: list[dict], images_dir: Path) -> list[Beat]:
    beats: list[Beat] = []
    for index, item in enumerate(pending, start=1):
        image_path = images_dir / f"beat_{index:03}.png"
        start_seconds = item["start_seconds"]
        end_seconds = item["end_seconds"]
        beats.append(
            Beat(
                beat_number=index,
                beat_type=item["beat_type"],
                start=seconds_to_timestamp(start_seconds),
                end=seconds_to_timestamp(end_seconds),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                duration_seconds=end_seconds - start_seconds,
                text_preview=item["text_preview"],
                segment_indexes=item["segment_indexes"],
                image_path=image_path.as_posix(),
            )
        )
    return beats


def _clean_preview_text(text: str) -> str:
    patterns = [
        r"\(?\s*transcribed by turboscribe\.?\s*go unlimited to remove this message\.?\s*\)?",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _preview(text: str, max_chars: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    truncated = compact[: max_chars - 3].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0].rstrip()
    return truncated + "..."
