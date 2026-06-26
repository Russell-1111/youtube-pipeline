from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

from .beat_io import load_beats, load_transcript_segments
from .beats import EPSILON, _clean_preview_text, _preview
from .config import BeatConfig, PipelineConfig
from .errors import PipelineError
from .manifest import write_beats
from .models import Beat, TranscriptSegment
from .time_utils import seconds_to_timestamp

PLANNER_VERSION = 1
PLANNER_STRATEGY = "hybrid_dense_rebuild"
JSON_REPORT_NAME = "dense_beat_plan.json"
MARKDOWN_REPORT_NAME = "dense_beat_plan.md"
PREVIEW_BEATS_NAME = "beats_dense_preview.json"
MAX_REJECTED_EXAMPLES = 20
DENSE_BOUNDARY_EXTENSION_SECONDS = 1.0
LOCAL_REBALANCE_MAX_WINDOW = 5
LOCAL_REBALANCE_MIN_IMPROVEMENT = 10.0

CONTINUATION_WORDS = {
    "and",
    "but",
    "because",
    "that",
    "while",
    "so",
}
TERMINAL_PUNCTUATION = {".", "!", "?", '"', "'", ";"}
WEAK_BOUNDARY_CHARS = {","}
UNFINISHED_END_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "bad",
    "before",
    "because",
    "for",
    "from",
    "if",
    "in",
    "into",
    "no",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "through",
    "to",
    "too",
    "while",
    "with",
    "without",
}
REJECTION_REASON_ORDER = (
    "under_minimum_duration",
    "awkward_continuation_start",
    "weak_punctuation",
    "unsplittable_segment",
    "low_quality_split",
)


@dataclass(frozen=True)
class DenseBeatPlanResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path
    preview_path: Path


@dataclass(frozen=True)
class PendingBeat:
    beat_type: str
    start_seconds: float
    end_seconds: float
    text_preview: str
    segment_indexes: list[int]
    reason: str
    boundary_confidence: str
    score: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def run_dense_beat_plan(config: PipelineConfig, base_dir: Path) -> DenseBeatPlanResult:
    data_dir = config.outputs.data_dir
    source_beats_path = data_dir / "beats.json"
    source_transcript_path = data_dir / "transcript_segments.json"
    preview_path = data_dir / PREVIEW_BEATS_NAME
    json_path = data_dir / JSON_REPORT_NAME
    markdown_path = data_dir / MARKDOWN_REPORT_NAME

    beats = load_beats(source_beats_path)
    segments = load_transcript_segments(source_transcript_path)
    preview_beats, report = build_dense_beat_plan(
        beats=beats,
        segments=segments,
        beat_config=config.beats,
        images_dir=config.outputs.images_dir,
        source_beats_path=source_beats_path,
        source_transcript_path=source_transcript_path,
        preview_beats_path=preview_path,
        base_dir=base_dir,
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    write_beats(preview_path, preview_beats)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return DenseBeatPlanResult(report=report, json_path=json_path, markdown_path=markdown_path, preview_path=preview_path)


def build_dense_beat_plan(
    beats: list[Beat],
    segments: list[TranscriptSegment],
    beat_config: BeatConfig,
    images_dir: Path,
    source_beats_path: Path = Path("data/beats.json"),
    source_transcript_path: Path = Path("data/transcript_segments.json"),
    preview_beats_path: Path = Path("data/beats_dense_preview.json"),
    base_dir: Path | None = None,
) -> tuple[list[Beat], dict[str, Any]]:
    if not beats:
        raise PipelineError("Cannot plan dense beats without source beats.")
    if not segments:
        raise PipelineError("Cannot plan dense beats without transcript segments.")

    standard_lookup = _standard_beat_lookup(beats)
    pending, rejected_examples = _build_dense_pending_beats(segments, beat_config)
    preview_beats = _finalize_pending_beats(pending, images_dir)
    dense_group_records = _dense_group_records(preview_beats, pending, standard_lookup)
    standard_stats = _duration_stats(beats)
    dense_stats = _duration_stats(preview_beats)
    warnings, safe_to_apply = _warnings_and_safety(
        preview_beats=preview_beats,
        segments=segments,
        beat_config=beat_config,
        dense_group_records=dense_group_records,
    )

    base = base_dir or Path.cwd()
    report = {
        "planner_version": PLANNER_VERSION,
        "planner_strategy": PLANNER_STRATEGY,
        "created_at": _utc_now(),
        "source_beats_path": _display_path(source_beats_path, base),
        "source_transcript_path": _display_path(source_transcript_path, base),
        "preview_beats_path": _display_path(preview_beats_path, base),
        "standard_beat_count": len(beats),
        "dense_preview_beat_count": len(preview_beats),
        "target_range_min": beat_config.dense_min_target,
        "target_range_max": beat_config.dense_max_target,
        "safe_to_apply": safe_to_apply,
        "warnings": warnings,
        "applied_splits": [],
        "dense_group_records": dense_group_records,
        "rejected_candidates": _cap_rejected_candidates(rejected_examples),
        "summary": {
            "standard_average_duration_seconds": standard_stats["average_duration_seconds"],
            "dense_average_duration_seconds": dense_stats["average_duration_seconds"],
            "standard_longest_duration_seconds": standard_stats["longest_duration_seconds"],
            "dense_longest_duration_seconds": dense_stats["longest_duration_seconds"],
            "target_range_reached": beat_config.dense_min_target <= len(preview_beats) <= beat_config.dense_max_target,
        },
    }
    return preview_beats, report


def print_dense_plan_summary(result: DenseBeatPlanResult, base_dir: Path) -> None:
    report = result.report
    summary = report["summary"]
    print("Dense beat planning complete")
    print(f"Planner strategy: {report['planner_strategy']}")
    print(f"Standard beats: {report['standard_beat_count']}")
    print(f"Dense preview beats: {report['dense_preview_beat_count']}")
    print(
        "Average beat duration: "
        f"{summary['standard_average_duration_seconds']:.3f}s -> {summary['dense_average_duration_seconds']:.3f}s"
    )
    print(f"Target range reached: {summary['target_range_reached']}")
    print(f"Safe to apply: {report['safe_to_apply']}")
    print("Reports written:")
    print(f"- {_display_path(result.json_path, base_dir)}")
    print(f"- {_display_path(result.markdown_path, base_dir)}")
    print(f"- {_display_path(result.preview_path, base_dir)}")


def _build_dense_pending_beats(
    segments: list[TranscriptSegment],
    beat_config: BeatConfig,
) -> tuple[list[PendingBeat], list[dict[str, Any]]]:
    pending: list[PendingBeat] = []
    rejected: list[dict[str, Any]] = []
    first_start = segments[0].start_seconds
    absorb_intro = 0.0 < first_start < beat_config.min_intro_beat_duration
    if first_start >= beat_config.min_intro_beat_duration:
        pending.append(
            PendingBeat(
                beat_type="intro",
                start_seconds=0.0,
                end_seconds=first_start,
                text_preview="Intro / silence",
                segment_indexes=[],
                reason="dense_rebuild_intro",
                boundary_confidence="not_applicable",
                score=0.0,
            )
        )

    current: list[TranscriptSegment] = []
    current_visual_start: float | None = 0.0 if absorb_intro else None
    previous_end = 0.0 if absorb_intro else first_start

    for segment in segments:
        if current:
            gap = segment.start_seconds - current[-1].end_seconds
            if gap >= beat_config.min_gap_beat_duration:
                pending.append(
                    _normal_pending(
                        current,
                        current_visual_start,
                        beat_config,
                        end_seconds=None,
                        reason="dense_rebuild_gap_flush",
                    )
                )
                pending.append(
                    PendingBeat(
                        beat_type="gap",
                        start_seconds=current[-1].end_seconds,
                        end_seconds=segment.start_seconds,
                        text_preview="Pause / silence",
                        segment_indexes=[],
                        reason="dense_rebuild_gap",
                        boundary_confidence="not_applicable",
                        score=0.0,
                    )
                )
                current = [segment]
                current_visual_start = segment.start_seconds
                previous_end = segment.end_seconds
                continue

            proposed_start = current_visual_start if current_visual_start is not None else current[0].start_seconds
            current_duration = current[-1].end_seconds - proposed_start
            should_flush = _should_flush_dense_group(
                current=current,
                next_segment=segment,
                visual_start=proposed_start,
                beat_config=beat_config,
            )
            if should_flush:
                normal_start = current_visual_start if current_visual_start is not None else current[0].start_seconds
                normal_end = _normal_flush_end(
                    current=current,
                    next_segment=segment,
                    visual_start=normal_start,
                    gap=gap,
                    beat_config=beat_config,
                )
                pending.append(
                    _normal_pending(
                        current,
                        current_visual_start,
                        beat_config,
                        end_seconds=normal_end,
                        reason=_flush_reason(current_duration, beat_config),
                    )
                )
                if gap > EPSILON:
                    current_visual_start = segment.start_seconds
                else:
                    current_visual_start = segment.start_seconds
                current = [segment]
            else:
                current.append(segment)
        else:
            gap = segment.start_seconds - previous_end
            if gap >= beat_config.min_gap_beat_duration:
                pending.append(
                    PendingBeat(
                        beat_type="gap",
                        start_seconds=previous_end,
                        end_seconds=segment.start_seconds,
                        text_preview="Pause / silence",
                        segment_indexes=[],
                        reason="dense_rebuild_gap",
                        boundary_confidence="not_applicable",
                        score=0.0,
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
        pending.append(
            _normal_pending(
                current,
                current_visual_start,
                beat_config,
                end_seconds=None,
                reason="dense_rebuild_final_tail",
            )
        )

    pending = _rebalance_fragment_windows(pending, {segment.index: segment for segment in segments}, beat_config)
    _record_short_artifact_candidates(pending, beat_config, rejected)
    return pending, rejected


def _normal_pending(
    segments: list[TranscriptSegment],
    start_seconds: float | None,
    beat_config: BeatConfig,
    end_seconds: float | None,
    reason: str,
) -> PendingBeat:
    visual_start = segments[0].start_seconds if start_seconds is None else start_seconds
    visual_end = segments[-1].end_seconds if end_seconds is None else end_seconds
    text = _preview(_clean_preview_text(" ".join(segment.text for segment in segments)), beat_config.max_preview_chars)
    confidence = _boundary_confidence(segments[-1].text)
    duration = visual_end - visual_start
    return PendingBeat(
        beat_type="normal",
        start_seconds=visual_start,
        end_seconds=visual_end,
        text_preview=text,
        segment_indexes=[segment.index for segment in segments],
        reason=reason,
        boundary_confidence=confidence,
        score=_group_score(duration, confidence, beat_config),
    )


def _should_flush_dense_group(
    current: list[TranscriptSegment],
    next_segment: TranscriptSegment,
    visual_start: float,
    beat_config: BeatConfig,
) -> bool:
    current_duration = current[-1].end_seconds - visual_start
    proposed_duration = next_segment.end_seconds - visual_start
    if current_duration + EPSILON < beat_config.dense_min_duration:
        return False
    if proposed_duration > beat_config.dense_hard_max_duration + EPSILON:
        return True

    current_boundary = _boundary_confidence(current[-1].text)
    next_boundary = _boundary_confidence(next_segment.text)
    current_promptable = _has_promptable_exit(current[-1].text)
    if current_promptable and next_boundary != "high":
        return True
    if proposed_duration <= beat_config.dense_soft_max_duration + EPSILON:
        return False
    boundary_extension_limit = min(
        beat_config.dense_hard_max_duration,
        beat_config.dense_soft_max_duration + DENSE_BOUNDARY_EXTENSION_SECONDS,
    )
    if proposed_duration > boundary_extension_limit + EPSILON:
        return True

    if current_boundary != "high" and next_boundary == "high":
        return False
    return True


def _normal_flush_end(
    current: list[TranscriptSegment],
    next_segment: TranscriptSegment,
    visual_start: float,
    gap: float,
    beat_config: BeatConfig,
) -> float:
    if gap <= EPSILON:
        return current[-1].end_seconds
    gap_extended_end = next_segment.start_seconds
    if gap_extended_end - visual_start > beat_config.dense_hard_max_duration + EPSILON:
        return current[-1].end_seconds
    return gap_extended_end


def _has_promptable_exit(text: str) -> bool:
    return _boundary_confidence(text) == "high" and not _ends_with_unfinished_word(text)


def _ends_with_unfinished_word(text: str) -> bool:
    words = _meaningful_words(text)
    if not words:
        return False
    return words[-1].lower() in UNFINISHED_END_WORDS


def _starts_with_continuation_fragment(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    match = re.search(r"[A-Za-z][A-Za-z']*", stripped)
    if not match:
        return False
    word = match.group(0)
    lowered = word.lower()
    return lowered in CONTINUATION_WORDS or word[0].islower()


def _rebalance_fragment_windows(
    pending: list[PendingBeat],
    segment_lookup: dict[int, TranscriptSegment],
    beat_config: BeatConfig,
) -> list[PendingBeat]:
    items = list(pending)
    start = 0
    while start < len(items):
        replacement = _fragment_window_replacement_at(items, start, segment_lookup, beat_config)
        if replacement is None:
            start += 1
            continue
        end, optimized = replacement
        items = items[:start] + optimized + items[end:]
        start += len(optimized)
    return items


def _fragment_window_replacement_at(
    pending: list[PendingBeat],
    start: int,
    segment_lookup: dict[int, TranscriptSegment],
    beat_config: BeatConfig,
) -> tuple[int, list[PendingBeat]] | None:
    if pending[start].beat_type != "normal":
        return None
    max_end = min(len(pending), start + LOCAL_REBALANCE_MAX_WINDOW)
    best: tuple[float, int, list[PendingBeat]] | None = None
    for end in range(start + 3, max_end + 1):
        window = pending[start:end]
        if any(item.beat_type != "normal" for item in window):
            break
        optimized, improvement = _optimize_fragment_window(window, segment_lookup, beat_config)
        if optimized is None or improvement < LOCAL_REBALANCE_MIN_IMPROVEMENT:
            continue
        if best is None or improvement > best[0] + EPSILON:
            best = (improvement, end, optimized)
    if best is None:
        return None
    _, end, optimized = best
    return end, optimized


def _optimize_fragment_window(
    window: list[PendingBeat],
    segment_lookup: dict[int, TranscriptSegment],
    beat_config: BeatConfig,
) -> tuple[list[PendingBeat] | None, float]:
    if not _has_fragment_window_issue(window, segment_lookup):
        return None, 0.0
    segment_indexes = [index for item in window for index in item.segment_indexes]
    if len(segment_indexes) <= len(window):
        return None, 0.0

    target_count = len(window)
    current_score = sum(_pending_quality_score(item, segment_lookup, beat_config) for item in window)
    states: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {(0, 0): (0.0, [])}
    total_segments = len(segment_indexes)
    for group_count in range(target_count):
        next_states: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {}
        for (position, count), (score, groups) in states.items():
            if count != group_count:
                continue
            remaining_groups = target_count - group_count - 1
            for end_position in range(position + 1, total_segments - remaining_groups + 1):
                group = segment_indexes[position:end_position]
                group_start = _window_group_start(window, segment_indexes, position, segment_lookup)
                group_end = _window_group_end(window, segment_indexes, end_position, segment_lookup)
                group_score = _segment_group_quality_score(
                    group,
                    group_start,
                    group_end,
                    segment_lookup,
                    beat_config,
                    allow_continuation_start=position == 0 and _pending_starts_with_continuation(window[0], segment_lookup),
                )
                if group_score is None:
                    continue
                key = (end_position, group_count + 1)
                candidate = (score + group_score, groups + [(position, end_position)])
                if key not in next_states or candidate[0] < next_states[key][0]:
                    next_states[key] = candidate
        states.update(next_states)

    best = states.get((total_segments, target_count))
    if best is None:
        return None, 0.0
    best_score, groups = best
    improvement = current_score - best_score
    if improvement <= EPSILON:
        return None, 0.0

    optimized = []
    for position, end_position in groups:
        indexes = segment_indexes[position:end_position]
        optimized.append(
            _normal_pending_from_indexes(
                indexes=indexes,
                start_seconds=_window_group_start(window, segment_indexes, position, segment_lookup),
                end_seconds=_window_group_end(window, segment_indexes, end_position, segment_lookup),
                segment_lookup=segment_lookup,
                beat_config=beat_config,
                reason="dense_rebuild_local_rebalance",
            )
        )
    return optimized, improvement


def _has_fragment_window_issue(window: list[PendingBeat], segment_lookup: dict[int, TranscriptSegment]) -> bool:
    for left, right in zip(window, window[1:]):
        if _pending_ends_unfinished(left, segment_lookup) or _pending_starts_with_continuation(right, segment_lookup):
            return True
        if _pending_is_garbled_or_thin(left, segment_lookup) or _pending_is_garbled_or_thin(right, segment_lookup):
            return True
    return False


def _pending_quality_score(
    item: PendingBeat,
    segment_lookup: dict[int, TranscriptSegment],
    beat_config: BeatConfig,
) -> float:
    return _segment_group_quality_score(
        item.segment_indexes,
        item.start_seconds,
        item.end_seconds,
        segment_lookup,
        beat_config,
    ) or math.inf


def _segment_group_quality_score(
    indexes: list[int],
    start_seconds: float,
    end_seconds: float,
    segment_lookup: dict[int, TranscriptSegment],
    beat_config: BeatConfig,
    allow_continuation_start: bool = True,
) -> float | None:
    duration = end_seconds - start_seconds
    if duration + EPSILON < beat_config.dense_min_duration:
        return None
    if duration > beat_config.dense_hard_max_duration + EPSILON:
        return None

    text = _source_text_for_indexes(indexes, segment_lookup)
    words = _meaningful_words(text)
    first_segment = segment_lookup[indexes[0]]
    last_segment = segment_lookup[indexes[-1]]
    score = abs(duration - beat_config.dense_preferred_duration)
    boundary = _boundary_confidence(last_segment.text)
    if boundary == "medium":
        score += 10.0
    elif boundary == "weak":
        score += 24.0
    if _ends_with_unfinished_word(last_segment.text):
        score += 18.0
    starts_with_continuation = _starts_with_continuation_fragment(first_segment.text)
    if starts_with_continuation and not allow_continuation_start:
        return None
    if starts_with_continuation:
        score += 24.0
    if _looks_garbled_or_thin(text):
        score += 24.0
    if len(words) < 6:
        score += 20.0
    return score


def _normal_pending_from_indexes(
    indexes: list[int],
    start_seconds: float,
    end_seconds: float,
    segment_lookup: dict[int, TranscriptSegment],
    beat_config: BeatConfig,
    reason: str,
) -> PendingBeat:
    text = _source_text_for_indexes(indexes, segment_lookup)
    confidence = _boundary_confidence(segment_lookup[indexes[-1]].text)
    return PendingBeat(
        beat_type="normal",
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        text_preview=_preview(_clean_preview_text(text), beat_config.max_preview_chars),
        segment_indexes=list(indexes),
        reason=reason,
        boundary_confidence=confidence,
        score=_group_score(end_seconds - start_seconds, confidence, beat_config),
    )


def _window_group_start(
    window: list[PendingBeat],
    segment_indexes: list[int],
    position: int,
    segment_lookup: dict[int, TranscriptSegment],
) -> float:
    if position == 0:
        return window[0].start_seconds
    return segment_lookup[segment_indexes[position]].start_seconds


def _window_group_end(
    window: list[PendingBeat],
    segment_indexes: list[int],
    end_position: int,
    segment_lookup: dict[int, TranscriptSegment],
) -> float:
    if end_position == len(segment_indexes):
        return window[-1].end_seconds
    return segment_lookup[segment_indexes[end_position]].start_seconds


def _source_text_for_indexes(indexes: list[int], segment_lookup: dict[int, TranscriptSegment]) -> str:
    return " ".join(segment_lookup[index].text for index in indexes if index in segment_lookup)


def _pending_ends_unfinished(item: PendingBeat, segment_lookup: dict[int, TranscriptSegment]) -> bool:
    if not item.segment_indexes:
        return False
    segment = segment_lookup[item.segment_indexes[-1]]
    return _boundary_confidence(segment.text) == "weak" or _ends_with_unfinished_word(segment.text)


def _pending_starts_with_continuation(item: PendingBeat, segment_lookup: dict[int, TranscriptSegment]) -> bool:
    if not item.segment_indexes:
        return False
    return _starts_with_continuation_fragment(segment_lookup[item.segment_indexes[0]].text)


def _pending_is_garbled_or_thin(item: PendingBeat, segment_lookup: dict[int, TranscriptSegment]) -> bool:
    return _looks_garbled_or_thin(_source_text_for_indexes(item.segment_indexes, segment_lookup))


def _looks_garbled_or_thin(text: str) -> bool:
    words = _meaningful_words(text)
    digit_count = sum(character.isdigit() for character in text)
    alpha_count = sum(character.isalpha() for character in text)
    return len(words) < 6 or (digit_count >= 6 and digit_count > alpha_count * 0.25)


def _flush_reason(current_duration: float, beat_config: BeatConfig) -> str:
    distance_to_preferred = abs(current_duration - beat_config.dense_preferred_duration)
    if current_duration >= beat_config.dense_min_duration and distance_to_preferred <= 1.5:
        return "dense_rebuild_preferred_boundary"
    return "dense_rebuild_soft_max_boundary"


def _group_score(duration: float, boundary_confidence: str, beat_config: BeatConfig) -> float:
    score = abs(duration - beat_config.dense_preferred_duration)
    if duration > beat_config.dense_soft_max_duration:
        score += (duration - beat_config.dense_soft_max_duration) * 4
    if duration > beat_config.dense_hard_max_duration:
        score += (duration - beat_config.dense_hard_max_duration) * 20
    if duration < beat_config.dense_min_duration:
        score += (beat_config.dense_min_duration - duration) * 12
    if boundary_confidence == "medium":
        score += 1.0
    elif boundary_confidence == "weak":
        score += 4.0
    return round(score, 3)


def _record_short_artifact_candidates(
    pending: list[PendingBeat],
    beat_config: BeatConfig,
    rejected: list[dict[str, Any]],
) -> None:
    for item in pending:
        if (
            item.beat_type == "normal"
            and len(item.segment_indexes) <= 1
            and item.duration_seconds > beat_config.dense_hard_max_duration
        ):
            rejected.append(
                {
                    "reason": "unsplittable_segment",
                    "dense_reason": item.reason,
                    "start_seconds": round(item.start_seconds, 3),
                    "end_seconds": round(item.end_seconds, 3),
                    "duration_seconds": round(item.duration_seconds, 3),
                    "detail": "Dense normal group exceeds hard max but contains one transcript segment.",
                }
            )
        if item.beat_type != "normal" or item.duration_seconds + EPSILON >= beat_config.dense_min_duration:
            continue
        rejected.append(
            {
                "reason": "under_minimum_duration",
                "dense_reason": item.reason,
                "start_seconds": round(item.start_seconds, 3),
                "end_seconds": round(item.end_seconds, 3),
                "duration_seconds": round(item.duration_seconds, 3),
                "detail": (
                    f"Dense normal group is {item.duration_seconds:.3f}s, "
                    f"below dense minimum {beat_config.dense_min_duration:.3f}s."
                ),
            }
        )


def _finalize_pending_beats(pending: list[PendingBeat], images_dir: Path) -> list[Beat]:
    beats = []
    for index, item in enumerate(pending, start=1):
        if item.end_seconds <= item.start_seconds:
            raise PipelineError(f"Invalid dense beat duration: {item.start_seconds:.3f}s to {item.end_seconds:.3f}s.")
        image_path = images_dir / f"beat_{index:03}.png"
        beats.append(
            Beat(
                beat_number=index,
                beat_type=item.beat_type,
                start=seconds_to_timestamp(item.start_seconds),
                end=seconds_to_timestamp(item.end_seconds),
                start_seconds=item.start_seconds,
                end_seconds=item.end_seconds,
                duration_seconds=item.duration_seconds,
                text_preview=item.text_preview,
                segment_indexes=list(item.segment_indexes),
                image_path=image_path.as_posix(),
            )
        )
    return beats


def _standard_beat_lookup(beats: list[Beat]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for beat in beats:
        for index in beat.segment_indexes:
            lookup[index] = beat.beat_number
    return lookup


def _dense_group_records(
    preview_beats: list[Beat],
    pending: list[PendingBeat],
    standard_lookup: dict[int, int],
) -> list[dict[str, Any]]:
    records = []
    for beat, item in zip(preview_beats, pending, strict=True):
        source_numbers = sorted({standard_lookup[index] for index in item.segment_indexes if index in standard_lookup})
        source_standard_beat_number = source_numbers[0] if len(source_numbers) == 1 else None
        records.append(
            {
                "dense_beat_number": beat.beat_number,
                "beat_type": beat.beat_type,
                "source_standard_beat_number": source_standard_beat_number,
                "start_seconds": round(beat.start_seconds, 3),
                "end_seconds": round(beat.end_seconds, 3),
                "duration_seconds": round(beat.duration_seconds, 3),
                "segment_indexes": list(beat.segment_indexes),
                "grouping_score": item.score,
                "boundary_confidence": item.boundary_confidence,
                "reason": item.reason,
            }
        )
    return records


def _warnings_and_safety(
    preview_beats: list[Beat],
    segments: list[TranscriptSegment],
    beat_config: BeatConfig,
    dense_group_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    segment_lookup = {segment.index: segment for segment in segments}
    warnings = []
    normal_beats = [beat for beat in preview_beats if beat.beat_type == "normal"]
    safe_to_apply = True

    if len(preview_beats) < beat_config.dense_min_target:
        warnings.append(
            _warning(
                "TARGET_NOT_REACHED_SAFELY",
                f"Dense preview reached {len(preview_beats)} beats, below target minimum {beat_config.dense_min_target}.",
            )
        )
        safe_to_apply = False
    if len(preview_beats) > beat_config.dense_max_target:
        warnings.append(
            _warning(
                "TARGET_MAX_EXCEEDED",
                f"Dense preview reached {len(preview_beats)} beats, above target maximum {beat_config.dense_max_target}.",
            )
        )
        safe_to_apply = False

    micro_beats, unavoidable_short = _short_normal_beats(normal_beats, dense_group_records, beat_config)
    if micro_beats:
        warnings.append(
            _warning(
                "MICRO_BEATS_BELOW_MINIMUM",
                f"{len(micro_beats)} proposed dense normal beats are below dense minimum duration.",
                len(micro_beats),
                _beat_examples(micro_beats),
            )
        )
        safe_to_apply = False
    if unavoidable_short:
        warnings.append(
            _warning(
                "UNAVOIDABLE_SHORT_FINAL_TAIL",
                f"{len(unavoidable_short)} dense normal final-tail beats are below minimum and explicitly reported.",
                len(unavoidable_short),
                _beat_examples(unavoidable_short),
            )
        )

    over_hard = [beat for beat in normal_beats if beat.duration_seconds > beat_config.dense_hard_max_duration]
    if over_hard:
        warnings.append(
            _warning(
                "DENSE_NORMAL_BEATS_ABOVE_HARD_MAX",
                f"{len(over_hard)} dense normal beats exceed dense hard max.",
                len(over_hard),
                _beat_examples(over_hard),
            )
        )
        safe_to_apply = False

    source_warnings = _source_text_warnings(normal_beats, segment_lookup)
    warnings.extend(source_warnings["warnings"])
    if source_warnings["no_meaningful_count"] > 0:
        safe_to_apply = False

    awkward_count = source_warnings["awkward_start_count"]
    awkward_threshold = max(5, math.floor(len(normal_beats) * 0.05) + 1)
    if awkward_count >= awkward_threshold:
        safe_to_apply = False

    weak_count = source_warnings["weak_boundary_count"]
    if normal_beats and (weak_count >= 18 and weak_count / len(normal_beats) > 0.25):
        warnings.append(
            _warning(
                "TOO_MANY_WEAK_BOUNDARIES",
                f"{weak_count} dense normal beats have weak punctuation boundaries.",
                weak_count,
                _beat_examples(source_warnings["weak_boundary_beats"]),
            )
        )
        safe_to_apply = False

    return warnings, safe_to_apply


def _short_normal_beats(
    normal_beats: list[Beat],
    dense_group_records: list[dict[str, Any]],
    beat_config: BeatConfig,
) -> tuple[list[Beat], list[Beat]]:
    record_by_number = {record["dense_beat_number"]: record for record in dense_group_records}
    micro = []
    unavoidable = []
    for beat in normal_beats:
        if beat.duration_seconds + EPSILON >= beat_config.dense_min_duration:
            continue
        record = record_by_number.get(beat.beat_number, {})
        if record.get("reason") == "dense_rebuild_final_tail":
            unavoidable.append(beat)
        else:
            micro.append(beat)
    return micro, unavoidable


def _source_text_warnings(
    beats: list[Beat],
    segment_lookup: dict[int, TranscriptSegment],
) -> dict[str, Any]:
    no_meaningful = []
    very_short = []
    awkward = []
    mid_thought = []
    weak_boundary = []

    for beat in beats:
        source_text = _source_text(beat, segment_lookup)
        words = _meaningful_words(source_text)
        compact = " ".join(source_text.split())
        if not words:
            no_meaningful.append(beat)
            continue
        if len(words) < 6:
            very_short.append(beat)
        if words[0].lower() in CONTINUATION_WORDS:
            awkward.append(beat)
        if compact and compact[-1] not in TERMINAL_PUNCTUATION:
            mid_thought.append(beat)
        if beat.segment_indexes:
            last_segment = segment_lookup.get(beat.segment_indexes[-1])
            if last_segment and _boundary_confidence(last_segment.text) != "high":
                weak_boundary.append(beat)

    warnings = []
    if no_meaningful:
        warnings.append(
            _warning(
                "NO_MEANINGFUL_SOURCE_TEXT",
                f"{len(no_meaningful)} proposed beats have no meaningful source text.",
                len(no_meaningful),
                _beat_examples(no_meaningful),
            )
        )
    if very_short:
        warnings.append(
            _warning(
                "VERY_SHORT_SOURCE_TEXT",
                f"{len(very_short)} proposed beats have fewer than 6 meaningful words.",
                len(very_short),
                _beat_examples(very_short),
            )
        )
    if awkward:
        warnings.append(
            _warning(
                "AWKWARD_CONTINUATION_START",
                f"{len(awkward)} proposed beats start with awkward continuation words.",
                len(awkward),
                _beat_examples(awkward),
            )
        )
    if mid_thought:
        warnings.append(
            _warning(
                "ENDS_MID_THOUGHT",
                f"{len(mid_thought)} proposed beats appear to end mid-thought.",
                len(mid_thought),
                _beat_examples(mid_thought),
            )
        )
    if weak_boundary:
        warnings.append(
            _warning(
                "WEAK_PUNCTUATION_BOUNDARY",
                f"{len(weak_boundary)} proposed beats end on weak punctuation boundaries.",
                len(weak_boundary),
                _beat_examples(weak_boundary),
            )
        )

    return {
        "warnings": warnings,
        "no_meaningful_count": len(no_meaningful),
        "awkward_start_count": len(awkward),
        "weak_boundary_count": len(weak_boundary),
        "weak_boundary_beats": weak_boundary,
    }


def _source_text(beat: Beat, segment_lookup: dict[int, TranscriptSegment]) -> str:
    if beat.segment_indexes:
        return " ".join(
            segment_lookup[index].text.strip()
            for index in beat.segment_indexes
            if index in segment_lookup and segment_lookup[index].text.strip()
        )
    return beat.text_preview


def _boundary_confidence(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "weak"
    last = stripped[-1]
    if last in TERMINAL_PUNCTUATION:
        return "high"
    if last in WEAK_BOUNDARY_CHARS:
        return "medium"
    return "weak"


def _meaningful_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z']*", text)


def _warning(
    code: str,
    message: str,
    count: int | None = None,
    examples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "message": message}
    if count is not None:
        item["count"] = count
    if examples:
        item["examples"] = examples[:5]
    return item


def _beat_examples(beats: list[Beat]) -> list[dict[str, Any]]:
    return [
        {
            "beat_number": beat.beat_number,
            "start_seconds": round(beat.start_seconds, 3),
            "end_seconds": round(beat.end_seconds, 3),
            "duration_seconds": round(beat.duration_seconds, 3),
            "source_preview": _preview(beat.text_preview, 100),
        }
        for beat in beats[:5]
    ]


def _cap_rejected_candidates(rejected: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {reason: [] for reason in REJECTION_REASON_ORDER}
    seen = set()
    for item in sorted(
        rejected,
        key=lambda value: (
            REJECTION_REASON_ORDER.index(value["reason"])
            if value.get("reason") in REJECTION_REASON_ORDER
            else len(REJECTION_REASON_ORDER),
            value.get("start_seconds", 0),
            value.get("end_seconds", 0),
            value.get("detail", ""),
        ),
    ):
        reason = item.get("reason", "low_quality_split")
        key = (reason, item.get("start_seconds"), item.get("end_seconds"), item.get("detail"))
        if key in seen:
            continue
        seen.add(key)
        if reason not in grouped:
            grouped[reason] = []
        if sum(len(entries) for entries in grouped.values()) >= MAX_REJECTED_EXAMPLES:
            break
        grouped[reason].append(item)
    return {reason: entries for reason, entries in grouped.items() if entries}


def _duration_stats(beats: list[Beat]) -> dict[str, float]:
    durations = [beat.duration_seconds for beat in beats]
    return {
        "average_duration_seconds": round(sum(durations) / len(durations), 3) if durations else 0.0,
        "longest_duration_seconds": round(max(durations), 3) if durations else 0.0,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Dense Beat Plan",
        "",
        f"- Planner version: {report['planner_version']}",
        f"- Planner strategy: `{report['planner_strategy']}`",
        f"- Created at: {report['created_at']}",
        f"- Source beats: `{report['source_beats_path']}`",
        f"- Source transcript: `{report['source_transcript_path']}`",
        f"- Preview beats: `{report['preview_beats_path']}`",
        f"- Standard beats: {report['standard_beat_count']}",
        f"- Dense preview beats: {report['dense_preview_beat_count']}",
        f"- Target range: {report['target_range_min']}-{report['target_range_max']}",
        f"- Average duration: {summary['standard_average_duration_seconds']:.3f}s -> {summary['dense_average_duration_seconds']:.3f}s",
        f"- Target range reached: {summary['target_range_reached']}",
        f"- Safe to apply: {report['safe_to_apply']}",
        "",
        "## Strategy",
        "",
        "`hybrid_dense_rebuild` rebuilds dense normal beats from transcript segments while preserving standard intro and gap behavior where practical. It does not overwrite production beats.",
        "",
        "## Warnings",
        "",
    ]
    if report["warnings"]:
        lines.extend(f"- `{warning['code']}`: {warning['message']}" for warning in report["warnings"])
    else:
        lines.append("- None")

    lines.extend(["", "## Dense Group Records", ""])
    records = report.get("dense_group_records", [])
    if records:
        lines.append("| Dense Beat | Type | Source Standard Beat | Duration | Boundary | Reason |")
        lines.append("| ---: | --- | ---: | ---: | --- | --- |")
        for item in records[:30]:
            source = item["source_standard_beat_number"] if item["source_standard_beat_number"] is not None else "-"
            lines.append(
                f"| {item['dense_beat_number']} | {item['beat_type']} | {source} | "
                f"{item['duration_seconds']:.3f}s | {item['boundary_confidence']} | {item['reason']} |"
            )
        if len(records) > 30:
            lines.append(f"| ... | ... | ... | ... | ... | {len(records) - 30} more records omitted from Markdown preview |")
    else:
        lines.append("- None")

    lines.extend(["", "## Applied Splits", ""])
    if report["applied_splits"]:
        lines.append("| Parent | Split | Score | Reason |")
        lines.append("| ---: | ---: | ---: | --- |")
        for item in report["applied_splits"]:
            lines.append(
                f"| {item['parent_beat_number']} | {item['split_timestamp']:.3f} | "
                f"{item['score']:.3f} | {item['reason']} |"
            )
    else:
        lines.append("- None; this plan uses transcript-first hybrid dense rebuild.")

    rejected = report["rejected_candidates"]
    lines.extend(["", "## Rejected Candidates", ""])
    if rejected:
        for reason in REJECTION_REASON_ORDER:
            items = rejected.get(reason, [])
            if not items:
                continue
            lines.append(f"### {reason.replace('_', ' ').title()}")
            lines.append("")
            for item in items:
                lines.append(f"- {item['detail']}")
            lines.append("")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
