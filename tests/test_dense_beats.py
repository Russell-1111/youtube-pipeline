from pathlib import Path

from youtube_pipeline.config import BeatConfig, load_config, validate_config
from youtube_pipeline.dense_beats import PendingBeat, _rebalance_fragment_windows, build_dense_beat_plan
from youtube_pipeline.models import Beat, TranscriptSegment
from youtube_pipeline.time_utils import seconds_to_timestamp


BEAT_CONFIG = BeatConfig(
    min_duration=6,
    target_duration=9,
    max_duration=12,
    min_gap_beat_duration=1.5,
    min_intro_beat_duration=1.0,
    max_preview_chars=80,
)


def segment(index: int, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        index=index,
        start=seconds_to_timestamp(start),
        end=seconds_to_timestamp(end),
        start_seconds=start,
        end_seconds=end,
        duration_seconds=end - start,
        text=text,
    )


def beat(
    tmp_path: Path,
    number: int,
    start: float,
    end: float,
    indexes: list[int],
    beat_type: str = "normal",
    preview: str = "Preview sentence.",
) -> Beat:
    return Beat(
        beat_number=number,
        beat_type=beat_type,
        start=seconds_to_timestamp(start),
        end=seconds_to_timestamp(end),
        start_seconds=start,
        end_seconds=end,
        duration_seconds=end - start,
        text_preview=preview,
        segment_indexes=indexes,
        image_path=(tmp_path / "assets" / "images" / f"beat_{number:03}.png").as_posix(),
    )


def pending_item(indexes: list[int], start: float, end: float) -> PendingBeat:
    return PendingBeat(
        beat_type="normal",
        start_seconds=start,
        end_seconds=end,
        text_preview="Preview sentence.",
        segment_indexes=indexes,
        reason="test",
        boundary_confidence="high",
        score=0,
    )


def test_existing_config_files_get_dense_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
inputs:
  voiceover: input/voiceover.mp3
  transcript: input/transcript.srt

outputs:
  data_dir: data
  images_dir: assets/images
  contact_sheet: assets/contact_sheets/contact_sheet.png
  final_video: output/final_video.mp4

video:
  width: 1920
  height: 1080
  fps: 30

beats:
  min_duration: 6
  target_duration: 9
  max_duration: 12
  min_gap_beat_duration: 1.5
  min_intro_beat_duration: 1.0
  max_preview_chars: 80

timing:
  duration_mismatch_tolerance: 1.0
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    validate_config(config)

    assert config.beats.density_mode == "standard"
    assert config.beats.dense_min_target == 70
    assert config.beats.dense_max_target == 90
    assert config.beats.allow_dense_split is True


def test_dense_planner_reaches_target_range_when_boundaries_support_it(tmp_path):
    beats = []
    segments = []
    current = 0.0
    segment_index = 1
    for number in range(1, 73):
        indexes = []
        group_start = current
        for part in range(3):
            start = current
            end = current + 2.75
            segments.append(segment(segment_index, start, end, f"Complete idea {number}-{part}."))
            indexes.append(segment_index)
            segment_index += 1
            current = end
        if number <= 55:
            beats.append(beat(tmp_path, number, group_start, current, indexes))

    preview, report = build_dense_beat_plan(beats, segments, BEAT_CONFIG, tmp_path / "assets" / "images")

    assert 70 <= len(preview) <= 90
    assert report["planner_strategy"] == "hybrid_dense_rebuild"
    assert report["summary"]["target_range_reached"] is True
    assert 6.5 <= report["summary"]["dense_average_duration_seconds"] <= 9.0
    assert report["applied_splits"] == []
    assert len(report["dense_group_records"]) == len(preview)
    assert all(item.duration_seconds >= BEAT_CONFIG.dense_min_duration for item in preview if item.beat_type == "normal")


def test_dense_planner_does_not_split_inside_single_segment(tmp_path):
    beats = [beat(tmp_path, 1, 0, 14, [1], preview="One long caption.")]
    segments = [segment(1, 0, 14, "One long caption.")]

    preview, report = build_dense_beat_plan(beats, segments, BEAT_CONFIG, tmp_path / "assets" / "images")

    assert len(preview) == 1
    assert preview[0].segment_indexes == [1]
    assert "unsplittable_segment" in report["rejected_candidates"]
    assert any(warning["code"] == "DENSE_NORMAL_BEATS_ABOVE_HARD_MAX" for warning in report["warnings"])
    assert report["safe_to_apply"] is False


def test_dense_planner_waits_for_terminal_boundary_under_hard_max(tmp_path):
    beats = [beat(tmp_path, 1, 0, 17, [1, 2, 3, 4], preview="Combined caption.")]
    segments = [
        segment(1, 0.0, 3.0, "The first idea begins"),
        segment(2, 3.0, 6.4, "and keeps moving toward"),
        segment(3, 6.4, 9.7, "a complete sentence."),
        segment(4, 9.7, 17.0, "A second sentence follows."),
    ]

    preview, report = build_dense_beat_plan(beats, segments, BEAT_CONFIG, tmp_path / "assets" / "images")

    assert preview[0].segment_indexes == [1, 2, 3]
    assert preview[0].duration_seconds == 9.7
    assert report["dense_group_records"][0]["boundary_confidence"] == "high"


def test_dense_planner_splits_before_degrading_complete_boundary(tmp_path):
    beats = [beat(tmp_path, 1, 0, 14, [1, 2, 3, 4], preview="Combined caption.")]
    segments = [
        segment(1, 0.0, 3.2, "Railways needed common time."),
        segment(2, 3.2, 6.7, "Offices needed predictable hours."),
        segment(3, 6.7, 11.5, "Shipping networks all benefited"),
        segment(4, 11.5, 14.0, "when bodies synchronized."),
    ]

    preview, _ = build_dense_beat_plan(beats, segments, BEAT_CONFIG, tmp_path / "assets" / "images")

    assert [item.segment_indexes for item in preview] == [[1, 2], [3, 4]]
    assert all(item.duration_seconds >= BEAT_CONFIG.dense_min_duration for item in preview)
    assert all(item.duration_seconds <= BEAT_CONFIG.dense_hard_max_duration for item in preview)


def test_dense_planner_rebalances_fragment_pair_and_restores_density():
    segments = [
        segment(1, 0.0, 6.0, "The first sound is not a"),
        segment(2, 6.0, 9.0, "bird, it is a phone."),
        segment(3, 9.0, 15.1, "Another complete image appears."),
        segment(4, 15.1, 21.2, "A third complete image appears."),
        segment(5, 21.2, 27.3, "A final complete image appears."),
    ]
    pending = [
        pending_item([1], 0.0, 6.0),
        pending_item([2], 6.0, 9.0),
        pending_item([3], 9.0, 15.1),
        pending_item([4, 5], 15.1, 27.3),
    ]

    rebalanced = _rebalance_fragment_windows(pending, {item.index: item for item in segments}, BEAT_CONFIG)

    assert [item.segment_indexes for item in rebalanced] == [[1, 2], [3], [4], [5]]
    assert all(item.duration_seconds >= BEAT_CONFIG.dense_min_duration for item in rebalanced)


def test_dense_planner_does_not_rebalance_fragment_pair_above_hard_max():
    segments = [
        segment(1, 0.0, 7.0, "The first sound is not a"),
        segment(2, 7.0, 13.1, "bird, it is a phone."),
        segment(3, 13.1, 19.2, "Another complete image appears."),
    ]
    pending = [
        pending_item([1], 0.0, 7.0),
        pending_item([2], 7.0, 13.1),
        pending_item([3], 13.1, 19.2),
    ]

    rebalanced = _rebalance_fragment_windows(pending, {item.index: item for item in segments}, BEAT_CONFIG)

    assert [item.segment_indexes for item in rebalanced] == [[1], [2], [3]]


def test_dense_planner_rebalance_avoids_new_lowercase_continuation_starts():
    segments = [
        segment(1, 0.0, 6.0, "The first sound is not a"),
        segment(2, 6.0, 9.0, "bird, it is a phone."),
        segment(3, 9.0, 15.1, "Another complete image appears."),
        segment(4, 15.1, 21.2, "A third complete image appears."),
        segment(5, 21.2, 27.3, "A final complete image appears."),
    ]
    pending = [
        pending_item([1], 0.0, 6.0),
        pending_item([2], 6.0, 9.0),
        pending_item([3], 9.0, 15.1),
        pending_item([4, 5], 15.1, 27.3),
    ]

    rebalanced = _rebalance_fragment_windows(pending, {item.index: item for item in segments}, BEAT_CONFIG)

    assert all(not segments[item.segment_indexes[0] - 1].text[0].islower() for item in rebalanced)


def test_dense_planner_rebalances_numeric_fragment_with_coherent_text():
    segments = [
        segment(1, 0.0, 4.0, "903 11 58 215."),
        segment(2, 4.0, 7.0, "A grid of symbols with no"),
        segment(3, 7.0, 10.0, "body, no voice, and still it tightens."),
        segment(4, 10.0, 16.1, "Another complete image appears."),
        segment(5, 16.1, 22.2, "A final complete image appears."),
    ]
    pending = [
        pending_item([1, 2], 0.0, 7.0),
        pending_item([3], 7.0, 10.0),
        pending_item([4, 5], 10.0, 22.2),
    ]

    rebalanced = _rebalance_fragment_windows(pending, {item.index: item for item in segments}, BEAT_CONFIG)

    assert [item.segment_indexes for item in rebalanced] == [[1, 2, 3], [4], [5]]


def test_dense_planner_flushes_before_hard_max_even_for_terminal_boundary(tmp_path):
    beats = [beat(tmp_path, 1, 0, 12.4, [1, 2, 3], preview="Combined caption.")]
    segments = [
        segment(1, 0.0, 3.0, "The first idea begins"),
        segment(2, 3.0, 6.4, "and keeps moving toward"),
        segment(3, 6.4, 12.4, "a complete sentence."),
    ]

    preview, _ = build_dense_beat_plan(beats, segments, BEAT_CONFIG, tmp_path / "assets" / "images")

    assert preview[0].segment_indexes == [1, 2]
    assert preview[0].duration_seconds == 6.4
    assert preview[1].segment_indexes == [3]


def test_dense_planner_preserves_intro_and_gap_beats(tmp_path):
    beats = [
        beat(tmp_path, 1, 0, 2, [], "intro", "Intro / silence"),
        beat(tmp_path, 2, 2, 8, [1]),
        beat(tmp_path, 3, 8, 16, [], "gap", "Pause / silence"),
        beat(tmp_path, 4, 16, 22, [2]),
    ]
    segments = [
        segment(1, 2, 8, "First complete idea."),
        segment(2, 16, 22, "Second complete idea."),
    ]

    preview, _ = build_dense_beat_plan(beats, segments, BEAT_CONFIG, tmp_path / "assets" / "images")

    assert [item.beat_type for item in preview] == ["intro", "normal", "gap", "normal"]
    assert preview[0].segment_indexes == []
    assert preview[2].segment_indexes == []
    assert preview[2].start_seconds == 8
    assert preview[2].end_seconds == 16


def test_dense_planner_caps_rejected_candidate_examples(tmp_path):
    beats = [beat(tmp_path, 1, 0, 20, list(range(1, 21)))]
    segments = [segment(index, index - 1, index, f"Tiny part {index}.") for index in range(1, 21)]

    _, report = build_dense_beat_plan(beats, segments, BEAT_CONFIG, tmp_path / "assets" / "images")

    rejected_count = sum(len(items) for items in report["rejected_candidates"].values())
    assert rejected_count <= 20
    assert all(len(items) <= 20 for items in report["rejected_candidates"].values())


def test_source_text_safety_can_make_plan_unsafe(tmp_path):
    beats = [beat(tmp_path, 1, 0, 12, [1, 2], preview="123 456")]
    segments = [
        segment(1, 0, 6, "123"),
        segment(2, 6, 12, "456"),
    ]

    _, report = build_dense_beat_plan(beats, segments, BEAT_CONFIG, tmp_path / "assets" / "images")

    assert any(warning["code"] == "NO_MEANINGFUL_SOURCE_TEXT" for warning in report["warnings"])
    assert report["safe_to_apply"] is False


def test_local_style_dense_rebuild_reaches_target_without_lowering_minimum(tmp_path):
    beats = []
    segments = []
    current = 0.0
    segment_index = 1
    for number in range(1, 73):
        indexes = []
        group_start = current
        for part in range(3):
            start = current
            end = current + 2.74
            segments.append(segment(segment_index, start, end, f"Local style complete idea {number}-{part}."))
            indexes.append(segment_index)
            segment_index += 1
            current = end
        if number <= 55:
            beats.append(beat(tmp_path, number, group_start, current, indexes))

    preview, report = build_dense_beat_plan(beats, segments, BEAT_CONFIG, tmp_path / "assets" / "images")

    assert BEAT_CONFIG.dense_min_duration == 6.0
    assert 70 <= len(preview) <= 90
    assert report["summary"]["target_range_reached"] is True
    assert all(item.duration_seconds >= BEAT_CONFIG.dense_min_duration for item in preview if item.beat_type == "normal")
    assert all(item.duration_seconds <= BEAT_CONFIG.dense_hard_max_duration for item in preview if item.beat_type == "normal")
