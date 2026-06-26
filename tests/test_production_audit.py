import json
from pathlib import Path

from PIL import Image

from youtube_pipeline.__main__ import main
from youtube_pipeline.config import load_config
from youtube_pipeline.manifest import write_beats, write_transcript_segments
from youtube_pipeline.models import Beat, TranscriptSegment
from youtube_pipeline.production_audit import run_production_audit


def write_config(path: Path) -> None:
    path.write_text(
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


def beat(tmp_path: Path, number: int, start_seconds: float, end_seconds: float, preview: str) -> Beat:
    return Beat(
        beat_number=number,
        beat_type="normal",
        start=f"00:00:{int(start_seconds):02},000",
        end=f"00:00:{int(end_seconds):02},000",
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        duration_seconds=end_seconds - start_seconds,
        text_preview=preview,
        segment_indexes=[number],
        image_path=(tmp_path / "assets" / "images" / f"beat_{number:03}.png").as_posix(),
    )


def segment(index: int, start_seconds: float, end_seconds: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        index=index,
        start=f"00:00:{int(start_seconds):02},000",
        end=f"00:00:{int(end_seconds):02},000",
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        duration_seconds=end_seconds - start_seconds,
        text=text,
    )


def write_prompt_payload(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "style_profile": "melancholic_minimalist_2d_explainer",
                "prompts": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def prompt_record(number: int, text: str, keywords: list[str]) -> dict:
    return {
        "beat_number": number,
        "source_text": text,
        "visual_concept_text": "concept" if keywords else "quiet symbolic scene drawn from the emotional tone",
        "visual_concept_keywords": keywords,
    }


def write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1920, 1080), color).save(path)


def audit(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    return run_production_audit(load_config(config_path), tmp_path)


def finding_codes(report: dict) -> list[str]:
    return [finding["code"] for finding in report["findings"]]


def visual_density_codes(report: dict) -> list[str]:
    return [finding["code"] for finding in report["sections"]["visual_density"]["findings"]]


def test_production_audit_writes_deterministic_json_and_markdown(tmp_path):
    write_beats(tmp_path / "data" / "beats.json", [beat(tmp_path, 1, 0, 6, "Complete sentence.")])
    write_transcript_segments(tmp_path / "data" / "transcript_segments.json", [segment(1, 0, 6, "Complete sentence.")])
    write_prompt_payload(tmp_path / "data" / "image_prompts.json", [prompt_record(1, "Complete sentence.", ["deadline"])])
    write_png(tmp_path / "assets" / "generated_images" / "beat_001.png", (80, 80, 80))

    first = audit(tmp_path)
    first_json = first.json_path.read_text(encoding="utf-8")
    second = audit(tmp_path)
    second_json = second.json_path.read_text(encoding="utf-8")

    report = first.report
    assert list(report) == [
        "audit_schema_version",
        "readiness_score",
        "readiness_label",
        "summary",
        "sections",
        "findings",
        "recommendations",
        "score_reasons",
    ]
    assert set(report["sections"]) == {
        "pacing",
        "visual_density",
        "generated_images",
        "prompts",
        "metaphors",
        "brightness",
        "render_risk",
        "outputs",
        "artifact_safety",
        "source_text",
    }
    assert first_json == second_json
    assert first.json_path.exists()
    assert first.markdown_path.exists()
    assert "# Production Audit Report" in first.markdown_path.read_text(encoding="utf-8")
    assert "## Visual Density" in first.markdown_path.read_text(encoding="utf-8")


def test_findings_have_required_shape_and_missing_optional_files_warn(tmp_path):
    write_beats(tmp_path / "data" / "beats.json", [beat(tmp_path, 1, 0, 6, "Complete sentence.")])

    result = audit(tmp_path)
    codes = finding_codes(result.report)

    assert "MISSING_IMAGE_PROMPTS_JSON" in codes
    assert "MISSING_TRANSCRIPT_SEGMENTS_JSON" in codes
    assert "MISSING_GENERATED_IMAGE" in codes
    assert result.report["readiness_label"] != "blocked"
    for finding in result.report["findings"]:
        assert {"severity", "code", "message"} <= set(finding)
        assert finding["severity"] in {"error", "warning", "info", "recommendation"}


def test_missing_beats_is_core_error_and_cli_returns_nonzero(tmp_path, capsys):
    write_config(tmp_path / "config.yaml")

    result = main(["--config", str(tmp_path / "config.yaml"), "--production-audit"])

    captured = capsys.readouterr()
    report = json.loads((tmp_path / "data" / "production_audit_report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert "Production audit complete" in captured.out
    assert report["readiness_label"] == "blocked"
    assert finding_codes(report) == ["MISSING_BEATS_JSON", "MISSING_IMAGE_PROMPTS_JSON", "MISSING_TRANSCRIPT_SEGMENTS_JSON"]


def test_invalid_beats_is_core_error(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "beats.json").write_text("{not json", encoding="utf-8")

    result = audit(tmp_path)

    assert result.has_core_errors
    assert "INVALID_BEATS_JSON" in finding_codes(result.report)


def test_long_beat_thresholds_and_first_30_seconds_are_bucketed(tmp_path):
    beats = [
        beat(tmp_path, 1, 0, 6, "Hook beat."),
        beat(tmp_path, 2, 30, 39, "Body review beat."),
        beat(tmp_path, 3, 39, 50, "Body high priority beat."),
        beat(tmp_path, 4, 50, 63, "Body pacing risk beat."),
    ]
    write_beats(tmp_path / "data" / "beats.json", beats)
    write_prompt_payload(
        tmp_path / "data" / "image_prompts.json",
        [
            prompt_record(1, "Hook beat.", ["time"]),
            prompt_record(2, "Body review beat.", ["deadline"]),
            prompt_record(3, "Body high priority beat.", ["deadline"]),
            prompt_record(4, "Body pacing risk beat.", ["clock"]),
        ],
    )

    result = audit(tmp_path)
    pacing = result.report["sections"]["pacing"]
    codes = finding_codes(result.report)

    assert [item["beat_number"] for item in pacing["first_30s_over_5s"]] == [1]
    assert [item["beat_number"] for item in pacing["body_over_8s"]] == [2, 3, 4]
    assert [item["beat_number"] for item in pacing["body_over_10s"]] == [3, 4]
    assert [item["beat_number"] for item in pacing["body_over_12s"]] == [4]
    assert "LONG_BEAT_FIRST_30S" in codes
    assert "LONG_BEAT_BODY_REVIEW" in codes
    assert "LONG_BEAT_BODY_HIGH_PRIORITY" in codes
    assert "LONG_BEAT_BODY_PACING_RISK" in codes


def test_visual_density_sparse_is_advisory_and_backward_compatible(tmp_path):
    beats = [
        beat(tmp_path, 1, 0, 10, "Sparse one."),
        beat(tmp_path, 2, 10, 20, "Sparse two."),
        beat(tmp_path, 3, 20, 30, "Sparse three."),
        beat(tmp_path, 4, 30, 40, "Sparse four."),
        beat(tmp_path, 5, 40, 50, "Sparse five."),
        beat(tmp_path, 6, 50, 60, "Sparse six."),
    ]
    write_beats(tmp_path / "data" / "beats.json", beats)

    result = audit(tmp_path)
    visual_density = result.report["sections"]["visual_density"]
    top_level_codes = finding_codes(result.report)

    assert visual_density["beat_count"] == 6
    assert visual_density["total_duration_seconds"] == 60
    assert visual_density["average_beat_seconds"] == 10
    assert visual_density["median_beat_seconds"] == 10
    assert visual_density["p90_beat_seconds"] == 10
    assert visual_density["estimated_dense_target_count"] == 8
    assert visual_density["estimated_preferred_beat_seconds"] == 7.5
    assert visual_density["target_range_min"] == 70
    assert visual_density["target_range_max"] == 90
    assert visual_density["density_label"] == "usable_sparse"
    assert "VISUAL_DENSITY_USABLE_SPARSE" in visual_density_codes(result.report)
    assert "DENSE_TARGET_ESTIMATE" in visual_density_codes(result.report)
    assert "VISUAL_DENSITY_USABLE_SPARSE" not in top_level_codes
    assert not result.has_core_errors


def test_visual_density_accepts_dense_average_without_sparse_warning(tmp_path):
    dense_beats = [
        beat(tmp_path, number, (number - 1) * 7.5, number * 7.5, f"Dense {number}.")
        for number in range(1, 9)
    ]
    write_beats(tmp_path / "data" / "beats.json", dense_beats)

    result = audit(tmp_path)
    visual_density = result.report["sections"]["visual_density"]

    assert visual_density["beat_count"] == 8
    assert visual_density["average_beat_seconds"] == 7.5
    assert visual_density["density_label"] == "dense_target_ready"
    assert "VISUAL_DENSITY_USABLE_SPARSE" not in visual_density_codes(result.report)
    assert visual_density["beats_over_8s"] == 0
    assert visual_density["review_priority_beats"] == []


def test_visual_density_reports_beats_over_config_max_as_nested_diagnostic(tmp_path):
    write_beats(
        tmp_path / "data" / "beats.json",
        [
            beat(tmp_path, 1, 0, 13, "Longer than configured max."),
            beat(tmp_path, 2, 13, 20, "Normal dense follow up."),
        ],
    )

    result = audit(tmp_path)
    visual_density = result.report["sections"]["visual_density"]

    assert visual_density["beats_over_12s"] == 1
    assert "BEATS_OVER_CONFIG_MAX_DURATION" in visual_density_codes(result.report)
    assert "BEATS_OVER_CONFIG_MAX_DURATION" not in finding_codes(result.report)
    assert visual_density["review_priority_beats"][0]["beat_number"] == 1
    assert "exceeds_config_max_duration" in visual_density["review_priority_beats"][0]["reason"]


def test_visual_density_caps_review_priority_and_example_lists(tmp_path):
    long_beats = []
    start = 0.0
    for number in range(1, 13):
        end = start + 10 + number / 10
        long_beats.append(beat(tmp_path, number, start, end, f"Long beat {number} with enough source detail."))
        start = end
    write_beats(tmp_path / "data" / "beats.json", long_beats)

    result = audit(tmp_path)
    visual_density = result.report["sections"]["visual_density"]

    assert visual_density["beats_over_8s"] == 12
    assert visual_density["beats_over_10s"] == 12
    assert len(visual_density["longest_beats"]) == 8
    assert len(visual_density["shortest_beats"]) == 8
    assert len(visual_density["review_priority_beats"]) == 8
    assert "LONG_STATIC_STRETCH_REVIEW" in visual_density_codes(result.report)
    assert set(visual_density["review_priority_beats"][0]) == {
        "beat_number",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "reason",
        "source_preview",
    }


def test_brightness_warnings_use_synthetic_images(tmp_path):
    beats = [
        beat(tmp_path, 1, 0, 6, "Dark."),
        beat(tmp_path, 2, 6, 12, "Bright."),
        beat(tmp_path, 3, 12, 18, "White."),
    ]
    write_beats(tmp_path / "data" / "beats.json", beats)
    write_png(tmp_path / "assets" / "generated_images" / "beat_001.png", (5, 5, 5))
    write_png(tmp_path / "assets" / "generated_images" / "beat_002.png", (220, 220, 220))
    write_png(tmp_path / "assets" / "generated_images" / "beat_003.png", (255, 255, 255))

    result = audit(tmp_path)
    codes = finding_codes(result.report)
    images = result.report["sections"]["brightness"]["images"]

    assert "IMAGE_TOO_DARK" in codes
    assert "IMAGE_TOO_BRIGHT" in codes
    assert "POSSIBLE_WHITEBOARD_FRAME" in codes
    assert [item["beat_number"] for item in images] == [1, 2, 3]


def test_metaphor_distribution_sorting_and_fallback_count(tmp_path):
    write_beats(
        tmp_path / "data" / "beats.json",
        [
            beat(tmp_path, 1, 0, 6, "One."),
            beat(tmp_path, 2, 6, 12, "Two."),
            beat(tmp_path, 3, 12, 18, "Three."),
            beat(tmp_path, 4, 18, 24, "Four."),
        ],
    )
    write_prompt_payload(
        tmp_path / "data" / "image_prompts.json",
        [
            prompt_record(1, "One.", ["time"]),
            prompt_record(2, "Two.", ["deadline"]),
            prompt_record(3, "Three.", ["time"]),
            prompt_record(4, "Four.", []),
        ],
    )

    result = audit(tmp_path)
    metaphors = result.report["sections"]["metaphors"]

    assert metaphors["keyword_distribution"] == [
        {"keyword": "time", "count": 2},
        {"keyword": "deadline", "count": 1},
    ]
    assert metaphors["fallback_count"] == 1
    assert "METAPHOR_REPETITION_RISK" in finding_codes(result.report)


def test_prompt_schema_style_and_missing_concept_warnings(tmp_path):
    write_beats(tmp_path / "data" / "beats.json", [beat(tmp_path, 1, 0, 6, "One.")])
    (tmp_path / "data" / "image_prompts.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "style_profile": "old_style",
                "prompts": [{"beat_number": 1, "source_text": "One."}],
            }
        ),
        encoding="utf-8",
    )
    write_png(tmp_path / "assets" / "generated_images" / "beat_001.png", (80, 80, 80))

    result = audit(tmp_path)
    codes = finding_codes(result.report)

    assert "PROMPT_SCHEMA_MISMATCH" in codes
    assert "PROMPT_STYLE_MISMATCH" in codes
    assert "MISSING_VISUAL_CONCEPT_FIELDS" in codes


def test_output_overwrite_and_source_text_split_warnings(tmp_path):
    write_beats(tmp_path / "data" / "beats.json", [beat(tmp_path, 1, 0, 6, "are unfinished")])
    output = tmp_path / "output" / "final_video_generated_kinetic.mp4"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"old")

    result = audit(tmp_path)
    codes = finding_codes(result.report)

    assert "OUTPUT_FILE_EXISTS" in codes
    assert "SOURCE_TEXT_SPLIT_WARNING" in codes


def test_cli_production_audit_is_mutually_exclusive(tmp_path):
    write_config(tmp_path / "config.yaml")

    try:
        main(["--config", str(tmp_path / "config.yaml"), "--dry-run", "--production-audit"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive modes")
