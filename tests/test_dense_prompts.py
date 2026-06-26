import hashlib
import json
from pathlib import Path

from youtube_pipeline.__main__ import main
from youtube_pipeline.dense_prompts import build_dense_prompt_payload
from youtube_pipeline.manifest import write_beats, write_transcript_segments
from youtube_pipeline.models import Beat, TranscriptSegment


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


def beat(number: int, text: str) -> Beat:
    start = float((number - 1) * 4)
    end = start + 4.0
    return Beat(
        beat_number=number,
        beat_type="normal",
        start=f"00:00:{int(start):02},000",
        end=f"00:00:{int(end):02},000",
        start_seconds=start,
        end_seconds=end,
        duration_seconds=4.0,
        text_preview=text,
        segment_indexes=[number],
        image_path=f"assets/images/beat_{number:03}.png",
    )


def segment(number: int, text: str) -> TranscriptSegment:
    start = float((number - 1) * 4)
    end = start + 4.0
    return TranscriptSegment(
        index=number,
        start=f"00:00:{int(start):02},000",
        end=f"00:00:{int(end):02},000",
        start_seconds=start,
        end_seconds=end,
        duration_seconds=4.0,
        text=text,
    )


def dense_plan(count: int = 3) -> dict:
    return {
        "planner_version": 1,
        "planner_strategy": "hybrid_dense_rebuild",
        "created_at": "2026-06-26T00:00:00Z",
        "dense_preview_beat_count": count,
        "target_range_min": 55,
        "target_range_max": 75,
        "safe_to_apply": True,
        "warnings": [],
        "summary": {"target_range_reached": True},
        "dense_group_records": [
            {
                "dense_beat_number": number,
                "beat_type": "normal",
                "source_standard_beat_number": 1,
                "start_seconds": float((number - 1) * 4),
                "end_seconds": float(number * 4),
                "duration_seconds": 4.0,
                "segment_indexes": [number],
                "grouping_score": 0.0,
                "boundary_confidence": "high",
                "reason": "test",
            }
            for number in range(1, count + 1)
        ],
    }


def dense_review(recommendations: list[str], readiness: str = "ready_with_review") -> dict:
    counts = {name: recommendations.count(name) for name in ("approve", "review", "risky", "blocked")}
    return {
        "review_schema_version": 1,
        "created_at": "2026-06-26T00:00:00Z",
        "total_dense_beats": len(recommendations),
        "recommendation_counts": counts,
        "readiness": readiness,
        "beats": [
            {
                "dense_beat_number": index,
                "beat_type": "normal",
                "start_seconds": float((index - 1) * 4),
                "end_seconds": float(index * 4),
                "duration_seconds": 4.0,
                "source_text_preview": f"Beat {index}",
                "warning_codes": ["WEAK_PUNCTUATION_BOUNDARY"] if recommendation == "review" else [],
                "boundary_confidence": "weak" if recommendation == "review" else "high",
                "nearest_standard_beat_number": 1,
                "source_coherence_label": "minor_boundary_issue" if recommendation == "review" else "coherent",
                "review_priority": 20 if recommendation == "review" else 0,
                "recommendation": recommendation,
                "score_reasons": ["weak_punctuation_boundary:+12"] if recommendation == "review" else ["no_review_flags:+0"],
            }
            for index, recommendation in enumerate(recommendations, start=1)
        ],
    }


def write_dense_inputs(tmp_path: Path, recommendations: list[str], readiness: str = "ready_with_review") -> None:
    data_dir = tmp_path / "data"
    beats = [
        beat(1, "The morning starts under a deadline."),
        beat(2, "because time keeps pressing forward"),
        beat(3, "The room returns to quiet focus."),
    ][: len(recommendations)]
    segments = [
        segment(1, "The morning starts under a deadline."),
        segment(2, "because time keeps pressing forward"),
        segment(3, "The room returns to quiet focus."),
    ][: len(recommendations)]
    write_beats(data_dir / "beats_dense_preview.json", beats)
    write_transcript_segments(data_dir / "transcript_segments.json", segments)
    (data_dir / "dense_beat_plan.json").write_text(json.dumps(dense_plan(len(recommendations)), indent=2), encoding="utf-8")
    (data_dir / "dense_beat_review.json").write_text(
        json.dumps(dense_review(recommendations, readiness), indent=2),
        encoding="utf-8",
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successful_dense_prompt_generation_writes_json_and_markdown_without_overwriting_protected_files(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_dense_inputs(tmp_path, ["approve", "review", "approve"])
    protected = {
        "image_prompts": tmp_path / "data" / "image_prompts.json",
        "beats": tmp_path / "data" / "beats.json",
        "transcript": tmp_path / "data" / "transcript_segments.json",
    }
    protected["image_prompts"].write_text('{"existing": "standard prompts"}', encoding="utf-8")
    write_beats(protected["beats"], [beat(1, "Standard beat")])
    before = {name: file_hash(path) for name, path in protected.items()}

    result = main(["--config", str(config_path), "--generate-dense-prompts"])

    after = {name: file_hash(path) for name, path in protected.items()}
    json_path = tmp_path / "data" / "image_prompts_dense_preview.json"
    markdown_path = tmp_path / "data" / "image_prompts_dense_preview.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert result == 0
    assert json_path.exists()
    assert markdown_path.exists()
    assert before == after
    assert payload["prompt_count"] == 3
    assert payload["recommendation_counts"] == {"approve": 2, "review": 1, "risky": 0, "blocked": 0}
    assert "# Dense Image Prompts Preview" in markdown


def test_review_beats_receive_previous_and_next_context_while_approve_beats_do_not(tmp_path):
    beats = [
        beat(1, "Previous complete sentence."),
        beat(2, "because the center fragment needs help"),
        beat(3, "Next complete sentence."),
    ]
    payload = build_dense_prompt_payload(
        preview_beats=beats,
        review=dense_review(["approve", "review", "approve"]),
        dense_plan=dense_plan(3),
        segments=[
            segment(1, "Previous complete sentence."),
            segment(2, "because the center fragment needs help"),
            segment(3, "Next complete sentence."),
        ],
        base_dir=tmp_path,
    )

    approve = payload["prompts"][0]
    review = payload["prompts"][1]
    assert approve["context_used"] is False
    assert approve["previous_context_text"] is None
    assert approve["next_context_text"] is None
    assert review["context_used"] is True
    assert review["previous_context_text"] == "Previous complete sentence."
    assert review["next_context_text"] == "Next complete sentence."
    assert "visualize only the target beat" in review["final_image_prompt"]
    assert "Do not depict a different beat" in review["final_image_prompt"]


def test_first_beat_review_context_handles_missing_previous_context(tmp_path):
    payload = build_dense_prompt_payload(
        preview_beats=[beat(1, "because the first fragment opens"), beat(2, "Next complete sentence.")],
        review=dense_review(["review", "approve"]),
        dense_plan=dense_plan(2),
        segments=[segment(1, "because the first fragment opens"), segment(2, "Next complete sentence.")],
        base_dir=tmp_path,
    )

    row = payload["prompts"][0]
    assert row["context_used"] is True
    assert row["previous_context_text"] is None
    assert row["next_context_text"] == "Next complete sentence."
    assert "Previous beat context: None available." in row["final_image_prompt"]


def test_json_output_has_required_top_level_and_prompt_row_fields(tmp_path):
    payload = build_dense_prompt_payload(
        preview_beats=[beat(1, "The deadline moves through the room.")],
        review=dense_review(["approve"], readiness="ready"),
        dense_plan=dense_plan(1),
        segments=[segment(1, "The deadline moves through the room.")],
        base_dir=tmp_path,
    )

    assert set(payload) == {
        "prompt_schema_version",
        "created_at",
        "source_dense_review_readiness",
        "total_dense_beats",
        "prompt_count",
        "context_policy",
        "source_paths",
        "output_paths",
        "recommendation_counts",
        "planner_metadata",
        "prompts",
    }
    assert set(payload["prompts"][0]) == {
        "dense_beat_number",
        "beat_type",
        "start_seconds",
        "end_seconds",
        "start_timecode",
        "end_timecode",
        "duration_seconds",
        "review_recommendation",
        "target_text",
        "previous_context_text",
        "next_context_text",
        "context_used",
        "visual_concept_text",
        "visual_concept_keywords",
        "final_image_prompt",
        "style_constraints",
        "prompt_risk_notes",
    }


def test_markdown_output_has_summary_policy_review_context_and_compact_prompt_list(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_dense_inputs(tmp_path, ["approve", "review"])

    result = main(["--config", str(config_path), "--generate-dense-prompts"])

    markdown = (tmp_path / "data" / "image_prompts_dense_preview.md").read_text(encoding="utf-8")
    assert result == 0
    assert "## Summary" in markdown
    assert "## Prompt-generation policy" in markdown
    assert "## Review beats using neighboring context" in markdown
    assert "## Compact prompt list" in markdown
    assert "## Output file paths" in markdown


def test_risky_beat_fails_before_writing_outputs(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_dense_inputs(tmp_path, ["approve", "risky"])

    result = main(["--config", str(config_path), "--generate-dense-prompts"])

    assert result == 1
    assert not (tmp_path / "data" / "image_prompts_dense_preview.json").exists()
    assert not (tmp_path / "data" / "image_prompts_dense_preview.md").exists()


def test_blocked_beat_fails_before_writing_outputs(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_dense_inputs(tmp_path, ["approve", "blocked"])

    result = main(["--config", str(config_path), "--generate-dense-prompts"])

    assert result == 1
    assert not (tmp_path / "data" / "image_prompts_dense_preview.json").exists()
    assert not (tmp_path / "data" / "image_prompts_dense_preview.md").exists()


def test_not_ready_review_fails_before_writing_outputs(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_dense_inputs(tmp_path, ["approve"], readiness="not_ready")

    result = main(["--config", str(config_path), "--generate-dense-prompts"])

    assert result == 1
    assert not (tmp_path / "data" / "image_prompts_dense_preview.json").exists()
    assert not (tmp_path / "data" / "image_prompts_dense_preview.md").exists()


def test_mismatched_dense_beat_and_review_counts_fail_before_writing_outputs(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_dense_inputs(tmp_path, ["approve", "approve"])
    review = dense_review(["approve"])
    (tmp_path / "data" / "dense_beat_review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")

    result = main(["--config", str(config_path), "--generate-dense-prompts"])

    assert result == 1
    assert not (tmp_path / "data" / "image_prompts_dense_preview.json").exists()
    assert not (tmp_path / "data" / "image_prompts_dense_preview.md").exists()


def test_malformed_json_fails_before_writing_outputs(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_dense_inputs(tmp_path, ["approve"])
    (tmp_path / "data" / "dense_beat_review.json").write_text("{not valid json", encoding="utf-8")

    result = main(["--config", str(config_path), "--generate-dense-prompts"])

    assert result == 1
    assert not (tmp_path / "data" / "image_prompts_dense_preview.json").exists()
    assert not (tmp_path / "data" / "image_prompts_dense_preview.md").exists()
