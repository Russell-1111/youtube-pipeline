import hashlib
import json
from pathlib import Path

from PIL import Image

from youtube_pipeline.__main__ import main
from youtube_pipeline.dense_images import build_dense_image_report
from youtube_pipeline.manifest import write_beats
from youtube_pipeline.models import Beat


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


def beat(number: int) -> Beat:
    start = float((number - 1) * 6)
    end = start + 6.0
    return Beat(
        beat_number=number,
        beat_type="normal",
        start=f"00:00:{int(start):02},000",
        end=f"00:00:{int(end):02},000",
        start_seconds=start,
        end_seconds=end,
        duration_seconds=6.0,
        text_preview=f"Dense beat {number}",
        segment_indexes=[number],
        image_path=f"assets/images/beat_{number:03}.png",
    )


def prompt_row(number: int, *, context_used: bool = False, concept: str | None = None) -> dict:
    return {
        "dense_beat_number": number,
        "review_recommendation": "review" if context_used else "approve",
        "context_used": context_used,
        "visual_concept_text": concept or f"visual concept family {number % 5}",
        "final_image_prompt": f"Prompt for dense beat {number}",
    }


def prompts_payload(count: int, review_context_numbers: set[int] | None = None) -> dict:
    review_context_numbers = review_context_numbers or set()
    return {
        "prompt_schema_version": 1,
        "created_at": "2026-06-26T00:00:00Z",
        "total_dense_beats": count,
        "prompt_count": count,
        "prompts": [
            prompt_row(number, context_used=number in review_context_numbers)
            for number in range(1, count + 1)
        ],
    }


def review_payload(count: int, review_context_numbers: set[int] | None = None) -> dict:
    review_context_numbers = review_context_numbers or set()
    return {
        "review_schema_version": 1,
        "created_at": "2026-06-26T00:00:00Z",
        "total_dense_beats": count,
        "recommendation_counts": {
            "approve": count - len(review_context_numbers),
            "review": len(review_context_numbers),
            "risky": 0,
            "blocked": 0,
        },
        "readiness": "ready_with_review",
        "beats": [
            {
                "dense_beat_number": number,
                "recommendation": "review" if number in review_context_numbers else "approve",
            }
            for number in range(1, count + 1)
        ],
    }


def dense_plan(count: int) -> dict:
    return {
        "planner_version": 1,
        "dense_preview_beat_count": count,
        "dense_group_records": [{"dense_beat_number": number} for number in range(1, count + 1)],
    }


def write_dense_inputs(tmp_path: Path, count: int = 3, review_context_numbers: set[int] | None = None) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_beats(data_dir / "beats_dense_preview.json", [beat(number) for number in range(1, count + 1)])
    (data_dir / "image_prompts_dense_preview.json").write_text(
        json.dumps(prompts_payload(count, review_context_numbers), indent=2),
        encoding="utf-8",
    )
    (data_dir / "dense_beat_review.json").write_text(
        json.dumps(review_payload(count, review_context_numbers), indent=2),
        encoding="utf-8",
    )
    (data_dir / "dense_beat_plan.json").write_text(json.dumps(dense_plan(count), indent=2), encoding="utf-8")


def write_png(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successful_dense_image_preparation_writes_json_and_markdown_and_creates_directory(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_dense_inputs(tmp_path, 3, {2})

    result = main(["--config", str(config_path), "--prepare-dense-images"])

    report_path = tmp_path / "data" / "dense_image_generation_report.json"
    markdown_path = tmp_path / "data" / "dense_image_generation_report.md"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert result == 0
    assert (tmp_path / "assets" / "generated_images_dense_preview").is_dir()
    assert report["expected_image_count"] == 3
    assert report["missing_image_count"] == 3
    assert report["review_context_image_count"] == 1
    assert "# Dense Image Generation Report" in markdown


def test_dense_image_preparation_does_not_touch_protected_or_standard_paths(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_dense_inputs(tmp_path, 2)
    protected = {
        "image_prompts": tmp_path / "data" / "image_prompts.json",
        "beats": tmp_path / "data" / "beats.json",
        "transcript": tmp_path / "data" / "transcript_segments.json",
    }
    protected["image_prompts"].write_text('{"standard": true}', encoding="utf-8")
    write_beats(protected["beats"], [beat(1)])
    protected["transcript"].write_text('{"segments": []}', encoding="utf-8")
    before = {name: file_hash(path) for name, path in protected.items()}

    result = main(["--config", str(config_path), "--prepare-dense-images"])

    after = {name: file_hash(path) for name, path in protected.items()}
    assert result == 0
    assert before == after
    assert not (tmp_path / "assets" / "generated_images").exists()


def test_missing_images_are_reported_but_exit_zero(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_dense_inputs(tmp_path, 2)

    result = main(["--config", str(config_path), "--prepare-dense-images"])

    report = json.loads((tmp_path / "data" / "dense_image_generation_report.json").read_text(encoding="utf-8"))
    assert result == 0
    assert report["missing_image_count"] == 2
    assert report["ready_for_dense_render"] is False
    assert all(row["status"] == "missing" for row in report["images"])


def test_existing_valid_png_images_are_reported_valid(tmp_path):
    image_dir = tmp_path / "assets" / "generated_images_dense_preview"
    write_png(image_dir / "dense_beat_001.png", (1920, 1080))

    report = build_dense_image_report(
        prompts_payload=prompts_payload(1),
        preview_beats=[beat(1)],
        review_payload=review_payload(1),
        dense_plan=dense_plan(1),
        image_dir=image_dir,
        base_dir=tmp_path,
    )

    assert report["present_image_count"] == 1
    assert report["invalid_image_count"] == 0
    assert report["ready_for_dense_render"] is True
    assert report["images"][0]["status"] == "valid"


def test_too_small_png_is_reported_invalid(tmp_path):
    image_dir = tmp_path / "assets" / "generated_images_dense_preview"
    write_png(image_dir / "dense_beat_001.png", (640, 360))

    report = build_dense_image_report(prompts_payload(1), [beat(1)], review_payload(1), dense_plan(1), image_dir, base_dir=tmp_path)

    assert report["invalid_image_count"] == 1
    assert report["images"][0]["status"] == "invalid"
    assert "too small" in report["images"][0]["validation_messages"][0]


def test_wrong_aspect_ratio_png_is_reported_invalid(tmp_path):
    image_dir = tmp_path / "assets" / "generated_images_dense_preview"
    write_png(image_dir / "dense_beat_001.png", (1280, 800))

    report = build_dense_image_report(prompts_payload(1), [beat(1)], review_payload(1), dense_plan(1), image_dir, base_dir=tmp_path)

    assert report["invalid_image_count"] == 1
    assert any("exact 16:9" in message for message in report["images"][0]["validation_messages"])


def test_non_preferred_valid_16_9_size_creates_warning_not_invalid(tmp_path):
    image_dir = tmp_path / "assets" / "generated_images_dense_preview"
    write_png(image_dir / "dense_beat_001.png", (1280, 720))

    report = build_dense_image_report(prompts_payload(1), [beat(1)], review_payload(1), dense_plan(1), image_dir, base_dir=tmp_path)

    assert report["invalid_image_count"] == 0
    assert report["warning_count"] == 1
    assert report["images"][0]["status"] == "warning"
    assert report["ready_for_dense_render"] is True


def test_extra_png_files_are_reported_as_warnings(tmp_path):
    image_dir = tmp_path / "assets" / "generated_images_dense_preview"
    write_png(image_dir / "dense_beat_001.png", (1920, 1080))
    write_png(image_dir / "dense_beat_999.png", (1920, 1080))

    report = build_dense_image_report(prompts_payload(1), [beat(1)], review_payload(1), dense_plan(1), image_dir, base_dir=tmp_path)

    assert report["warning_count"] == 1
    assert report["images"][-1]["status"] == "extra"
    assert report["images"][-1]["expected_filename"] == "dense_beat_999.png"


def test_duplicate_expected_filenames_are_detected_before_writing_outputs(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_dense_inputs(tmp_path, 2)
    payload = prompts_payload(2)
    payload["prompts"][1]["dense_beat_number"] = 1
    (tmp_path / "data" / "image_prompts_dense_preview.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(["--config", str(config_path), "--prepare-dense-images"])

    assert result == 1
    assert not (tmp_path / "data" / "dense_image_generation_report.json").exists()
    assert not (tmp_path / "data" / "dense_image_generation_report.md").exists()


def test_review_context_prompts_are_surfaced_in_report(tmp_path):
    report = build_dense_image_report(
        prompts_payload=prompts_payload(3, {2}),
        preview_beats=[beat(1), beat(2), beat(3)],
        review_payload=review_payload(3, {2}),
        dense_plan=dense_plan(3),
        image_dir=tmp_path / "assets" / "generated_images_dense_preview",
        base_dir=tmp_path,
    )

    assert report["review_context_image_count"] == 1
    assert [row["dense_beat_number"] for row in report["images"] if row["context_used"]] == [2]


def test_sample_batch_recommendation_is_deterministic_and_includes_review_context_beats(tmp_path):
    review_context = {7, 8, 13, 33, 48, 56, 63}
    report_a = build_dense_image_report(
        prompts_payload=prompts_payload(70, review_context),
        preview_beats=[beat(number) for number in range(1, 71)],
        review_payload=review_payload(70, review_context),
        dense_plan=dense_plan(70),
        image_dir=tmp_path / "assets" / "generated_images_dense_preview",
        base_dir=tmp_path,
    )
    report_b = build_dense_image_report(
        prompts_payload=prompts_payload(70, review_context),
        preview_beats=[beat(number) for number in range(1, 71)],
        review_payload=review_payload(70, review_context),
        dense_plan=dense_plan(70),
        image_dir=tmp_path / "assets" / "generated_images_dense_preview",
        base_dir=tmp_path,
    )

    sample = report_a["sample_batch_recommendation"]
    sample_numbers = [row["dense_beat_number"] for row in sample]
    assert sample == report_b["sample_batch_recommendation"]
    assert 8 <= len(sample) <= 12
    assert {7, 8, 13, 33, 48, 56, 63}.issubset(sample_numbers)
    assert any(row["review_context"] for row in sample)


def test_malformed_json_fails_before_writing_outputs(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_dense_inputs(tmp_path, 1)
    (tmp_path / "data" / "image_prompts_dense_preview.json").write_text("{not valid json", encoding="utf-8")

    result = main(["--config", str(config_path), "--prepare-dense-images"])

    assert result == 1
    assert not (tmp_path / "data" / "dense_image_generation_report.json").exists()
    assert not (tmp_path / "data" / "dense_image_generation_report.md").exists()


def test_mismatched_prompt_beat_review_structures_fail_before_writing_outputs(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    write_dense_inputs(tmp_path, 2)
    review = review_payload(1)
    (tmp_path / "data" / "dense_beat_review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")

    result = main(["--config", str(config_path), "--prepare-dense-images"])

    assert result == 1
    assert not (tmp_path / "data" / "dense_image_generation_report.json").exists()
    assert not (tmp_path / "data" / "dense_image_generation_report.md").exists()
