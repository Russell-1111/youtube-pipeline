import json
from pathlib import Path

from youtube_pipeline.manifest import write_beats, write_transcript_segments
from youtube_pipeline.models import Beat, TranscriptSegment
from youtube_pipeline.prompts import write_image_prompts


def beat(tmp_path: Path, number: int, start_seconds: float, end_seconds: float, preview: str) -> Beat:
    return Beat(
        beat_number=number,
        beat_type="normal",
        start="00:00:00,000",
        end="00:00:10,000",
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
        start="00:00:00,000",
        end="00:00:10,000",
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        duration_seconds=end_seconds - start_seconds,
        text=text,
    )


def test_prompt_json_schema_and_record_shape(tmp_path):
    beats_path = tmp_path / "data" / "beats.json"
    segments_path = tmp_path / "data" / "transcript_segments.json"
    output_path = tmp_path / "data" / "image_prompts.json"
    image_dir = tmp_path / "assets" / "generated_images"
    write_beats(beats_path, [beat(tmp_path, 1, 0, 10, "Preview fallback")])
    write_transcript_segments(
        segments_path,
        [
            segment(1, 0, 4, "First fuller line."),
            segment(2, 9, 12, "Second overlapping line."),
            segment(3, 10, 12, "Non overlapping line."),
        ],
    )

    payload = write_image_prompts(beats_path, segments_path, image_dir, output_path, tmp_path)

    assert image_dir.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    assert payload["schema_version"] == 1
    assert payload["style_profile"] == "minimalist_2d_explainer_clean_line_art"
    assert payload["image_width"] == 1920
    assert payload["image_height"] == 1080
    assert payload["image_directory"] == "assets/generated_images"

    record = payload["prompts"][0]
    assert set(record) == {
        "beat_number",
        "beat_id",
        "beat_type",
        "start_seconds",
        "end_seconds",
        "start_timecode",
        "end_timecode",
        "duration_seconds",
        "source_text",
        "source_text_origin",
        "image_filename",
        "image_path",
        "prompt",
    }
    assert record["beat_id"] == "beat_001"
    assert record["image_filename"] == "beat_001.png"
    assert record["image_path"] == "assets/generated_images/beat_001.png"
    assert record["source_text"] == "First fuller line. Second overlapping line."
    assert record["source_text_origin"] == "transcript_segments"
    assert isinstance(record["start_seconds"], float)
    assert isinstance(record["end_seconds"], float)
    assert record["start_timecode"] == "00:00:00,000"
    assert record["end_timecode"] == "00:00:10,000"
    assert "1920x1080" in record["prompt"]
    assert "Do not use gradients" in record["prompt"]
    assert "photorealism" in record["prompt"]
    assert "watermark" in record["prompt"]


def test_prompt_source_text_falls_back_to_text_preview_when_segments_missing(tmp_path):
    beats_path = tmp_path / "data" / "beats.json"
    output_path = tmp_path / "data" / "image_prompts.json"
    image_dir = tmp_path / "assets" / "generated_images"
    write_beats(beats_path, [beat(tmp_path, 1, 0, 10, "Preview fallback")])

    payload = write_image_prompts(
        beats_path,
        tmp_path / "data" / "transcript_segments.json",
        image_dir,
        output_path,
        tmp_path,
    )

    record = payload["prompts"][0]
    assert record["source_text"] == "Preview fallback"
    assert record["source_text_origin"] == "text_preview"


def test_prompt_source_text_falls_back_to_text_preview_when_segments_are_unusable(tmp_path):
    beats_path = tmp_path / "data" / "beats.json"
    segments_path = tmp_path / "data" / "transcript_segments.json"
    output_path = tmp_path / "data" / "image_prompts.json"
    image_dir = tmp_path / "assets" / "generated_images"
    write_beats(beats_path, [beat(tmp_path, 1, 0, 10, "Preview fallback")])
    segments_path.write_text("{not valid json", encoding="utf-8")

    payload = write_image_prompts(beats_path, segments_path, image_dir, output_path, tmp_path)

    record = payload["prompts"][0]
    assert record["source_text"] == "Preview fallback"
    assert record["source_text_origin"] == "text_preview"


def test_prompt_source_text_removes_turboscribe_watermark_from_source_and_prompt(tmp_path):
    beats_path = tmp_path / "data" / "beats.json"
    segments_path = tmp_path / "data" / "transcript_segments.json"
    output_path = tmp_path / "data" / "image_prompts.json"
    image_dir = tmp_path / "assets" / "generated_images"
    watermark = "(Transcribed by TurboScribe. Go Unlimited to remove this message.)"
    write_beats(beats_path, [beat(tmp_path, 1, 0, 10, f"{watermark} Preview fallback")])
    write_transcript_segments(
        segments_path,
        [
            segment(1, 0, 4, f"  {watermark}   Actual narration starts here."),
            segment(2, 4, 8, "More useful context."),
        ],
    )

    payload = write_image_prompts(beats_path, segments_path, image_dir, output_path, tmp_path)

    record = payload["prompts"][0]
    assert record["source_text"] == "Actual narration starts here. More useful context."
    assert record["source_text_origin"] == "transcript_segments"
    assert "TurboScribe" not in record["source_text"]
    assert "TurboScribe" not in record["prompt"]
