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


def prompt_record_for_text(tmp_path: Path, text: str) -> dict:
    beats_path = tmp_path / "data" / "beats.json"
    output_path = tmp_path / "data" / "image_prompts.json"
    image_dir = tmp_path / "assets" / "generated_images"
    write_beats(beats_path, [beat(tmp_path, 1, 0, 10, text)])

    payload = write_image_prompts(
        beats_path,
        tmp_path / "data" / "transcript_segments.json",
        image_dir,
        output_path,
        tmp_path,
    )

    return payload["prompts"][0]


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
    assert payload["schema_version"] == 2
    assert payload["style_profile"] == "melancholic_minimalist_2d_explainer"
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
        "visual_concept_text",
        "visual_concept_keywords",
        "image_filename",
        "image_path",
        "prompt",
    }
    assert record["beat_id"] == "beat_001"
    assert record["image_filename"] == "beat_001.png"
    assert record["image_path"] == "assets/generated_images/beat_001.png"
    assert record["source_text"] == "First fuller line. Second overlapping line."
    assert record["source_text_origin"] == "transcript_segments"
    assert record["visual_concept_text"] == "quiet symbolic scene drawn from the emotional tone of the narration"
    assert record["visual_concept_keywords"] == []
    assert isinstance(record["start_seconds"], float)
    assert isinstance(record["end_seconds"], float)
    assert record["start_timecode"] == "00:00:00,000"
    assert record["end_timecode"] == "00:00:10,000"
    assert "1920x1080" in record["prompt"]
    assert "dark charcoal" in record["prompt"]
    assert "midnight blue" in record["prompt"]
    assert "deep gray" in record["prompt"]
    assert "muted cream highlights" in record["prompt"]
    assert "dark but readable" in record["prompt"]
    assert "enough contrast to clearly see the main figure and symbolic object" in record["prompt"]
    assert "pure white backgrounds" in record["prompt"]
    assert "sterile whiteboard backgrounds" in record["prompt"]
    assert "whiteboard explainer style" in record["prompt"]
    assert "bright clinical backgrounds" in record["prompt"]
    assert "corporate dashboards" in record["prompt"]
    assert "business charts" in record["prompt"]
    assert "productivity app layouts" in record["prompt"]
    assert "calendar UI" in record["prompt"]
    assert "school presentation layouts" in record["prompt"]
    assert "slide decks" in record["prompt"]
    assert "lecture boards" in record["prompt"]
    assert "status cards" in record["prompt"]
    assert "UI panels" in record["prompt"]
    assert "Muted cream highlights are allowed for readability" in record["prompt"]
    assert "symbolic diagrams only if they feel dark, poetic, surreal, and metaphorical" in record["prompt"]
    assert "not horror" in record["prompt"]
    assert "not violent" in record["prompt"]
    assert "not grotesque" in record["prompt"]
    assert "0-2 short symbolic labels" in record["prompt"]
    assert "1-2 words" in record["prompt"]
    assert "captions" in record["prompt"]
    assert "subtitles" in record["prompt"]
    assert "embedded narration text" in record["prompt"]
    assert "no logo" in record["prompt"]
    assert "no watermark" in record["prompt"]
    assert "Not photorealistic" in record["prompt"]
    assert "Visual metaphor hints" in record["prompt"]


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


def test_keyword_matching_is_case_insensitive_and_whole_word_based(tmp_path):
    record = prompt_record_for_text(tmp_path, "The DEADLINE arrives while management waits.")

    assert record["visual_concept_text"] == "towering cracked hourglass; invisible strings controlling a lone figure"
    assert record["visual_concept_keywords"] == ["deadline", "management"]

    record = prompt_record_for_text(tmp_path, "The metadata, clockwise spiral, and untimely feeling remain.")

    assert record["visual_concept_text"] == "quiet symbolic scene drawn from the emotional tone of the narration"
    assert record["visual_concept_keywords"] == []


def test_keyword_matching_handles_requested_word_forms(tmp_path):
    cases = [
        ("Deadlines surround the bed.", "towering cracked hourglass", ["deadlines"]),
        ("They manage every morning.", "invisible strings controlling a lone figure", ["manage"]),
        ("The day is managed before waking.", "invisible strings controlling a lone figure", ["managed"]),
        ("Management is already in the room.", "invisible strings controlling a lone figure", ["management"]),
    ]

    for source_text, visual_concept, keywords in cases:
        record = prompt_record_for_text(tmp_path, source_text)
        assert record["visual_concept_text"] == visual_concept
        assert record["visual_concept_keywords"] == keywords


def test_weak_time_keyword_does_not_override_stronger_concept(tmp_path):
    record = prompt_record_for_text(tmp_path, "The deadline makes time feel hostile.")

    assert record["visual_concept_text"] == "towering cracked hourglass"
    assert record["visual_concept_keywords"] == ["deadline"]
    assert "vast shadowed space shaped by time" not in record["visual_concept_text"]


def test_weak_time_keyword_can_be_used_when_no_strong_concept_matches(tmp_path):
    record = prompt_record_for_text(tmp_path, "We wake up afraid of time.")

    assert (
        record["visual_concept_text"]
        == "vast shadowed space shaped by time, with a small human figure and subtle clocklike geometry"
    )
    assert record["visual_concept_keywords"] == ["time"]


def test_metaphor_hints_are_capped_at_three_in_priority_order(tmp_path):
    record = prompt_record_for_text(
        tmp_path,
        "Deadline management schedule productivity metrics counting late tasks all arrive together.",
    )

    assert record["visual_concept_text"] == (
        "towering cracked hourglass; invisible strings controlling a lone figure; endless grid-like labyrinth"
    )
    assert record["visual_concept_keywords"] == ["deadline", "management", "schedule"]


def test_keyword_abstraction_preserves_source_text(tmp_path):
    source_text = "Deadline and time remain exact, but source text must stay unchanged."

    record = prompt_record_for_text(tmp_path, source_text)

    assert record["source_text"] == source_text
    assert record["visual_concept_text"] == "towering cracked hourglass"
    assert record["visual_concept_keywords"] == ["deadline"]
    assert f"Narration beat: {source_text}" in record["prompt"]
