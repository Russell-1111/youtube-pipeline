from __future__ import annotations

import json
from pathlib import Path

from .beat_io import load_beats, try_load_transcript_segments
from .models import Beat, TranscriptSegment

STYLE_PROFILE = "minimalist_2d_explainer_clean_line_art"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
TURBOSCRIBE_WATERMARK = "(Transcribed by TurboScribe. Go Unlimited to remove this message.)"


def write_image_prompts(
    beats_path: Path,
    transcript_segments_path: Path,
    image_dir: Path,
    output_path: Path,
    base_dir: Path,
) -> dict:
    beats = load_beats(beats_path)
    segments = try_load_transcript_segments(transcript_segments_path)
    image_dir.mkdir(parents=True, exist_ok=True)

    image_directory = _display_path(image_dir, base_dir)
    payload = {
        "schema_version": 1,
        "style_profile": STYLE_PROFILE,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "image_directory": image_directory,
        "prompts": [
            _prompt_record(beat, segments, image_dir, base_dir)
            for beat in beats
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _prompt_record(
    beat: Beat,
    segments: list[TranscriptSegment] | None,
    image_dir: Path,
    base_dir: Path,
) -> dict:
    source_text, source_text_origin = _source_text(beat, segments)
    beat_id = f"beat_{beat.beat_number:03}"
    image_filename = f"{beat_id}.png"
    image_path = image_dir / image_filename
    return {
        "beat_number": beat.beat_number,
        "beat_id": beat_id,
        "beat_type": beat.beat_type,
        "start_seconds": beat.start_seconds,
        "end_seconds": beat.end_seconds,
        "start_timecode": beat.start,
        "end_timecode": beat.end,
        "duration_seconds": beat.duration_seconds,
        "source_text": source_text,
        "source_text_origin": source_text_origin,
        "image_filename": image_filename,
        "image_path": _display_path(image_path, base_dir),
        "prompt": _build_prompt(beat, source_text),
    }


def _source_text(beat: Beat, segments: list[TranscriptSegment] | None) -> tuple[str, str]:
    if segments is not None:
        overlapping = [
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
            and segment.start_seconds < beat.end_seconds
            and segment.end_seconds > beat.start_seconds
        ]
        if overlapping:
            return _sanitize_source_text(" ".join(overlapping)), "transcript_segments"
    return _sanitize_source_text(beat.text_preview), "text_preview"


def _sanitize_source_text(text: str) -> str:
    return " ".join(text.replace(TURBOSCRIBE_WATERMARK, " ").split())


def _build_prompt(beat: Beat, source_text: str) -> str:
    return (
        "Create one clear visual concept for this narration beat, not a subtitle card. "
        "Style: Minimalist 2D Explainer / Clean Line Art Animation, clean medium-weight black outlines, "
        "rudimentary stick figures, flat solid colors, sparse backgrounds, infographic layout. "
        "Use simple charts, timelines, arrows, boxes, UI mockups, labels, or diagrams when useful. "
        "Allow only short on-image text labels, ideally 1-5 words. "
        "Avoid long sentences inside the image. "
        "Frame: 16:9 YouTube frame, 1920x1080. "
        "Do not use gradients, realistic textures, photorealism, cinematic realism, watermark, logo, or subtitles. "
        f"Beat type: {beat.beat_type}. Narration beat: {source_text}"
    )


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
