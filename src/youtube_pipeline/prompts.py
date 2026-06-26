from __future__ import annotations

import json
import re
from pathlib import Path

from .beat_io import load_beats, try_load_transcript_segments
from .models import Beat, TranscriptSegment

SCHEMA_VERSION = 2
STYLE_PROFILE = "melancholic_minimalist_2d_explainer"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
TURBOSCRIBE_WATERMARK = "(Transcribed by TurboScribe. Go Unlimited to remove this message.)"
MAX_VISUAL_CONCEPTS = 3

STRONG_VISUAL_CONCEPTS = (
    (("deadline", "deadlines"), "towering cracked hourglass"),
    (("management", "manage", "managed"), "invisible strings controlling a lone figure"),
    (("schedule", "calendar"), "endless grid-like labyrinth"),
    (("productivity", "productive"), "machine consuming human time"),
    (("metrics", "data", "chart", "charts"), "cold measuring eyes in the dark"),
    (("counting", "count", "numbers"), "tally marks carved into shadow"),
    (("late", "lateness"), "figure chased by a red clock hand"),
    (("task", "tasks"), "heavy repeating stone blocks"),
)
WEAK_VISUAL_CONCEPTS = (
    (
        ("time", "clock"),
        "vast shadowed space shaped by time, with a small human figure and subtle clocklike geometry",
    ),
)
DEFAULT_VISUAL_CONCEPT = "quiet symbolic scene drawn from the emotional tone of the narration"


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
        "schema_version": SCHEMA_VERSION,
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
    visual_concept_text, visual_concept_keywords = _visual_concepts(source_text)
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
        "visual_concept_text": visual_concept_text,
        "visual_concept_keywords": visual_concept_keywords,
        "image_filename": image_filename,
        "image_path": _display_path(image_path, base_dir),
        "prompt": _build_prompt(beat, source_text, visual_concept_text),
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


def sanitize_prompt_source_text(text: str) -> str:
    return _sanitize_source_text(text)


def _visual_concepts(source_text: str) -> tuple[str, list[str]]:
    strong_concepts, strong_keywords = _matching_concepts(source_text, STRONG_VISUAL_CONCEPTS)
    if strong_concepts:
        return "; ".join(strong_concepts[:MAX_VISUAL_CONCEPTS]), strong_keywords[:MAX_VISUAL_CONCEPTS]

    weak_concepts, weak_keywords = _matching_concepts(source_text, WEAK_VISUAL_CONCEPTS)
    if weak_concepts:
        return "; ".join(weak_concepts[:MAX_VISUAL_CONCEPTS]), weak_keywords[:MAX_VISUAL_CONCEPTS]

    return DEFAULT_VISUAL_CONCEPT, []


def visual_concepts_for_text(source_text: str) -> tuple[str, list[str]]:
    return _visual_concepts(source_text)


def _matching_concepts(
    source_text: str,
    concept_rules: tuple[tuple[tuple[str, ...], str], ...],
) -> tuple[list[str], list[str]]:
    concepts = []
    keywords = []
    for terms, concept in concept_rules:
        matched_term = _first_whole_word_match(source_text, terms)
        if matched_term is None:
            continue
        concepts.append(concept)
        keywords.append(matched_term)
        if len(concepts) >= MAX_VISUAL_CONCEPTS:
            break
    return concepts, keywords


def _first_whole_word_match(source_text: str, terms: tuple[str, ...]) -> str | None:
    for term in terms:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", source_text, flags=re.IGNORECASE):
            return term
    return None


def _build_prompt(beat: Beat, source_text: str, visual_concept_text: str) -> str:
    return (
        "Create one sparse symbolic image for this narration beat, using a consistent Melancholic Minimalist 2D Explainer style. "
        "Visual identity: dark charcoal, midnight blue, deep gray, muted cream highlights, "
        "soft faded yellow accents if useful, and restrained red accent only when emotionally justified. "
        "Use clean 2D line art, restrained flat shapes, soft cinematic shadow, and clear negative space. "
        "The image should feel philosophical, existential, quiet, and poetic rather than corporate or instructional. "
        "Readability: dark but readable, with muted cream highlights and enough contrast to clearly see the main figure and symbolic object. "
        "Do not make the image pure black, visually muddy, or unreadable. "
        "Composition: one central metaphor, readable at 16:9, with a lone figure or symbolic object when useful. "
        "Prefer surreal metaphorical imagery such as shadows, hourglasses, strings, labyrinths, moonlit rooms, "
        "distorted clocks, heavy blocks, empty rooms, or vast spaces surrounding a small human figure. "
        "Use symbolic diagrams only if they feel dark, poetic, surreal, and metaphorical. "
        "Avoid: pure white backgrounds, sterile whiteboard backgrounds, whiteboard explainer style, bright clinical backgrounds, "
        "corporate dashboards, business charts, productivity app layouts, calendar UI, school presentation layouts, slide decks, "
        "lecture boards, status cards, UI panels, repeated labels, text-heavy infographic cards, literal business icons, "
        "and literal productivity/scheduling/management visuals. Muted cream highlights are allowed for readability. "
        "Tone limits: philosophical and melancholic, not horror, not violent, not grotesque. "
        "Text rule: no subtitles, no embedded narration text, no captions, no logo, no watermark. "
        "Use no on-image text unless absolutely necessary; if used, limit to 0-2 short symbolic labels of 1-2 words. "
        "Frame: 16:9 YouTube frame, 1920x1080. Not photorealistic. Not a busy multi-panel storyboard. "
        f"Beat type: {beat.beat_type}. Visual metaphor hints: {visual_concept_text}. Narration beat: {source_text}"
    )


def build_standard_image_prompt(beat: Beat, source_text: str, visual_concept_text: str) -> str:
    return _build_prompt(beat, source_text, visual_concept_text)


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
