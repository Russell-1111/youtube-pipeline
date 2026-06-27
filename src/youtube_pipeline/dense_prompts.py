from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .beat_io import load_beats, load_transcript_segments
from .config import PipelineConfig
from .errors import InputFileError, PipelineError
from .models import Beat, TranscriptSegment
from .prompts import (
    sanitize_prompt_source_text,
    visual_concepts_for_text,
)

PROMPT_SCHEMA_VERSION = 1
PREVIEW_BEATS_NAME = "beats_dense_preview.json"
DENSE_REVIEW_NAME = "dense_beat_review.json"
DENSE_PLAN_NAME = "dense_beat_plan.json"
TRANSCRIPT_SEGMENTS_NAME = "transcript_segments.json"
JSON_OUTPUT_NAME = "image_prompts_dense_preview.json"
MARKDOWN_OUTPUT_NAME = "image_prompts_dense_preview.md"
ALLOWED_READINESS = {"ready", "ready_with_review"}
ALLOWED_RECOMMENDATIONS = {"approve", "review"}
RECOMMENDATIONS = ("approve", "review", "risky", "blocked")
CONTEXT_POLICY = (
    "Approve beats use only the target dense beat text. Review beats include available previous and next dense beat text "
    "only to complete fragments, avoid misleading imagery, and preserve meaning; final prompts must visualize the target beat only."
)
STYLE_CONSTRAINTS = (
    "Consistent Melancholic Minimalist 2D Explainer visual language; dark but readable 16:9 cinematic 2D symbolic imagery; "
    "no subtitles, embedded narration text, captions, logos, or watermarks; not photorealistic, not corporate, not whiteboard."
)


@dataclass(frozen=True)
class DenseScene:
    composition_family: str
    camera_shot: str
    scene_anchor: str
    main_object: str
    concept: str


@dataclass(frozen=True)
class DenseVisualFamily:
    name: str
    terms: tuple[str, ...]
    scenes: tuple[DenseScene, ...]


BODY_FATIGUE = DenseVisualFamily(
    name="body_fatigue",
    terms=(
        "wake",
        "waking",
        "woke",
        "sleep",
        "bed",
        "body",
        "chest",
        "heart",
        "breath",
        "tired",
        "exhaustion",
        "exhausted",
        "dread",
        "panic",
        "lazy",
        "rest",
    ),
    scenes=(
        DenseScene("body posture/fatigue scene", "medium side profile", "edge of a dark room at first light", "slumped shoulders", "a half-awake person sitting upright, shoulders heavy, with the room still darker than the body"),
        DenseScene("close-up human detail", "close-up", "pale morning light on skin and blanket folds", "hand on chest", "a close view of a hand resting on a tired chest while shallow breath is shown as thin cream lines"),
        DenseScene("wide empty room", "wide shot", "large quiet bedroom with most of the frame empty", "small seated body", "an exhausted body made small by an oversized dark room before the day has begun"),
        DenseScene("reflection/shadow-based composition", "reflection/shadow-based composition", "soft shadow cast across the floor beside a low bed", "body shadow", "a human shadow slowly separating from the bed, suggesting the body returning before thought"),
        DenseScene("object still life", "overhead", "rumpled sheet, pillow crease, and one unmoving hand", "rumpled sheet", "a restrained still life of bedding and a still hand, making fatigue visible without showing a device"),
    ),
)
ALARM_MACHINE_PHONE = DenseVisualFamily(
    name="alarm_machine_phone",
    terms=(
        "alarm",
        "phone",
        "clock",
        "stopwatch",
        "6.30",
        "6.31",
        "6.32",
        "903",
        "summons",
        "screaming",
    ),
    scenes=(
        DenseScene("screen-lit face or hands", "close-up", "a dim bedside surface with only one red pulse of light", "phone glow", "a hand caught in cold phone light before the person is fully awake"),
        DenseScene("object still life", "overhead", "bedside table, water glass, and a small alarm device in negative space", "alarm device", "a small alarm device isolated on a bedside table, its glow heavier than the surrounding darkness"),
        DenseScene("wide empty room", "low angle", "dark room with light slicing across the floor from the bedside", "light pulse", "a low red alarm pulse stretching across an otherwise empty room toward a still body"),
        DenseScene("threshold/window/light scene", "view through doorway/window", "bedroom seen from the hallway before anyone moves", "bedroom doorway", "a distant bedroom doorway with a small device glow inside, making the summons feel external"),
        DenseScene("reflection/shadow-based composition", "reflection/shadow-based composition", "phone light reflected faintly in a window at dawn", "window reflection", "a device glow reflected in dark glass while the human form remains blurred and secondary"),
    ),
)
OFFICE_INSTITUTION = DenseVisualFamily(
    name="office_institution",
    terms=(
        "railway",
        "railways",
        "office",
        "offices",
        "school",
        "schools",
        "bell",
        "bells",
        "factory",
        "factories",
        "hospital",
        "hospitals",
        "bank",
        "banks",
        "army",
        "armies",
        "shipping",
        "line",
        "whistle",
        "whistles",
        "shift",
        "shifts",
        "train",
        "trains",
        "permission",
        "synchronized",
        "synchronization",
    ),
    scenes=(
        DenseScene("transit/railway/station scene", "wide shot", "empty railway platform under early industrial light", "railway platform", "a small commuter shape waiting on a wide platform while rails pull the eye toward the horizon"),
        DenseScene("workplace/institution corridor", "low angle", "long institutional corridor with closed doors and one lit exit", "corridor lights", "a low-angle corridor of bells and doors, with one person paused rather than reduced to a diagram"),
        DenseScene("crowd/system pressure scene", "wide shot", "factory gate at shift change in muted dawn", "factory gate", "a narrow line of workers moving through a factory gate, the system visible through posture and spacing"),
        DenseScene("workplace/institution corridor", "side profile", "school hallway after a bell, lockers and floor shadows receding", "school hallway", "a side-profile figure in a school hallway, held by bell sound and long floor shadows"),
        DenseScene("exterior city morning", "wide shot", "office windows beginning to light above an empty street", "office windows", "office windows waking before the street does, with one small body below them for scale"),
    ),
)
CALENDAR_PRODUCTIVITY = DenseVisualFamily(
    name="calendar_productivity",
    terms=(
        "calendar",
        "schedule",
        "schedules",
        "productivity",
        "productive",
        "task",
        "tasks",
        "deadline",
        "deadlines",
        "friday",
        "year",
        "habit",
        "track",
        "measure",
        "measured",
        "optimize",
        "batch",
        "slot",
        "allowance",
        "vacation",
        "lunch",
        "evidence",
        "visible",
        "failure",
        "failed",
        "manage",
        "planning",
    ),
    scenes=(
        DenseScene("object still life", "overhead", "desk surface with scattered blank cards and one cold cup", "blank task cards", "unfinished blank task cards scattered around a cold cup, avoiding any readable checklist or UI"),
        DenseScene("body posture/fatigue scene", "medium shot", "seated body at a table with stacked paper blocks nearby", "paper stack", "a seated person beside stacked paper blocks, the weight of tasks shown through posture"),
        DenseScene("wide empty room", "wide shot", "plain room with empty frames on one wall and a person near the floor", "empty frames", "empty wall frames pressing quietly toward a small person without forming a calendar grid"),
        DenseScene("threshold/window/light scene", "view through doorway/window", "kitchen doorway with morning light and an untouched bag by the exit", "untouched bag", "a doorway scene where an untouched bag and angled light imply a schedule without showing a planner"),
        DenseScene("close-up human detail", "close-up", "fingers hovering above blank paper in low light", "blank paper", "a close view of fingers above blank paper, showing pressure before any task is written"),
    ),
)
CITY_APARTMENT = DenseVisualFamily(
    name="city_apartment",
    terms=(
        "room",
        "window",
        "breakfast",
        "household",
        "apartment",
        "pocket",
        "arrive",
        "arrival",
        "late",
        "lateness",
        "bathroom",
    ),
    scenes=(
        DenseScene("domestic apartment interior", "wide shot", "small apartment kitchen before sunrise", "untouched breakfast", "a narrow apartment kitchen with untouched breakfast and one lit window holding the morning outside"),
        DenseScene("exterior city morning", "view through doorway/window", "city blocks seen beyond a dark apartment window", "apartment window", "a person crossing a dim room while the city waits beyond the window in pale geometric light"),
        DenseScene("object still life", "overhead", "table corner with keys, shoes, and a cooling cup", "keys and shoes", "keys, shoes, and a cooling cup arranged like evidence of leaving before being ready"),
        DenseScene("threshold/window/light scene", "side profile", "half-lit doorway between bedroom and hall", "threshold", "a side-profile figure paused at a threshold, one shoe on, body not yet caught up with the day"),
        DenseScene("reflection/shadow-based composition", "reflection/shadow-based composition", "bathroom mirror with muted dawn behind the viewer", "mirror reflection", "a tired reflection in a bathroom mirror, with the room suggested by light rather than symbols"),
    ),
)
PREINDUSTRIAL_LIGHT = DenseVisualFamily(
    name="preindustrial_light",
    terms=(
        "sun",
        "sunlight",
        "light",
        "darkness",
        "wind",
        "bird",
        "birds",
        "rain",
        "harvest",
        "field",
        "fields",
        "fire",
        "animals",
        "animal",
        "seasons",
        "season",
        "village",
        "town",
        "church",
        "bell",
        "bells",
        "cooking",
        "rice",
        "hunger",
        "fullness",
        "birth",
        "death",
        "weather",
        "illness",
    ),
    scenes=(
        DenseScene("rural/preindustrial landscape", "wide shot", "field edge under weather-colored morning light", "field path", "a field path under changing weather, with one small worker moving at the pace of light"),
        DenseScene("domestic apartment interior", "medium shot", "low fire and cooking pot in a dark simple room", "cooking fire", "a low cooking fire, quiet animals outside, and fading light as a human-scale measure of time"),
        DenseScene("rural/preindustrial landscape", "side profile", "rain line crossing a rural path with sunlight behind it", "rain path", "sunlight and rain dividing a simple path, concrete and environmental rather than diagrammatic"),
        DenseScene("crowd/system pressure scene", "wide shot", "village square at dusk with people gathering loosely", "village square", "a village square gathering under dark sky, bodies spaced naturally around a shared sound"),
        DenseScene("threshold/window/light scene", "view through doorway/window", "doorway looking onto rice, firelight, and evening air", "firelit doorway", "rice, firelight, and a waiting doorway forming a slow domestic rhythm outside clock time"),
    ),
)
SCREEN_DIGITAL = DenseVisualFamily(
    name="screen_digital",
    terms=(
        "screen",
        "screens",
        "message",
        "messages",
        "notification",
        "notifications",
        "symbols",
        "symbol",
        "unanswered",
        "lights",
        "digital",
        "number",
        "numbers",
        "grid",
    ),
    scenes=(
        DenseScene("screen-lit face or hands", "close-up", "dark room lit only by a device below the frame", "screen-lit hands", "screen-lit hands and a partial face, attention pulled downward without showing fake UI"),
        DenseScene("object still life", "overhead", "small glowing rectangle beside fabric and a glass of water", "notification glow", "a tiny notification glow beside ordinary objects, heavy because the rest of the room is still"),
        DenseScene("reflection/shadow-based composition", "reflection/shadow-based composition", "pale device light reflected on a wall and cheek", "wall reflection", "digital light reflected across a wall and cheek, with no readable icons or labels"),
        DenseScene("close-up human detail", "close-up", "thumb hovering above a dark screen with no text", "hovering thumb", "a thumb suspended above a dark screen, the choice delayed in a tight human detail"),
        DenseScene("wide empty room", "wide shot", "empty room made colder by one small rectangle of light", "small rectangle of light", "one small rectangle of light making a large room feel colder and less human"),
    ),
)
ABSTRACT_TIME_PRESSURE = DenseVisualFamily(
    name="abstract_time_pressure",
    terms=(
        "time",
        "minute",
        "minutes",
        "hour",
        "hours",
        "guilt",
        "anxiety",
        "pressure",
        "debt",
        "threat",
        "danger",
        "shame",
        "ashamed",
        "judge",
        "judged",
        "value",
        "worth",
        "attention",
        "coordination",
        "coordinate",
        "control",
        "controlled",
        "money",
        "truth",
        "slipping",
        "ruler",
        "colonized",
    ),
    scenes=(
        DenseScene("wide empty room", "low angle", "low ceiling of shadow above one standing body", "low ceiling shadow", "invisible time pressure shown as a low ceiling of shadow above a body, with no clock or diagram"),
        DenseScene("reflection/shadow-based composition", "side profile", "long floor shadow following a walking person", "debt shadow", "a quiet debt-like shadow following one body through negative space"),
        DenseScene("close-up human detail", "close-up", "eye and cheek lit by one narrow band of cream light", "narrow light band", "a close human detail where attention is squeezed by a narrow band of light"),
        DenseScene("threshold/window/light scene", "view through doorway/window", "window light cut into thin strips across a room", "striped window light", "thin window light tightening around a person holding a dim light, without becoming a grid"),
        DenseScene("object still life", "overhead", "open hands catching pale dust-like grains of light", "open hands", "life slipping through open hands as pale grains of light, restrained and philosophical"),
    ),
)
DENSE_VISUAL_FAMILIES = (
    BODY_FATIGUE,
    ALARM_MACHINE_PHONE,
    OFFICE_INSTITUTION,
    CALENDAR_PRODUCTIVITY,
    SCREEN_DIGITAL,
    PREINDUSTRIAL_LIGHT,
    CITY_APARTMENT,
    ABSTRACT_TIME_PRESSURE,
)
DENSE_FALLBACK_SCENES = (
    DenseScene("threshold/window/light scene", "view through doorway/window", "plain threshold with one angled strip of morning light", "threshold light", "a quiet threshold scene with a person between rest and obligation, cinematic and minimal"),
    DenseScene("wide empty room", "wide shot", "spare modern interior with one chair and a body-sized pool of light", "pool of light", "a body-sized pool of light surrounded by dark negative space, grounded in modern life"),
    DenseScene("object still life", "overhead", "ordinary table with cup, keys, and a long shadow", "cup and keys", "ordinary objects on a table carrying the beat's emotional weight through light and spacing"),
    DenseScene("reflection/shadow-based composition", "reflection/shadow-based composition", "wall shadow of a person outside the visible frame", "wall shadow", "a human shadow on a dim wall, restrained and concrete rather than abstract"),
    DenseScene("exterior city morning", "wide shot", "empty street just before morning traffic begins", "empty street", "an empty early street with one lit window, grounding the idea in place and weather"),
)


@dataclass(frozen=True)
class DensePromptResult:
    payload: dict[str, Any]
    json_path: Path
    markdown_path: Path


def run_dense_prompt_generation(config: PipelineConfig, base_dir: Path) -> DensePromptResult:
    data_dir = config.outputs.data_dir
    preview_path = data_dir / PREVIEW_BEATS_NAME
    review_path = data_dir / DENSE_REVIEW_NAME
    plan_path = data_dir / DENSE_PLAN_NAME
    transcript_path = data_dir / TRANSCRIPT_SEGMENTS_NAME
    json_path = data_dir / JSON_OUTPUT_NAME
    markdown_path = data_dir / MARKDOWN_OUTPUT_NAME

    preview_beats = load_beats(preview_path)
    review = _load_json_object(review_path, "dense beat review")
    dense_plan = _load_json_object(plan_path, "dense beat plan")
    segments = load_transcript_segments(transcript_path)

    payload = build_dense_prompt_payload(
        preview_beats=preview_beats,
        review=review,
        dense_plan=dense_plan,
        segments=segments,
        preview_path=preview_path,
        review_path=review_path,
        plan_path=plan_path,
        transcript_path=transcript_path,
        json_path=json_path,
        markdown_path=markdown_path,
        base_dir=base_dir,
    )
    markdown = _markdown_report(payload)

    data_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return DensePromptResult(payload=payload, json_path=json_path, markdown_path=markdown_path)


def build_dense_prompt_payload(
    preview_beats: list[Beat],
    review: dict[str, Any],
    dense_plan: dict[str, Any],
    segments: list[TranscriptSegment],
    preview_path: Path = Path("data/beats_dense_preview.json"),
    review_path: Path = Path("data/dense_beat_review.json"),
    plan_path: Path = Path("data/dense_beat_plan.json"),
    transcript_path: Path = Path("data/transcript_segments.json"),
    json_path: Path = Path("data/image_prompts_dense_preview.json"),
    markdown_path: Path = Path("data/image_prompts_dense_preview.md"),
    base_dir: Path | None = None,
) -> dict[str, Any]:
    base = base_dir or Path.cwd()
    review_rows = _validate_inputs(preview_beats, review, dense_plan, segments)
    segment_by_index = {segment.index: segment for segment in segments}
    source_text_by_number = {
        beat.beat_number: _source_text(beat, segment_by_index)
        for beat in preview_beats
    }
    prompts = []
    recent_families: list[str] = []
    recent_objects: list[str] = []
    for index, beat in enumerate(preview_beats):
        row = _prompt_row(
            beat=beat,
            review_row=review_rows[beat.beat_number],
            source_text_by_number=source_text_by_number,
            previous_beat=preview_beats[index - 1] if index > 0 else None,
            next_beat=preview_beats[index + 1] if index < len(preview_beats) - 1 else None,
            recent_families=recent_families,
            recent_objects=recent_objects,
        )
        prompts.append(row)
        recent_families.append(row["composition_family"])
        recent_objects.append(row["main_object"])
        recent_families = recent_families[-3:]
        recent_objects = recent_objects[-4:]
    counts = _recommendation_counts(review, prompts)
    return {
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source_dense_review_readiness": review["readiness"],
        "total_dense_beats": len(preview_beats),
        "prompt_count": len(prompts),
        "context_policy": CONTEXT_POLICY,
        "source_paths": {
            "dense_preview_beats": _display_path(preview_path, base),
            "dense_review": _display_path(review_path, base),
            "dense_plan": _display_path(plan_path, base),
            "transcript_segments": _display_path(transcript_path, base),
        },
        "output_paths": {
            "json": _display_path(json_path, base),
            "markdown": _display_path(markdown_path, base),
        },
        "recommendation_counts": counts,
        "planner_metadata": _planner_metadata(dense_plan),
        "prompts": prompts,
    }


def print_dense_prompt_summary(result: DensePromptResult, base_dir: Path) -> None:
    payload = result.payload
    counts = payload["recommendation_counts"]
    context_count = sum(1 for row in payload["prompts"] if row["context_used"])
    print("Dense prompt generation complete")
    print(f"Total dense beats: {payload['total_dense_beats']}")
    print(f"Prompts: {payload['prompt_count']}")
    print(f"Review readiness: {payload['source_dense_review_readiness']}")
    print(
        "Recommendations: "
        f"{counts['approve']} approve, {counts['review']} review, "
        f"{counts['risky']} risky, {counts['blocked']} blocked"
    )
    print(f"Review-context prompts: {context_count}")
    print("Reports written:")
    print(f"- {_display_path(result.json_path, base_dir)}")
    print(f"- {_display_path(result.markdown_path, base_dir)}")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise InputFileError(f"Missing {label} file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid {label} JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise PipelineError(f"Could not read {label} JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"Invalid {label} JSON: expected an object in {path}")
    return payload


def _validate_inputs(
    preview_beats: list[Beat],
    review: dict[str, Any],
    dense_plan: dict[str, Any],
    segments: list[TranscriptSegment],
) -> dict[int, dict[str, Any]]:
    if not preview_beats:
        raise PipelineError("Cannot generate dense prompts without dense preview beats.")
    if not segments:
        raise PipelineError("Cannot generate dense prompts without transcript segments.")
    if not isinstance(dense_plan.get("dense_group_records"), list):
        raise PipelineError("Invalid dense beat plan JSON: expected dense_group_records list.")

    readiness = review.get("readiness")
    if readiness not in ALLOWED_READINESS:
        raise PipelineError(f"Dense review readiness must be ready or ready_with_review; found {readiness!r}.")

    rows = review.get("beats")
    if not isinstance(rows, list):
        raise PipelineError("Invalid dense beat review JSON: expected beats list.")
    total_dense_beats = review.get("total_dense_beats")
    if not isinstance(total_dense_beats, int):
        raise PipelineError("Invalid dense beat review JSON: missing total_dense_beats.")
    if len(preview_beats) != total_dense_beats or len(preview_beats) != len(rows):
        raise PipelineError(
            "Dense beat count mismatch: "
            f"{len(preview_beats)} preview beats, {total_dense_beats} review total, {len(rows)} review rows."
        )

    review_by_number: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise PipelineError(f"Invalid dense review row at index {index}: expected an object.")
        beat_number = row.get("dense_beat_number")
        if not isinstance(beat_number, int):
            raise PipelineError(f"Invalid dense review row at index {index}: missing dense_beat_number.")
        if beat_number in review_by_number:
            raise PipelineError(f"Invalid dense beat review JSON: duplicate review row for beat {beat_number}.")
        recommendation = row.get("recommendation")
        if recommendation not in set(RECOMMENDATIONS):
            raise PipelineError(f"Invalid dense review row for beat {beat_number}: unknown recommendation {recommendation!r}.")
        review_by_number[beat_number] = row

    missing = [beat.beat_number for beat in preview_beats if beat.beat_number not in review_by_number]
    extra = sorted(set(review_by_number) - {beat.beat_number for beat in preview_beats})
    if missing or extra:
        raise PipelineError(f"Dense beat/review rows do not match. Missing: {missing or []}. Extra: {extra or []}.")

    counts = _normalized_counts(review.get("recommendation_counts"), rows)
    if counts["risky"]:
        raise PipelineError(f"Dense prompt generation is blocked: {counts['risky']} risky beats exist.")
    if counts["blocked"]:
        raise PipelineError(f"Dense prompt generation is blocked: {counts['blocked']} blocked beats exist.")
    disallowed = [
        row["dense_beat_number"]
        for row in rows
        if row.get("recommendation") not in ALLOWED_RECOMMENDATIONS
    ]
    if disallowed:
        raise PipelineError(f"Dense prompt generation is blocked by disallowed recommendations on beats: {disallowed}.")
    return review_by_number


def _normalized_counts(source_counts: object, rows: list[dict[str, Any]]) -> dict[str, int]:
    actual = Counter(row.get("recommendation") for row in rows)
    counts = {recommendation: int(actual.get(recommendation, 0)) for recommendation in RECOMMENDATIONS}
    if source_counts is not None:
        if not isinstance(source_counts, dict):
            raise PipelineError("Invalid dense beat review JSON: recommendation_counts must be an object.")
        for recommendation in RECOMMENDATIONS:
            value = source_counts.get(recommendation, 0)
            if not isinstance(value, int):
                raise PipelineError(f"Invalid dense beat review JSON: recommendation_counts.{recommendation} must be an integer.")
            if value != counts[recommendation]:
                raise PipelineError(
                    f"Dense review recommendation_counts mismatch for {recommendation}: "
                    f"report has {value}, rows have {counts[recommendation]}."
                )
    return counts


def _recommendation_counts(review: dict[str, Any], prompts: list[dict[str, Any]]) -> dict[str, int]:
    rows = [{"recommendation": prompt["review_recommendation"]} for prompt in prompts]
    return _normalized_counts(review.get("recommendation_counts"), rows)


def _prompt_row(
    beat: Beat,
    review_row: dict[str, Any],
    source_text_by_number: dict[int, str],
    previous_beat: Beat | None,
    next_beat: Beat | None,
    recent_families: list[str] | None = None,
    recent_objects: list[str] | None = None,
) -> dict[str, Any]:
    recommendation = review_row["recommendation"]
    target_text = source_text_by_number[beat.beat_number]
    context_used = recommendation == "review"
    previous_context = source_text_by_number.get(previous_beat.beat_number) if context_used and previous_beat else None
    next_context = source_text_by_number.get(next_beat.beat_number) if context_used and next_beat else None
    scene, visual_concept_keywords = _dense_visual_scene_for_text(
        target_text,
        beat.beat_number,
        recent_families or [],
        recent_objects or [],
    )
    visual_concept_text = scene.concept
    final_prompt = _build_dense_image_prompt(beat, target_text, scene)
    if context_used:
        final_prompt = _with_review_context(final_prompt, target_text, previous_context, next_context)

    return {
        "dense_beat_number": beat.beat_number,
        "beat_type": beat.beat_type,
        "start_seconds": round(beat.start_seconds, 3),
        "end_seconds": round(beat.end_seconds, 3),
        "start_timecode": beat.start,
        "end_timecode": beat.end,
        "duration_seconds": round(beat.duration_seconds, 3),
        "review_recommendation": recommendation,
        "target_text": target_text,
        "previous_context_text": previous_context,
        "next_context_text": next_context,
        "context_used": context_used,
        "visual_concept_text": visual_concept_text,
        "visual_concept_keywords": visual_concept_keywords,
        "composition_family": scene.composition_family,
        "camera_shot": scene.camera_shot,
        "scene_anchor": scene.scene_anchor,
        "main_object": scene.main_object,
        "final_image_prompt": final_prompt,
        "style_constraints": STYLE_CONSTRAINTS,
        "prompt_risk_notes": _prompt_risk_notes(review_row),
    }


def _with_review_context(
    base_prompt: str,
    target_text: str,
    previous_context: str | None,
    next_context: str | None,
) -> str:
    previous = previous_context or "None available."
    next_text = next_context or "None available."
    return (
        f"{base_prompt} Dense review context policy: visualize only the target beat: {target_text} "
        "Use adjacent context only to complete fragments, avoid misleading imagery, and preserve meaning. "
        f"Previous beat context: {previous} Next beat context: {next_text} "
        "Do not depict a different beat's event, metaphor, setting, or action."
    )


def _prompt_risk_notes(review_row: dict[str, Any]) -> list[str]:
    notes = []
    for key in ("warning_codes", "score_reasons"):
        value = review_row.get(key, [])
        if isinstance(value, list):
            notes.extend(str(item) for item in value)
    return notes


def _build_dense_image_prompt(beat: Beat, source_text: str, scene: DenseScene) -> str:
    allowed_motifs = _allowed_repeated_motifs(source_text, scene)
    motif_policy = (
        f"Target-relevant repeated motif allowance: {', '.join(allowed_motifs)}. Keep any allowed repeated motif concrete and secondary unless it is the main subject."
        if allowed_motifs
        else "Avoid the failed QA motifs: wall timepieces, handheld devices, sleeping furniture, rectangular exit icons, tracker tables, and institutional schematics."
    )
    return (
        f"Create one {scene.composition_family} image for this narration beat in a Melancholic Minimalist 2D Explainer style. "
        f"Camera and framing: {scene.camera_shot}, 16:9 YouTube frame, 1920x1080, one readable focal area with strong negative space. "
        f"Concrete scene anchor: {scene.scene_anchor}. Main subject or object: {scene.main_object}. "
        f"Visual target: {scene.concept}. "
        "Use dark charcoal, midnight blue, deep gray, muted cream highlights, soft faded yellow accents only when useful, "
        "and restrained red accent only when emotionally justified. Use clean 2D line art, restrained flat shapes, soft cinematic shadow, "
        "and enough contrast to keep the figure, object, and spatial relation readable. "
        "Make the image concrete and cinematic: specify place, subject, object, light, and spatial relation through the composition itself. "
        "Avoid generic symbolic rooms, icon-like people, schematic institution layouts, tracker-table compositions, productivity dashboards, "
        "fake UI, readable text, labels, subtitles, captions, logos, and watermarks. "
        f"{motif_policy} "
        "Tone limits: philosophical and melancholic, not horror, not violent, not grotesque, not corporate, not instructional, not whiteboard. "
        f"Beat type: {beat.beat_type}. Narration beat: {source_text}"
    )


def _allowed_repeated_motifs(source_text: str, scene: DenseScene) -> list[str]:
    combined = f"{source_text} {scene.main_object} {scene.scene_anchor} {scene.concept}"
    rules = (
        ("clock", ("clock", "alarm", "6.30", "6.31", "6.32")),
        ("phone", ("phone", "screen", "message", "notification", "device")),
        ("bed", ("bed", "sleep", "waking", "bedroom", "bedsheet", "pillow")),
        ("calendar/grid", ("calendar", "schedule", "habit", "tracker", "grid")),
        ("doorway", ("door", "doorway", "threshold", "exit")),
    )
    return [
        label
        for label, terms in rules
        if _matching_dense_terms(combined, terms)
    ]


def _dense_visual_scene_for_text(
    source_text: str,
    beat_number: int,
    recent_families: list[str],
    recent_objects: list[str],
) -> tuple[DenseScene, list[str]]:
    family_matches = [
        (family, matched_terms)
        for family in DENSE_VISUAL_FAMILIES
        if (matched_terms := _matching_dense_terms(source_text, family.terms))
    ]
    if family_matches:
        family, matched_terms = _select_dense_family(family_matches, beat_number, recent_families)
        return _dense_scene_variant(source_text, beat_number, family.scenes, recent_families, recent_objects), matched_terms[:3]

    standard_concept, standard_keywords = visual_concepts_for_text(source_text)
    if standard_keywords:
        fallback_scene = _dense_scene_variant(
            f"{source_text} {standard_concept}",
            beat_number,
            DENSE_FALLBACK_SCENES,
            recent_families,
            recent_objects,
        )
        return DenseScene(
            fallback_scene.composition_family,
            fallback_scene.camera_shot,
            fallback_scene.scene_anchor,
            fallback_scene.main_object,
            standard_concept,
        ), standard_keywords

    return _dense_scene_variant(source_text, beat_number, DENSE_FALLBACK_SCENES, recent_families, recent_objects), []


def _select_dense_family(
    family_matches: list[tuple[DenseVisualFamily, list[str]]],
    beat_number: int,
    recent_families: list[str],
) -> tuple[DenseVisualFamily, list[str]]:
    max_match_count = max(len(matched_terms) for _, matched_terms in family_matches)
    strongest = [match for match in family_matches if len(match[1]) == max_match_count]
    non_recent = [
        match
        for match in strongest
        if any(scene.composition_family not in recent_families[-2:] for scene in match[0].scenes)
    ]
    candidates = non_recent or strongest
    index = _fingerprint(str(beat_number) + " ".join(term for _, terms in candidates for term in terms)) % len(candidates)
    return candidates[index]


def _matching_dense_terms(source_text: str, terms: tuple[str, ...]) -> list[str]:
    matches = []
    for term in terms:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", source_text, flags=re.IGNORECASE):
            matches.append(term)
    return matches


def _dense_scene_variant(
    source_text: str,
    beat_number: int,
    scenes: tuple[DenseScene, ...],
    recent_families: list[str],
    recent_objects: list[str],
) -> DenseScene:
    if not scenes:
        raise PipelineError("Cannot select dense visual scene from an empty scene list.")
    start = _fingerprint(f"{beat_number} {source_text}") % len(scenes)
    ordered = [scenes[(start + offset) % len(scenes)] for offset in range(len(scenes))]
    for scene in ordered:
        if scene.composition_family not in recent_families[-2:] and scene.main_object not in recent_objects[-3:]:
            return scene
    for scene in ordered:
        if scene.main_object not in recent_objects[-3:]:
            return scene
    for scene in ordered:
        if scene.composition_family not in recent_families[-2:]:
            return scene
    return ordered[0]


def _fingerprint(source_text: str) -> int:
    normalized = source_text.lower()
    return sum((index + 1) * ord(char) for index, char in enumerate(normalized))


def _source_text(beat: Beat, segment_by_index: dict[int, TranscriptSegment]) -> str:
    if beat.segment_indexes:
        text = " ".join(
            segment_by_index[index].text.strip()
            for index in beat.segment_indexes
            if index in segment_by_index and segment_by_index[index].text.strip()
        )
        if text:
            return sanitize_prompt_source_text(text)
    return sanitize_prompt_source_text(beat.text_preview)


def _planner_metadata(dense_plan: dict[str, Any]) -> dict[str, Any]:
    summary = dense_plan.get("summary")
    return {
        "planner_version": dense_plan.get("planner_version"),
        "planner_strategy": dense_plan.get("planner_strategy"),
        "planner_created_at": dense_plan.get("created_at"),
        "dense_preview_beat_count": dense_plan.get("dense_preview_beat_count"),
        "safe_to_apply": dense_plan.get("safe_to_apply"),
        "target_range_min": dense_plan.get("target_range_min"),
        "target_range_max": dense_plan.get("target_range_max"),
        "warnings_count": len(dense_plan.get("warnings", [])) if isinstance(dense_plan.get("warnings"), list) else None,
        "summary": summary if isinstance(summary, dict) else {},
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    counts = payload["recommendation_counts"]
    context_rows = [row for row in payload["prompts"] if row["context_used"]]
    lines = [
        "# Dense Image Prompts Preview",
        "",
        "## Summary",
        "",
        f"- Dense beat count: {payload['total_dense_beats']}",
        f"- Prompt count: {payload['prompt_count']}",
        f"- Review readiness: `{payload['source_dense_review_readiness']}`",
        f"- Recommendation counts: {counts['approve']} approve, {counts['review']} review, {counts['risky']} risky, {counts['blocked']} blocked",
        f"- Context policy: {payload['context_policy']}",
        "",
        "## Prompt-generation policy",
        "",
        "- This preview writes dense prompt artifacts only and does not overwrite production prompt, beat, or transcript files.",
        "- Approved beats are generated from target dense beat text only.",
        "- Review beats may use neighboring context only to preserve the target beat's meaning; the image prompt must stay focused on the target beat.",
        "",
        "## Review beats using neighboring context",
        "",
    ]
    if context_rows:
        lines.append("| Beat | Previous Context | Target | Next Context |")
        lines.append("| ---: | --- | --- | --- |")
        for row in context_rows:
            lines.append(
                f"| {row['dense_beat_number']} | {_md_cell(row['previous_context_text'] or '-')} | "
                f"{_md_cell(row['target_text'])} | {_md_cell(row['next_context_text'] or '-')} |"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Compact prompt list", ""])
    lines.append("| Beat | Time | Recommendation | Context | Family | Camera | Main Object | Visual Concept | Target Preview |")
    lines.append("| ---: | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["prompts"]:
        lines.append(
            f"| {row['dense_beat_number']} | {row['start_seconds']:.3f}-{row['end_seconds']:.3f}s | "
            f"{row['review_recommendation']} | {row['context_used']} | "
            f"{_md_cell(row['composition_family'])} | {_md_cell(row['camera_shot'])} | {_md_cell(row['main_object'])} | "
            f"{_md_cell(row['visual_concept_text'])} | {_md_cell(_preview(row['target_text'], 90))} |"
        )

    lines.extend(["", "## Output file paths", ""])
    lines.append(f"- JSON: `{payload['output_paths']['json']}`")
    lines.append(f"- Markdown: `{payload['output_paths']['markdown']}`")
    lines.append("")
    return "\n".join(lines)


def _preview(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
