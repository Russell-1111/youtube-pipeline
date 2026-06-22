from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .models import Beat
from .whiteboard import (
    BLACK,
    DARK_GRAY,
    GRAY,
    LIGHT_GRAY,
    MIDNIGHT,
    RED,
    WHITE,
    YELLOW,
    draw_bed,
    draw_book,
    draw_house,
    draw_light_bulb,
    draw_stick_figure,
    draw_sun,
    draw_switch,
    draw_timeline,
    draw_upper_text,
    draw_wrapped_upper_text,
    hand_arrow,
    hand_ellipse,
    hand_line,
    hand_rect,
    hand_x,
    marker_font,
    preview_text,
    rng_for_beat,
    scribble_fill,
    select_layout,
    short_label,
)


def generate_placeholder_images(beats: list[Beat], width: int, height: int) -> None:
    image_dirs = {Path(beat.image_path).parent for beat in beats}
    for image_dir in image_dirs:
        image_dir.mkdir(parents=True, exist_ok=True)
        for old_image in image_dir.glob("beat_*.png"):
            if old_image.is_file():
                old_image.unlink()

    for beat in beats:
        output_path = Path(beat.image_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = render_whiteboard_frame(beat, width, height)
        image.save(output_path)


def render_whiteboard_frame(beat: Beat, width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    rng = rng_for_beat(beat)
    layout = select_layout(beat)

    _draw_whiteboard_border(draw, width, height, rng)
    _draw_meta(draw, beat, width, height, rng)

    if layout == "title":
        _draw_title_layout(draw, beat, width, height, rng)
    elif layout == "pause":
        _draw_pause_layout(draw, beat, width, height, rng)
    elif layout == "timeline":
        _draw_timeline_layout(draw, beat, width, height, rng)
    elif layout == "switch":
        _draw_switch_layout(draw, beat, width, height, rng)
    elif layout == "bed":
        _draw_bed_layout(draw, beat, width, height, rng)
    elif layout == "book":
        _draw_book_layout(draw, beat, width, height, rng)
    elif layout == "comparison":
        _draw_comparison_layout(draw, beat, width, height, rng)
    elif layout == "cause_effect":
        _draw_cause_effect_layout(draw, beat, width, height, rng)
    elif layout == "house":
        _draw_house_layout(draw, beat, width, height, rng)
    elif layout == "idea":
        _draw_idea_layout(draw, beat, width, height, rng)
    else:
        _draw_general_layout(draw, beat, width, height, rng)

    _draw_preview_strip(draw, beat, width, height, rng)
    return image


def _draw_whiteboard_border(draw: ImageDraw.ImageDraw, width: int, height: int, rng) -> None:
    margin = max(28, width // 70)
    hand_rect(draw, (margin, margin, width - margin, height - margin), rng, outline=LIGHT_GRAY, width=max(2, width // 520))


def _draw_meta(draw: ImageDraw.ImageDraw, beat: Beat, width: int, height: int, rng) -> None:
    font = marker_font(max(18, width // 54), bold=True)
    small = marker_font(max(12, width // 90), bold=True)
    x = max(50, width // 32)
    y = max(45, height // 24)
    draw_upper_text(draw, (x, y), f"BEAT {beat.beat_number:03}", font, fill=DARK_GRAY)
    hand_line(draw, (x, y + int(font.size * 1.25)), (x + width * 0.16, y + int(font.size * 1.25)), rng, fill=YELLOW, width=max(4, width // 260))
    timestamp = _timestamp_label(beat, compact=width < 900)
    draw_upper_text(draw, (width - int(width * 0.39), y + 6), timestamp, small, fill=DARK_GRAY)
    draw_upper_text(draw, (width - int(width * 0.16), y + int(small.size * 1.45)), beat.beat_type, small, fill=RED)


def _draw_preview_strip(draw: ImageDraw.ImageDraw, beat: Beat, width: int, height: int, rng) -> None:
    box = (int(width * 0.08), int(height * 0.78), int(width * 0.92), int(height * 0.91))
    scribble_fill(draw, box, rng, fill=LIGHT_GRAY, width=max(2, width // 480), spacing=max(16, height // 46))
    hand_rect(draw, box, rng, outline=GRAY, width=max(3, width // 380), fill=None)
    label_font = marker_font(max(14, width // 66), bold=True)
    body_font = marker_font(max(15, width // 58), bold=False)
    draw_upper_text(draw, (box[0] + 28, box[1] + 20), "PREVIEW", label_font, fill=MIDNIGHT)
    draw_wrapped_upper_text(
        draw,
        (box[0] + max(120, int(width * 0.14)), box[1] + 22),
        preview_text(beat.text_preview, 85),
        body_font,
        max_chars=max(28, width // 38),
        fill=BLACK,
        max_lines=2,
    )


def _draw_title_layout(draw: ImageDraw.ImageDraw, beat: Beat, width: int, height: int, rng) -> None:
    title_font = marker_font(max(34, width // 17), bold=True)
    label_font = marker_font(max(20, width // 42), bold=True)
    center_x = width // 2
    y = int(height * 0.25)
    title = short_label(beat.text_preview, "INTRO", words=5)
    _center_text(draw, (center_x, y), title, title_font, fill=MIDNIGHT)
    hand_ellipse(draw, (center_x - int(width * 0.25), y - 45, center_x + int(width * 0.25), y + 95), rng, outline=YELLOW, width=max(6, width // 210))
    draw_stick_figure(draw, (int(width * 0.24), int(height * 0.56)), width / 1920, rng)
    draw_light_bulb(draw, (int(width * 0.69), int(height * 0.49)), width / 1920, rng)
    hand_arrow(draw, (int(width * 0.34), int(height * 0.53)), (int(width * 0.56), int(height * 0.49)), rng)
    _center_text(draw, (center_x, int(height * 0.65)), "CLEAN EXPLAINER", label_font, fill=DARK_GRAY)


def _draw_pause_layout(draw: ImageDraw.ImageDraw, beat: Beat, width: int, height: int, rng) -> None:
    title_font = marker_font(max(30, width // 20), bold=True)
    label_font = marker_font(max(20, width // 45), bold=True)
    _center_text(draw, (width // 2, int(height * 0.25)), "PAUSE / SILENCE", title_font, fill=MIDNIGHT)
    x1, y1, x2, y2 = int(width * 0.38), int(height * 0.38), int(width * 0.62), int(height * 0.58)
    hand_rect(draw, (x1, y1, x2, y2), rng, width=max(6, width // 220), fill=LIGHT_GRAY)
    bar_w = int((x2 - x1) * 0.18)
    hand_rect(draw, (x1 + int(width * 0.045), y1 + 25, x1 + int(width * 0.045) + bar_w, y2 - 25), rng, width=max(5, width // 300), fill=GRAY)
    hand_rect(draw, (x2 - int(width * 0.045) - bar_w, y1 + 25, x2 - int(width * 0.045), y2 - 25), rng, width=max(5, width // 300), fill=GRAY)
    hand_arrow(draw, (int(width * 0.28), int(height * 0.49)), (x1 - 40, int(height * 0.49)), rng)
    hand_arrow(draw, (x2 + 40, int(height * 0.49)), (int(width * 0.72), int(height * 0.49)), rng)
    _center_text(draw, (width // 2, int(height * 0.66)), "TRANSITION", label_font, fill=RED)


def _draw_timeline_layout(draw: ImageDraw.ImageDraw, beat: Beat, width: int, height: int, rng) -> None:
    title_font = marker_font(max(28, width // 22), bold=True)
    label_font = marker_font(max(18, width // 50), bold=True)
    _center_text(draw, (width // 2, int(height * 0.2)), "TIMELINE", title_font, fill=MIDNIGHT)
    draw_timeline(draw, (int(width * 0.14), int(height * 0.36), int(width * 0.86), int(height * 0.57)), rng)
    labels = ["PAST", "NOW", "CHANGE", "NEXT"]
    for idx, label in enumerate(labels):
        x = int(width * (0.17 + idx * 0.22))
        draw_upper_text(draw, (x, int(height * 0.62)), label, label_font, fill=RED if label == "CHANGE" else BLACK)


def _draw_switch_layout(draw: ImageDraw.ImageDraw, beat: Beat, width: int, height: int, rng) -> None:
    title_font = marker_font(max(28, width // 22), bold=True)
    label_font = marker_font(max(20, width // 46), bold=True)
    _center_text(draw, (width // 2, int(height * 0.2)), "SWITCH STATE", title_font, fill=MIDNIGHT)
    draw_switch(draw, (int(width * 0.34), int(height * 0.39), int(width * 0.66), int(height * 0.55)), rng, on=False)
    draw_upper_text(draw, (int(width * 0.28), int(height * 0.61)), "OFF", label_font)
    draw_upper_text(draw, (int(width * 0.62), int(height * 0.61)), "ON?", label_font, fill=RED)
    hand_arrow(draw, (int(width * 0.47), int(height * 0.68)), (int(width * 0.56), int(height * 0.6)), rng)


def _draw_bed_layout(draw: ImageDraw.ImageDraw, beat: Beat, width: int, height: int, rng) -> None:
    title_font = marker_font(max(28, width // 22), bold=True)
    _center_text(draw, (width // 2, int(height * 0.19)), "NIGHT ROUTINE", title_font, fill=MIDNIGHT)
    draw_bed(draw, (int(width * 0.18), int(height * 0.33), int(width * 0.62), int(height * 0.62)), rng)
    hand_x(draw, (int(width * 0.68), int(height * 0.38), int(width * 0.81), int(height * 0.57)), rng)
    draw_upper_text(draw, (int(width * 0.66), int(height * 0.61)), "AVOID", marker_font(max(20, width // 48), bold=True), fill=RED)


def _draw_book_layout(draw: ImageDraw.ImageDraw, beat: Beat, width: int, height: int, rng) -> None:
    title_font = marker_font(max(28, width // 22), bold=True)
    label_font = marker_font(max(20, width // 48), bold=True)
    _center_text(draw, (width // 2, int(height * 0.2)), "LEARN THE IDEA", title_font, fill=MIDNIGHT)
    draw_book(draw, (int(width * 0.2), int(height * 0.34), int(width * 0.52), int(height * 0.62)), rng)
    draw_light_bulb(draw, (int(width * 0.72), int(height * 0.47)), width / 1920, rng)
    hand_arrow(draw, (int(width * 0.55), int(height * 0.48)), (int(width * 0.65), int(height * 0.48)), rng)
    draw_upper_text(draw, (int(width * 0.62), int(height * 0.64)), "KNOW", label_font, fill=RED)


def _draw_comparison_layout(draw: ImageDraw.ImageDraw, beat: Beat, width: int, height: int, rng) -> None:
    title_font = marker_font(max(28, width // 24), bold=True)
    label_font = marker_font(max(20, width // 48), bold=True)
    _center_text(draw, (width // 2, int(height * 0.18)), "BEFORE / AFTER", title_font, fill=MIDNIGHT)
    left = (int(width * 0.13), int(height * 0.34), int(width * 0.43), int(height * 0.62))
    right = (int(width * 0.57), int(height * 0.34), int(width * 0.87), int(height * 0.62))
    hand_rect(draw, left, rng, width=max(6, width // 250), fill=WHITE)
    hand_rect(draw, right, rng, width=max(6, width // 250), fill=WHITE)
    draw_upper_text(draw, (left[0] + 40, left[1] + 30), "BEFORE", label_font)
    draw_upper_text(draw, (right[0] + 40, right[1] + 30), "AFTER", label_font, fill=RED)
    fill_top = min(left[1] + 100, left[3] - 55)
    scribble_fill(draw, (left[0] + 45, fill_top, left[2] - 45, left[3] - 35), rng, fill=GRAY, spacing=16)
    hand_arrow(draw, (int(width * 0.46), int(height * 0.48)), (int(width * 0.54), int(height * 0.48)), rng)
    draw_light_bulb(draw, (int(width * 0.72), int(height * 0.5)), width / 2500, rng)


def _draw_cause_effect_layout(draw: ImageDraw.ImageDraw, beat: Beat, width: int, height: int, rng) -> None:
    title_font = marker_font(max(28, width // 24), bold=True)
    label_font = marker_font(max(20, width // 48), bold=True)
    _center_text(draw, (width // 2, int(height * 0.18)), "CAUSE -> EFFECT", title_font, fill=MIDNIGHT)
    left = (int(width * 0.14), int(height * 0.38), int(width * 0.40), int(height * 0.58))
    right = (int(width * 0.60), int(height * 0.38), int(width * 0.86), int(height * 0.58))
    hand_rect(draw, left, rng, fill=LIGHT_GRAY, width=6)
    hand_rect(draw, right, rng, fill=YELLOW, width=6)
    draw_upper_text(draw, (left[0] + 45, left[1] + 60), "CAUSE", label_font)
    draw_upper_text(draw, (right[0] + 45, right[1] + 60), "EFFECT", label_font, fill=RED)
    hand_arrow(draw, (left[2] + 35, int(height * 0.48)), (right[0] - 35, int(height * 0.48)), rng)


def _draw_house_layout(draw: ImageDraw.ImageDraw, beat: Beat, width: int, height: int, rng) -> None:
    title_font = marker_font(max(28, width // 22), bold=True)
    _center_text(draw, (width // 2, int(height * 0.2)), "HOME BASE", title_font, fill=MIDNIGHT)
    draw_house(draw, (int(width * 0.24), int(height * 0.34), int(width * 0.55), int(height * 0.64)), rng)
    draw_sun(draw, (int(width * 0.73), int(height * 0.38)), int(width * 0.04), rng)
    hand_arrow(draw, (int(width * 0.57), int(height * 0.51)), (int(width * 0.67), int(height * 0.43)), rng)


def _draw_idea_layout(draw: ImageDraw.ImageDraw, beat: Beat, width: int, height: int, rng) -> None:
    title_font = marker_font(max(28, width // 22), bold=True)
    label_font = marker_font(max(20, width // 48), bold=True)
    _center_text(draw, (width // 2, int(height * 0.19)), "BIG IDEA", title_font, fill=MIDNIGHT)
    draw_stick_figure(draw, (int(width * 0.28), int(height * 0.58)), width / 1920, rng)
    draw_light_bulb(draw, (int(width * 0.58), int(height * 0.43)), width / 1700, rng)
    hand_arrow(draw, (int(width * 0.38), int(height * 0.48)), (int(width * 0.49), int(height * 0.43)), rng)
    draw_upper_text(draw, (int(width * 0.66), int(height * 0.47)), "AHA!", label_font, fill=RED)


def _draw_general_layout(draw: ImageDraw.ImageDraw, beat: Beat, width: int, height: int, rng) -> None:
    title_font = marker_font(max(26, width // 26), bold=True)
    label_font = marker_font(max(20, width // 48), bold=True)
    title = short_label(beat.text_preview, "MAIN IDEA", words=4)
    _center_text(draw, (width // 2, int(height * 0.18)), title, title_font, fill=MIDNIGHT)

    draw_stick_figure(draw, (int(width * 0.21), int(height * 0.58)), width / 1920, rng)
    concept = (int(width * 0.42), int(height * 0.35), int(width * 0.78), int(height * 0.58))
    hand_rect(draw, concept, rng, width=max(5, width // 260), fill=LIGHT_GRAY)
    draw_upper_text(draw, (concept[0] + 40, concept[1] + 45), "CONCEPT", label_font)
    hand_arrow(draw, (int(width * 0.30), int(height * 0.49)), (concept[0] - 35, int(height * 0.47)), rng)
    hand_ellipse(draw, (concept[2] - 100, concept[1] - 45, concept[2] + 70, concept[1] + 85), rng, outline=RED, width=max(6, width // 260))


def _center_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    font,
    fill: tuple[int, int, int] = BLACK,
) -> None:
    bbox = draw.textbbox((0, 0), text.upper(), font=font)
    x = center[0] - (bbox[2] - bbox[0]) // 2
    y = center[1] - (bbox[3] - bbox[1]) // 2
    draw_upper_text(draw, (x, y), text, font, fill=fill)


def _timestamp_label(beat: Beat, compact: bool) -> str:
    if not compact:
        return f"{beat.start} -> {beat.end}"
    return f"{beat.start[3:8]}-{beat.end[3:8]}"
