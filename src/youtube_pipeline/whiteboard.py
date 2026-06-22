from __future__ import annotations

import hashlib
import math
import random
import re
import textwrap
from typing import Iterable

from PIL import ImageDraw, ImageFont

from .models import Beat

BLACK = (18, 24, 32)
MIDNIGHT = (28, 48, 80)
YELLOW = (246, 197, 83)
RED = (190, 64, 64)
GRAY = (214, 220, 228)
DARK_GRAY = (82, 91, 104)
LIGHT_GRAY = (242, 244, 247)
WHITE = (255, 255, 255)

KEYWORD_LAYOUTS = [
    (("history", "timeline", "years", "year", "past", "future"), "timeline"),
    (("switch", "button", "turn on", "turn off", "toggle"), "switch"),
    (("sleep", "bed", "night", "dream", "rest"), "bed"),
    (("learn", "book", "know", "study", "read"), "book"),
    (("before", "after", "versus", "vs", "compare"), "comparison"),
    (("because", "cause", "effect", "result", "therefore"), "cause_effect"),
    (("home", "house", "room"), "house"),
    (("idea", "light", "bulb"), "idea"),
]


def rng_for_beat(beat: Beat, salt: str = "") -> random.Random:
    key = f"{beat.beat_number}|{beat.beat_type}|{beat.start}|{beat.end}|{beat.text_preview}|{salt}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def select_layout(beat: Beat) -> str:
    if beat.beat_type == "intro":
        return "title"
    if beat.beat_type == "gap":
        return "pause"

    text = beat.text_preview.lower()
    for keywords, layout in KEYWORD_LAYOUTS:
        if any(keyword in text for keyword in keywords):
            return layout
    return "general"


def marker_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_upper_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int] = BLACK,
) -> None:
    draw.text(xy, text.upper(), fill=fill, font=font)


def draw_wrapped_upper_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_chars: int,
    fill: tuple[int, int, int] = BLACK,
    max_lines: int = 3,
    line_gap: float = 1.2,
) -> None:
    lines = textwrap.wrap(text.upper(), width=max_chars)[:max_lines] or [""]
    y = xy[1]
    line_height = int(getattr(font, "size", 24) * line_gap)
    for line in lines:
        draw.text((xy[0], y), line, fill=fill, font=font)
        y += line_height


def short_label(text: str, fallback: str = "MAIN IDEA", words: int = 4) -> str:
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = [token for token in clean.split() if len(token) > 1]
    if not tokens:
        return fallback
    return " ".join(tokens[:words]).upper()


def preview_text(text: str, max_chars: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def hand_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    rng: random.Random,
    fill: tuple[int, int, int] = BLACK,
    width: int = 7,
    passes: int = 2,
) -> None:
    del rng, passes
    draw.line([start, end], fill=fill, width=width)


def hand_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    rng: random.Random,
    outline: tuple[int, int, int] = BLACK,
    width: int = 7,
    fill: tuple[int, int, int] | None = None,
) -> None:
    del rng
    radius = max(0, min(24, width * 2))
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def hand_ellipse(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    rng: random.Random,
    outline: tuple[int, int, int] = BLACK,
    width: int = 7,
    fill: tuple[int, int, int] | None = None,
) -> None:
    del rng
    draw.ellipse(xy, fill=fill, outline=outline, width=width)


def hand_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    rng: random.Random,
    fill: tuple[int, int, int] = RED,
    width: int = 9,
) -> None:
    hand_line(draw, start, end, rng, fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_length = width * 4.2
    left = (
        end[0] + math.cos(angle + math.pi * 0.78) * head_length,
        end[1] + math.sin(angle + math.pi * 0.78) * head_length,
    )
    right = (
        end[0] + math.cos(angle - math.pi * 0.78) * head_length,
        end[1] + math.sin(angle - math.pi * 0.78) * head_length,
    )
    draw.polygon([end, left, right], fill=fill)


def hand_x(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    rng: random.Random,
    fill: tuple[int, int, int] = RED,
    width: int = 10,
) -> None:
    x1, y1, x2, y2 = xy
    hand_line(draw, (x1, y1), (x2, y2), rng, fill=fill, width=width)
    hand_line(draw, (x1, y2), (x2, y1), rng, fill=fill, width=width)


def scribble_fill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    rng: random.Random,
    fill: tuple[int, int, int] = LIGHT_GRAY,
    width: int = 4,
    spacing: int = 18,
) -> None:
    del rng, width, spacing
    draw.rounded_rectangle(xy, radius=18, fill=fill)


def draw_stick_figure(draw: ImageDraw.ImageDraw, center: tuple[int, int], scale: float, rng: random.Random) -> None:
    cx, cy = center
    head = int(44 * scale)
    body = int(110 * scale)
    arm = int(78 * scale)
    leg = int(92 * scale)
    stroke = max(4, int(7 * scale))
    hand_ellipse(draw, (cx - head, cy - body - head * 2, cx + head, cy - body), rng, width=stroke, fill=YELLOW)
    hand_line(draw, (cx, cy - body), (cx, cy), rng, width=stroke)
    hand_line(draw, (cx - arm, cy - body // 2), (cx + arm, cy - body // 2), rng, width=stroke)
    hand_line(draw, (cx, cy), (cx - leg, cy + leg), rng, width=stroke)
    hand_line(draw, (cx, cy), (cx + leg, cy + leg), rng, width=stroke)


def draw_book(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], rng: random.Random) -> None:
    x1, y1, x2, y2 = xy
    hand_rect(draw, (x1, y1, x2, y2), rng, width=7, fill=LIGHT_GRAY)
    mid = (x1 + x2) // 2
    hand_line(draw, (mid, y1), (mid, y2), rng, width=5)
    for offset in (35, 70, 105):
        hand_line(draw, (x1 + 30, y1 + offset), (mid - 25, y1 + offset), rng, fill=DARK_GRAY, width=4, passes=1)
        hand_line(draw, (mid + 25, y1 + offset), (x2 - 30, y1 + offset), rng, fill=DARK_GRAY, width=4, passes=1)


def draw_bed(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], rng: random.Random) -> None:
    x1, y1, x2, y2 = xy
    bed_y = y2 - 80
    hand_rect(draw, (x1, bed_y, x2, y2 - 20), rng, fill=GRAY, width=7)
    hand_rect(draw, (x1 + 35, bed_y - 60, x1 + 170, bed_y), rng, fill=WHITE, width=6)
    hand_line(draw, (x1, y2 - 20), (x1, y2 + 30), rng, width=6)
    hand_line(draw, (x2, y2 - 20), (x2, y2 + 30), rng, width=6)
    draw_moon(draw, (x2 - 125, y1 + 60), 55, rng)


def draw_moon(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, rng: random.Random) -> None:
    cx, cy = center
    hand_ellipse(draw, (cx - radius, cy - radius, cx + radius, cy + radius), rng, outline=BLACK, width=6, fill=YELLOW)
    draw.ellipse((cx - radius // 5, cy - radius, cx + radius, cy + radius), fill=WHITE)


def draw_sun(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, rng: random.Random) -> None:
    cx, cy = center
    hand_ellipse(draw, (cx - radius, cy - radius, cx + radius, cy + radius), rng, outline=BLACK, width=6, fill=YELLOW)
    for angle in range(0, 360, 45):
        start = (cx + math.cos(math.radians(angle)) * (radius + 14), cy + math.sin(math.radians(angle)) * (radius + 14))
        end = (cx + math.cos(math.radians(angle)) * (radius + 54), cy + math.sin(math.radians(angle)) * (radius + 54))
        hand_line(draw, start, end, rng, width=5)


def draw_house(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], rng: random.Random) -> None:
    x1, y1, x2, y2 = xy
    roof_y = y1 + (y2 - y1) // 3
    hand_line(draw, (x1, roof_y), ((x1 + x2) // 2, y1), rng, width=8)
    hand_line(draw, ((x1 + x2) // 2, y1), (x2, roof_y), rng, width=8)
    hand_rect(draw, (x1 + 25, roof_y, x2 - 25, y2), rng, fill=LIGHT_GRAY, width=7)
    hand_rect(draw, ((x1 + x2) // 2 - 38, y2 - 110, (x1 + x2) // 2 + 38, y2), rng, fill=LIGHT_GRAY, width=5)
    hand_rect(draw, (x1 + 65, roof_y + 60, x1 + 145, roof_y + 135), rng, fill=WHITE, width=5)


def draw_light_bulb(draw: ImageDraw.ImageDraw, center: tuple[int, int], scale: float, rng: random.Random) -> None:
    cx, cy = center
    radius = int(72 * scale)
    hand_ellipse(draw, (cx - radius, cy - radius, cx + radius, cy + radius), rng, width=max(5, int(7 * scale)), fill=YELLOW)
    hand_rect(draw, (cx - radius // 2, cy + radius - 5, cx + radius // 2, cy + radius + int(48 * scale)), rng, width=max(4, int(6 * scale)))
    for angle in range(0, 360, 60):
        start = (cx + math.cos(math.radians(angle)) * (radius + 18), cy + math.sin(math.radians(angle)) * (radius + 18))
        end = (cx + math.cos(math.radians(angle)) * (radius + 55), cy + math.sin(math.radians(angle)) * (radius + 55))
        hand_line(draw, start, end, rng, fill=RED if angle in (0, 180) else BLACK, width=max(4, int(5 * scale)))


def draw_switch(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], rng: random.Random, on: bool = False) -> None:
    x1, y1, x2, y2 = xy
    hand_rect(draw, xy, rng, width=8, fill=GRAY)
    knob_radius = (y2 - y1) // 2 - 18
    knob_cx = x2 - knob_radius - 24 if on else x1 + knob_radius + 24
    knob_cy = (y1 + y2) // 2
    hand_ellipse(draw, (knob_cx - knob_radius, knob_cy - knob_radius, knob_cx + knob_radius, knob_cy + knob_radius), rng, width=7, fill=YELLOW if on else WHITE)
    if not on:
        hand_x(draw, (x2 + 35, y1 - 25, x2 + 125, y2 + 25), rng, width=10)


def draw_timeline(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], rng: random.Random) -> None:
    x1, y1, x2, y2 = xy
    mid_y = (y1 + y2) // 2
    hand_line(draw, (x1, mid_y), (x2, mid_y), rng, width=8)
    for idx, x in enumerate(_linspace(x1 + 50, x2 - 50, 4)):
        hand_ellipse(draw, (int(x) - 22, mid_y - 22, int(x) + 22, mid_y + 22), rng, outline=RED if idx == 2 else MIDNIGHT, width=6, fill=YELLOW if idx == 2 else WHITE)
        hand_line(draw, (x, mid_y + 35), (x, mid_y + 90), rng, width=4)


def _linspace(start: float, end: float, count: int) -> Iterable[float]:
    if count == 1:
        yield start
        return
    step = (end - start) / (count - 1)
    for index in range(count):
        yield start + step * index


def _jitter_point(point: tuple[float, float], rng: random.Random, amount: float) -> tuple[float, float]:
    del rng, amount
    return point
