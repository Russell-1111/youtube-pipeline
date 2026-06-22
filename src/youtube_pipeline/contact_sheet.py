from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

from .images import render_whiteboard_frame
from .models import Beat
from .whiteboard import LIGHT_GRAY, WHITE, hand_rect, rng_for_beat


def generate_contact_sheet(beats: list[Beat], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = 3
    tile_width = 640
    tile_height = 360
    gap = 22
    rows = max(1, math.ceil(len(beats) / columns))
    sheet_width = columns * tile_width + (columns + 1) * gap
    sheet_height = rows * tile_height + (rows + 1) * gap
    sheet = Image.new("RGB", (sheet_width, sheet_height), WHITE)
    draw = ImageDraw.Draw(sheet)

    for index, beat in enumerate(beats):
        col = index % columns
        row = index // columns
        x = gap + col * (tile_width + gap)
        y = gap + row * (tile_height + gap)
        tile = render_whiteboard_frame(beat, tile_width, tile_height)
        sheet.paste(tile, (x, y))
        hand_rect(
            draw,
            (x - 4, y - 4, x + tile_width + 4, y + tile_height + 4),
            rng_for_beat(beat, "contact-sheet"),
            outline=LIGHT_GRAY,
            width=3,
        )

    sheet.save(output_path)
