from __future__ import annotations


def seconds_to_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("Timestamp seconds cannot be negative.")
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def format_range(start_seconds: float, end_seconds: float) -> str:
    return f"{seconds_to_timestamp(start_seconds)} --> {seconds_to_timestamp(end_seconds)}"
