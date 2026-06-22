import pytest

from youtube_pipeline.errors import SRTParseError
from youtube_pipeline.srt_parser import parse_srt


def test_valid_srt_parsing():
    segments = parse_srt(
        """1
00:00:00,000 --> 00:00:02,500
Hello world.

2
00:00:03,000 --> 00:00:05,000
Next caption.
"""
    )

    assert len(segments) == 2
    assert segments[0].index == 1
    assert segments[0].start_seconds == 0.0
    assert segments[0].end_seconds == 2.5
    assert segments[0].duration_seconds == 2.5
    assert segments[0].text == "Hello world."


def test_multi_line_captions_are_joined():
    segments = parse_srt(
        """1
00:00:00,000 --> 00:00:02,000
First line
Second line
"""
    )

    assert segments[0].text == "First line Second line"


def test_invalid_timestamp_is_rejected():
    with pytest.raises(SRTParseError, match="Invalid timestamp"):
        parse_srt(
            """1
00:00:AA,000 --> 00:00:02,000
Bad timestamp.
"""
        )


def test_end_before_start_is_rejected():
    with pytest.raises(SRTParseError, match="end timestamp"):
        parse_srt(
            """1
00:00:03,000 --> 00:00:02,000
Backwards.
"""
        )


def test_non_monotonic_ordering_is_rejected():
    with pytest.raises(SRTParseError, match="non-monotonic"):
        parse_srt(
            """1
00:00:00,000 --> 00:00:05,000
First.

2
00:00:04,000 --> 00:00:06,000
Overlaps earlier segment.
"""
        )
