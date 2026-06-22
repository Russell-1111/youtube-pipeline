import csv
import json

from youtube_pipeline.manifest import write_beats, write_manifest, write_transcript_segments
from youtube_pipeline.models import Beat, TranscriptSegment


def test_json_and_csv_output_shape(tmp_path):
    segment = TranscriptSegment(
        index=1,
        start="00:00:00,000",
        end="00:00:05,000",
        start_seconds=0.0,
        end_seconds=5.0,
        duration_seconds=5.0,
        text="Hello world",
    )
    beat = Beat(
        beat_number=1,
        beat_type="normal",
        start="00:00:00,000",
        end="00:00:05,000",
        start_seconds=0.0,
        end_seconds=5.0,
        duration_seconds=5.0,
        text_preview="Hello world",
        segment_indexes=[1],
        image_path="assets/images/beat_001.png",
    )

    segments_path = tmp_path / "transcript_segments.json"
    beats_path = tmp_path / "beats.json"
    manifest_path = tmp_path / "manifest.csv"

    write_transcript_segments(segments_path, [segment])
    write_beats(beats_path, [beat])
    write_manifest(manifest_path, [beat])

    segments_payload = json.loads(segments_path.read_text(encoding="utf-8"))
    beats_payload = json.loads(beats_path.read_text(encoding="utf-8"))
    manifest_rows = list(csv.DictReader(manifest_path.open(encoding="utf-8", newline="")))

    assert set(segments_payload[0]) == {
        "index",
        "start",
        "end",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "text",
    }
    assert set(beats_payload[0]) == {
        "beat_number",
        "beat_type",
        "start",
        "end",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "text_preview",
        "segment_indexes",
        "image_path",
    }
    assert manifest_rows[0] == {
        "beat_number": "1",
        "beat_type": "normal",
        "start": "00:00:00,000",
        "end": "00:00:05,000",
        "duration_seconds": "5.000",
        "image_path": "assets/images/beat_001.png",
        "text_preview": "Hello world",
    }
