from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .beats import build_beats
from .config import ensure_output_dirs, load_config, validate_config
from .contact_sheet import generate_contact_sheet
from .errors import InputFileError, PipelineError
from .images import generate_placeholder_images
from .manifest import write_beats, write_manifest, write_transcript_segments
from .render import get_audio_duration, render_video
from .srt_parser import parse_srt_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local YouTube video production pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--dry-run", action="store_true", help="Generate metadata/images/contact sheet but skip MP4 rendering.")
    args = parser.parse_args(argv)

    try:
        config = load_config(Path(args.config))
        validate_config(config)
        ensure_output_dirs(config)
        _validate_inputs(config.inputs.voiceover, config.inputs.transcript)

        audio_duration = None if args.dry_run else get_audio_duration(config.inputs.voiceover)
        segments = parse_srt_file(config.inputs.transcript)
        beats = build_beats(
            segments=segments,
            beat_config=config.beats,
            images_dir=config.outputs.images_dir,
            audio_duration=audio_duration,
            duration_mismatch_tolerance=config.timing.duration_mismatch_tolerance,
        )

        write_transcript_segments(config.outputs.data_dir / "transcript_segments.json", segments)
        write_beats(config.outputs.data_dir / "beats.json", beats)
        write_manifest(config.outputs.data_dir / "manifest.csv", beats)
        generate_placeholder_images(beats, config.video.width, config.video.height)
        generate_contact_sheet(beats, config.outputs.contact_sheet)

        if not args.dry_run:
            render_video(beats, config.inputs.voiceover, config.outputs.final_video, config.video.fps)

        print(f"Transcript segments: {len(segments)}")
        print(f"Visual beats: {len(beats)}")
        print(f"Dry run: {args.dry_run}")
        if not args.dry_run:
            print(f"Final video: {config.outputs.final_video}")
        return 0
    except PipelineError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1


def _validate_inputs(voiceover: Path, transcript: Path) -> None:
    if not voiceover.exists():
        raise InputFileError(f"Missing voiceover file: {voiceover}")
    if not transcript.exists():
        raise InputFileError(f"Missing transcript file: {transcript}")


if __name__ == "__main__":
    raise SystemExit(main())
