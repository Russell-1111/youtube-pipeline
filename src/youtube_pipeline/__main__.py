from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .beats import build_beats
from .config import ensure_output_dirs, load_config, validate_config
from .contact_sheet import generate_contact_sheet
from .dense_beats import print_dense_plan_summary, run_dense_beat_plan
from .dense_beat_review import print_dense_review_summary, run_dense_beat_review
from .dense_images import print_dense_image_summary, run_dense_image_preparation
from .dense_prompts import print_dense_prompt_summary, run_dense_prompt_generation
from .dense_render import print_dense_render_summary, run_dense_preview_render
from .errors import InputFileError, PipelineError
from .generated_images import beats_with_generated_image_paths, load_and_validate_generated_images
from .images import generate_placeholder_images
from .manifest import write_beats, write_manifest, write_transcript_segments
from .production_audit import print_audit_summary, run_production_audit
from .prompts import write_image_prompts
from .render import get_audio_duration, render_video, render_video_kinetic
from .render_benchmark import run_kinetic_benchmark
from .render_ffmpeg import render_video_kinetic_ffmpeg
from .script_audit import print_script_audit_summary, run_script_audit
from .script_quality import print_script_quality_summary, run_script_quality_audit
from .srt_parser import parse_srt_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local YouTube video production pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dry-run", action="store_true", help="Generate metadata/images/contact sheet but skip MP4 rendering.")
    mode_group.add_argument("--generate-prompts", action="store_true", help="Generate image prompt records from data/beats.json.")
    mode_group.add_argument(
        "--validate-generated-images",
        action="store_true",
        help="Validate images in assets/generated_images against data/beats.json.",
    )
    mode_group.add_argument(
        "--use-generated-images",
        action="store_true",
        help="Render video using validated assets/generated_images/beat_NNN.png files.",
    )
    mode_group.add_argument(
        "--use-generated-images-kinetic",
        action="store_true",
        help="Render video using validated generated images with subtle deterministic motion.",
    )
    mode_group.add_argument(
        "--use-generated-images-kinetic-ffmpeg",
        action="store_true",
        help="Experimentally render validated generated images with FFmpeg kinetic motion.",
    )
    mode_group.add_argument(
        "--benchmark-kinetic-render",
        action="store_true",
        help="Render a sample kinetic benchmark under output/benchmarks/.",
    )
    mode_group.add_argument(
        "--production-audit",
        action="store_true",
        help="Write advisory production-readiness JSON and Markdown reports without rendering.",
    )
    mode_group.add_argument(
        "--plan-dense-beats",
        action="store_true",
        help="Preview optional dense beat planning reports without overwriting data/beats.json.",
    )
    mode_group.add_argument(
        "--review-dense-beats",
        action="store_true",
        help="Review dense preview beats without generating prompts or modifying production files.",
    )
    mode_group.add_argument(
        "--generate-dense-prompts",
        action="store_true",
        help="Generate preview-only dense image prompt artifacts without modifying production files.",
    )
    mode_group.add_argument(
        "--prepare-dense-images",
        action="store_true",
        help="Prepare dense preview image slots and validation reports without generating images.",
    )
    mode_group.add_argument(
        "--render-dense-preview",
        action="store_true",
        help="Safely render preview-only dense video to output/final_dense_preview.mp4.",
    )
    mode_group.add_argument(
        "--script-audit",
        nargs="?",
        const="input/script.md",
        default=None,
        help="Write pre-voiceover script runtime JSON and Markdown reports without rendering.",
    )
    mode_group.add_argument(
        "--script-quality-audit",
        nargs="?",
        const="input/script.md",
        default=None,
        help="Write pre-voiceover script quality heuristic JSON and Markdown reports without rendering.",
    )
    parser.add_argument(
        "--benchmark-renderer",
        choices=("ffmpeg", "moviepy"),
        default="ffmpeg",
        help="Renderer to use with --benchmark-kinetic-render.",
    )
    parser.add_argument(
        "--sample-beats",
        type=int,
        default=5,
        help="Number of beats to render for --benchmark-kinetic-render when --full is not set.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Render all beats for --benchmark-kinetic-render.",
    )
    parser.add_argument(
        "--stress-beats",
        type=int,
        default=None,
        help="Synthetic beat count for --benchmark-kinetic-render stress tests.",
    )
    parser.add_argument(
        "--stress-duration",
        type=float,
        default=None,
        help="Synthetic duration in seconds for --benchmark-kinetic-render stress tests.",
    )
    args = parser.parse_args(argv)

    try:
        config_path = Path(args.config)
        config = load_config(config_path)
        validate_config(config)

        beats_path = config.outputs.data_dir / "beats.json"
        transcript_segments_path = config.outputs.data_dir / "transcript_segments.json"
        generated_image_dir = config.outputs.images_dir.parent / "generated_images"
        base_dir = config_path.resolve().parent

        if args.script_audit is not None:
            script_path = Path(args.script_audit)
            if not script_path.is_absolute():
                script_path = base_dir / script_path
            result = run_script_audit(script_path, config.outputs.data_dir, base_dir)
            print_script_audit_summary(result, base_dir)
            if result.report["status"] == "error":
                for finding in result.report["findings"]:
                    if finding["severity"] == "error":
                        print(f"Script audit error: {finding['message']}", file=sys.stderr)
            return 1 if result.should_fail else 0

        if args.script_quality_audit is not None:
            script_path = Path(args.script_quality_audit)
            if not script_path.is_absolute():
                script_path = base_dir / script_path
            result = run_script_quality_audit(script_path, config.outputs.data_dir, base_dir)
            print_script_quality_summary(result, base_dir)
            if result.report["status"] == "error":
                for finding in result.report["findings"]:
                    if finding["severity"] == "error":
                        print(f"Script quality audit error: {finding['message']}", file=sys.stderr)
            return 1 if result.should_fail else 0

        if args.production_audit:
            result = run_production_audit(config, base_dir)
            print_audit_summary(result, base_dir)
            return 1 if result.has_core_errors else 0

        if args.plan_dense_beats:
            result = run_dense_beat_plan(config, base_dir)
            print_dense_plan_summary(result, base_dir)
            return 0

        if args.review_dense_beats:
            result = run_dense_beat_review(config, base_dir)
            print_dense_review_summary(result, base_dir)
            return 0

        if args.generate_dense_prompts:
            result = run_dense_prompt_generation(config, base_dir)
            print_dense_prompt_summary(result, base_dir)
            return 0

        if args.prepare_dense_images:
            result = run_dense_image_preparation(config, base_dir)
            print_dense_image_summary(result, base_dir)
            return 0

        if args.render_dense_preview:
            result = run_dense_preview_render(config, base_dir)
            print_dense_render_summary(result, base_dir)
            return 0

        if args.generate_prompts:
            payload = write_image_prompts(
                beats_path=beats_path,
                transcript_segments_path=transcript_segments_path,
                image_dir=generated_image_dir,
                output_path=config.outputs.data_dir / "image_prompts.json",
                base_dir=config_path.resolve().parent,
            )
            print(f"Image prompts: {len(payload['prompts'])}")
            print(f"Prompt file: {config.outputs.data_dir / 'image_prompts.json'}")
            print(f"Generated image directory: {generated_image_dir}")
            return 0

        if args.validate_generated_images:
            beats, report = load_and_validate_generated_images(beats_path, generated_image_dir)
            _print_validation_report(report)
            if not report.ok:
                return 1
            print(f"Generated image validation passed: {len(beats)} images")
            return 0

        if args.use_generated_images:
            beats, report = load_and_validate_generated_images(beats_path, generated_image_dir)
            _print_validation_report(report)
            if not report.ok:
                return 1
            _validate_voiceover(config.inputs.voiceover)
            generated_beats = beats_with_generated_image_paths(beats, generated_image_dir)
            output_path = config.outputs.final_video.parent / "final_video_generated.mp4"
            render_video(generated_beats, config.inputs.voiceover, output_path, config.video.fps)
            print(f"Generated image validation passed: {len(beats)} images")
            print(f"Final video: {output_path}")
            return 0

        if args.use_generated_images_kinetic:
            beats, report = load_and_validate_generated_images(beats_path, generated_image_dir)
            _print_validation_report(report)
            if not report.ok:
                return 1
            _validate_voiceover(config.inputs.voiceover)
            generated_beats = beats_with_generated_image_paths(beats, generated_image_dir)
            output_path = config.outputs.final_video.parent / "final_video_generated_kinetic.mp4"
            render_video_kinetic(
                generated_beats,
                config.inputs.voiceover,
                output_path,
                config.video.fps,
                config.video.width,
                config.video.height,
            )
            print(f"Generated image validation passed: {len(beats)} images")
            print(f"Final video: {output_path}")
            return 0

        if args.use_generated_images_kinetic_ffmpeg:
            beats, report = load_and_validate_generated_images(beats_path, generated_image_dir)
            _print_validation_report(report)
            if not report.ok:
                return 1
            _validate_voiceover(config.inputs.voiceover)
            generated_beats = beats_with_generated_image_paths(beats, generated_image_dir)
            output_path = config.outputs.final_video.parent / "final_video_generated_kinetic_ffmpeg.mp4"
            benchmark_dir = config.outputs.final_video.parent / "benchmarks"
            result = render_video_kinetic_ffmpeg(
                generated_beats,
                config.inputs.voiceover,
                output_path,
                config.video.fps,
                config.video.width,
                config.video.height,
                work_dir=benchmark_dir / "final_video_generated_kinetic_ffmpeg_segments",
                audio_duration_seconds=get_audio_duration(config.inputs.voiceover),
            )
            print(f"Generated image validation passed: {len(beats)} images")
            print(f"Final video: {output_path}")
            print(f"Renderer: experimental ffmpeg")
            print(f"Motion note: {result.motion_equivalence_note}")
            return 0

        if args.benchmark_kinetic_render:
            beats, report = load_and_validate_generated_images(beats_path, generated_image_dir)
            _print_validation_report(report)
            if not report.ok:
                return 1
            _validate_voiceover(config.inputs.voiceover)
            generated_beats = beats_with_generated_image_paths(beats, generated_image_dir)
            benchmark_report = run_kinetic_benchmark(
                renderer_name=args.benchmark_renderer,
                beats=generated_beats,
                audio_path=config.inputs.voiceover,
                output_dir=config.outputs.final_video.parent / "benchmarks",
                fps=config.video.fps,
                width=config.video.width,
                height=config.video.height,
                sample_beats=args.sample_beats,
                full=args.full,
                stress_beats=args.stress_beats,
                stress_duration=args.stress_duration,
            )
            print(f"Benchmark renderer: {benchmark_report.renderer_name}")
            print(f"Benchmark success: {benchmark_report.success}")
            print(f"Benchmark output: {benchmark_report.output_path}")
            print(f"Benchmark elapsed seconds: {benchmark_report.elapsed_seconds}")
            print(f"Benchmark report: {config.outputs.final_video.parent / 'benchmarks' / 'render_benchmark_report.json'}")
            if benchmark_report.warnings:
                print(f"Benchmark warnings: {len(benchmark_report.warnings)}")
            if benchmark_report.errors:
                for error in benchmark_report.errors:
                    print(f"Benchmark error: {error}", file=sys.stderr)
            return 0 if benchmark_report.success else 1

        return _run_v1_pipeline(config, dry_run=args.dry_run)
    except PipelineError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1


def _run_v1_pipeline(config, dry_run: bool) -> int:
    ensure_output_dirs(config)
    _validate_inputs(config.inputs.voiceover, config.inputs.transcript)

    audio_duration = None if dry_run else get_audio_duration(config.inputs.voiceover)
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

    if not dry_run:
        render_video(beats, config.inputs.voiceover, config.outputs.final_video, config.video.fps)

    print(f"Transcript segments: {len(segments)}")
    print(f"Visual beats: {len(beats)}")
    print(f"Dry run: {dry_run}")
    if not dry_run:
        print(f"Final video: {config.outputs.final_video}")
    return 0


def _print_validation_report(report) -> None:
    for warning in report.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    for error in report.errors:
        print(f"Validation error: {error}", file=sys.stderr)


def _validate_inputs(voiceover: Path, transcript: Path) -> None:
    _validate_voiceover(voiceover)
    if not transcript.exists():
        raise InputFileError(f"Missing transcript file: {transcript}")


def _validate_voiceover(voiceover: Path) -> None:
    if not voiceover.exists():
        raise InputFileError(f"Missing voiceover file: {voiceover}")


if __name__ == "__main__":
    raise SystemExit(main())
