import json

from youtube_pipeline.render_benchmark import RenderBenchmarkReport, write_benchmark_report


def test_write_benchmark_report_includes_required_fields(tmp_path):
    report = RenderBenchmarkReport(
        renderer_name="ffmpeg",
        elapsed_seconds=1.5,
        beat_count=5,
        video_duration_seconds=50.0,
        audio_duration_seconds=50.0,
        output_path="output/benchmarks/ffmpeg_sample_5beats.mp4",
        output_size_bytes=1234,
        width=1920,
        height=1080,
        fps=30.0,
        audio_present=True,
        duration_delta_seconds=0.0,
        success=True,
        warnings=["experimental"],
        errors=[],
    )

    path = tmp_path / "render_benchmark_report.json"
    write_benchmark_report(path, report)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "renderer_name",
        "elapsed_seconds",
        "beat_count",
        "video_duration_seconds",
        "audio_duration_seconds",
        "output_path",
        "output_size_bytes",
        "width",
        "height",
        "fps",
        "audio_present",
        "duration_delta_seconds",
        "success",
        "warnings",
        "errors",
    }
    assert payload["renderer_name"] == "ffmpeg"
    assert payload["success"] is True
