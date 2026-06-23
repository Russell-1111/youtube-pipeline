import json
from pathlib import Path

from youtube_pipeline.__main__ import main
from youtube_pipeline.script_quality import audit_script_quality_text


def write_config(path: Path) -> None:
    path.write_text(
        """
inputs:
  voiceover: input/voiceover.mp3
  transcript: input/transcript.srt

outputs:
  data_dir: data
  images_dir: assets/images
  contact_sheet: assets/contact_sheets/contact_sheet.png
  final_video: output/final_video.mp4

video:
  width: 1920
  height: 1080
  fps: 30

beats:
  min_duration: 6
  target_duration: 9
  max_duration: 12
  min_gap_beat_duration: 1.5
  min_intro_beat_duration: 1.0
  max_preview_chars: 80

timing:
  duration_mismatch_tolerance: 1.0
""".lstrip(),
        encoding="utf-8",
    )


def report_for(text: str, tmp_path: Path) -> dict:
    return audit_script_quality_text(text, tmp_path / "input" / "script.md", tmp_path)


def codes(report: dict) -> list[str]:
    return [finding["code"] for finding in report["findings"]]


def strong_script() -> str:
    return """
<!-- SCRIPT_BRIEF
title: The Time Trap
thumbnail_promise: Why saving time makes life feel smaller
viewer_problem: The viewer is busy, optimized, and still feels behind.
central_tension: We are obsessed with saving time, but we feel like we have less of it.
central_question: Why does productivity make free time feel more pressured?
viewer_payoff: The viewer will see the hidden cost of treating every hour like a debt.
emotional_arc: uneasy observation to sharper consequence to quiet release
open_loop: Saving time was supposed to create freedom, but it may be stealing attention.
final_takeaway: Time feels different when it is protected instead of optimized.
target_runtime: 10:00-11:00
target_viewer: reflective productivity viewers
-->

<!-- VOICEOVER_START -->

## 1. Hook / opening tension

You saved ten minutes by answering messages in line, another ten by eating at your desk, and another ten by turning rest into a task. But somehow the day did not become wider. It became tighter. The strange thing is that modern productivity promised freedom, yet the more time we save, the less free time seems to feel. Why does every shortcut leave behind the same pressure?

## 2. Setup / problem frame

The problem begins with the calendar. It looks neutral, almost harmless, a clean grid of hours waiting to be used well. But a modern schedule does more than organize work. It teaches you to see every blank space as unfinished business. Because the phone, the inbox, the meeting reminder, and the deadline all arrive inside the same frame, even rest starts to feel like a slot that must justify itself.

## 3. Core idea / explanation

This is why productivity becomes confusing. The mechanism is simple: when every hour is measured, every hour starts asking for proof. A saved minute no longer feels like a gift. It feels like a resource that should be reinvested. The system turns attention into debt, and then calls the repayment discipline. More importantly, the pressure is not only external. It becomes a private voice that asks whether you used the day correctly.

## 4. Deeper turn / consequence

But the deeper cost is identity. If the schedule defines what counts, then the parts of you that cannot be measured begin to look suspicious. Wandering, grieving, noticing, doing nothing, changing your mind, all of it starts to seem inefficient. Worse, the hidden cost is that you can become loyal to a version of yourself that only exists when it is producing. That is the trap: saving time can become another way to lose contact with your life.

## 5. Payoff / ending

The answer is not to abandon structure. The point is to stop treating every saved minute as proof that you owe the world more output. Freedom begins when time is protected instead of optimized. This is why the real question is not how much time you can save. It is whether the time you save still belongs to you.

<!-- VOICEOVER_END -->
""".strip()


def test_script_quality_default_cli_path_writes_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path / "config.yaml")
    script = tmp_path / "input" / "script.md"
    script.parent.mkdir()
    script.write_text(strong_script(), encoding="utf-8")

    result = main(["--script-quality-audit"])

    captured = capsys.readouterr()
    report = json.loads((tmp_path / "data" / "script_quality_report.json").read_text(encoding="utf-8"))
    assert result == 0
    assert "Script quality audit complete" in captured.out
    assert "Retention-risk score" in captured.out
    assert report["summary"]["input_path"] == "input/script.md"
    assert (tmp_path / "data" / "script_quality_report.md").exists()


def test_script_quality_explicit_cli_path_writes_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path / "config.yaml")
    script = tmp_path / "input" / "script.md"
    script.parent.mkdir()
    script.write_text(strong_script(), encoding="utf-8")

    result = main(["--script-quality-audit", "input/script.md"])

    report = json.loads((tmp_path / "data" / "script_quality_report.json").read_text(encoding="utf-8"))
    assert result == 0
    assert report["summary"]["input_path"] == "input/script.md"


def test_missing_script_writes_error_report_and_returns_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path / "config.yaml")

    result = main(["--script-quality-audit"])

    captured = capsys.readouterr()
    report = json.loads((tmp_path / "data" / "script_quality_report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert "Script quality audit error" in captured.err
    assert report["status"] == "error"
    assert report["quality_score"] == 0
    assert report["blocked_reasons"]
    assert codes(report) == ["MISSING_SCRIPT"]


def test_directory_input_is_unreadable_script_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path / "config.yaml")
    (tmp_path / "input" / "script.md").mkdir(parents=True)

    result = main(["--script-quality-audit", "input/script.md"])

    report = json.loads((tmp_path / "data" / "script_quality_report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert report["status"] == "error"
    assert codes(report) == ["UNREADABLE_SCRIPT"]


def test_json_top_level_keys_are_stable(tmp_path):
    report = report_for(strong_script(), tmp_path)

    assert list(report) == [
        "script_quality_schema_version",
        "status",
        "quality_score",
        "quality_label",
        "summary",
        "required_brief_fields",
        "sections",
        "hook_analysis",
        "retention_structure",
        "score_breakdown",
        "findings",
        "recommendations",
        "blocked_reasons",
    ]


def test_strong_script_passes_quality_gate(tmp_path):
    report = report_for(strong_script(), tmp_path)

    assert report["status"] == "pass"
    assert report["quality_label"] == "strong_script"
    assert report["quality_score"] >= 85
    assert report["score_breakdown"] == []


def test_missing_required_metadata_blocks(tmp_path):
    text = strong_script().replace("title: The Time Trap\n", "").replace(
        "central_tension: We are obsessed with saving time, but we feel like we have less of it.\n",
        "",
    )

    report = report_for(text, tmp_path)

    assert report["status"] == "blocked_weak_script"
    assert "MISSING_TITLE" in codes(report)
    assert "MISSING_CENTRAL_TENSION" in codes(report)
    assert report["blocked_reasons"]


def test_missing_voiceover_text_blocks(tmp_path):
    text = """
<!-- SCRIPT_BRIEF
title: The Time Trap
thumbnail_promise: Why saving time makes life feel smaller
viewer_problem: busy viewer
central_tension: Saving time makes time feel smaller
central_question: Why?
viewer_payoff: a better frame
open_loop: saved time feels stolen
final_takeaway: protect time instead of optimizing it
target_viewer: reflective viewers
-->
<!-- VOICEOVER_START -->
<!-- VOICEOVER_END -->
""".strip()

    report = report_for(text, tmp_path)

    assert report["status"] == "blocked_weak_script"
    assert "MISSING_VOICEOVER_TEXT" in codes(report)


def test_generic_intro_phrase_warns(tmp_path):
    text = strong_script().replace(
        "You saved ten minutes by answering messages in line",
        "In today's video, we're going to talk about productivity. You saved ten minutes by answering messages in line",
    )

    report = report_for(text, tmp_path)

    assert report["status"] in {"needs_rewrite", "blocked_weak_script"}
    assert "GENERIC_INTRO_PHRASE" in codes(report)
    assert any(item["reason"] == "Generic intro phrase detected" for item in report["score_breakdown"])


def test_weak_hook_blocks_through_low_score(tmp_path):
    weak = strong_script()
    weak = weak.replace(
        "You saved ten minutes by answering messages in line, another ten by eating at your desk, and another ten by turning rest into a task. But somehow the day did not become wider. It became tighter. The strange thing is that modern productivity promised freedom, yet the more time we save, the less free time seems to feel. Why does every shortcut leave behind the same pressure?",
        "First we need to define productivity and explain some background. Productivity is a useful topic for people in society today. Many people have different thoughts about productivity and time. It is very important to understand this topic before moving forward.",
    )

    report = report_for(weak, tmp_path)

    assert report["status"] == "blocked_weak_script"
    assert "WEAK_HOOK" in codes(report)
    assert "LOW_QUALITY_SCORE" in codes(report)


def test_missing_payoff_section_blocks(tmp_path):
    text = strong_script().replace("## 5. Payoff / ending", "## 5. Extra background")
    text = text.replace("The answer is not to abandon structure. The point is", "Another example is available. This background is")

    report = report_for(text, tmp_path)

    assert report["status"] == "blocked_weak_script"
    assert "MISSING_PAYOFF_SECTION" in codes(report)


def test_repetition_and_vague_language_warn(tmp_path):
    vague = strong_script().replace(
        "The problem begins with the calendar. It looks neutral, almost harmless, a clean grid of hours waiting to be used well. But a modern schedule does more than organize work. It teaches you to see every blank space as unfinished business. Because the phone, the inbox, the meeting reminder, and the deadline all arrive inside the same frame, even rest starts to feel like a slot that must justify itself.",
        "The problem is life. The problem is society. The problem is people. The problem is things. The problem is success. The problem is mindset. The problem is world. The problem is time. The problem is productivity. The problem is happiness. The problem is life society people things success mindset world time productivity happiness.",
    )

    report = report_for(vague, tmp_path)

    assert "REPETITIVE_SENTENCE_STARTS" in codes(report)
    assert "EXCESSIVE_VAGUE_LANGUAGE" in codes(report)


def test_findings_are_sorted_deterministically(tmp_path):
    report = report_for("plain text without metadata", tmp_path)
    sorted_codes = [
        "LOW_QUALITY_SCORE",
        "MISSING_CENTRAL_TENSION",
        "MISSING_FINAL_TAKEAWAY",
        "MISSING_OPEN_LOOP",
        "MISSING_PAYOFF_SECTION",
        "MISSING_THUMBNAIL_PROMISE",
        "MISSING_TITLE",
    ]

    assert codes(report)[:7] == sorted_codes


def test_script_quality_is_mutually_exclusive_with_existing_modes(tmp_path):
    write_config(tmp_path / "config.yaml")

    try:
        main(["--config", str(tmp_path / "config.yaml"), "--dry-run", "--script-quality-audit"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive modes")
