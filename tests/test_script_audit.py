import json
from pathlib import Path

from youtube_pipeline.__main__ import main
from youtube_pipeline.script_audit import ScriptAuditRules, audit_script_text


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


def report_for(text: str, tmp_path: Path, rules: ScriptAuditRules | None = None) -> dict:
    return audit_script_text(text, tmp_path / "input" / "script.md", tmp_path, rules)


def codes(report: dict) -> list[str]:
    return [finding["code"] for finding in report["findings"]]


def five_section_script(section_3_words: int = 4) -> str:
    return "\n\n".join(
        [
            "## 1. Hook\nSmall opening line.",
            "## 2. Setup\nSmall setup line.",
            f"## 3. Core\n{'word ' * section_3_words}",
            "## 4. Turn\nSmall turn line.",
            "## 5. Payoff\nSmall ending line.",
        ]
    )


def test_script_audit_default_cli_path_writes_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path / "config.yaml")
    script = tmp_path / "input" / "script.md"
    script.parent.mkdir()
    script.write_text(five_section_script(), encoding="utf-8")

    result = main(["--script-audit"])

    captured = capsys.readouterr()
    report = json.loads((tmp_path / "data" / "script_audit_report.json").read_text(encoding="utf-8"))
    assert result == 0
    assert "Script audit complete" in captured.out
    assert report["summary"]["input_path"] == "input/script.md"
    assert (tmp_path / "data" / "script_audit_report.md").exists()


def test_script_audit_explicit_cli_path_writes_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path / "config.yaml")
    script = tmp_path / "input" / "script.md"
    script.parent.mkdir()
    script.write_text(five_section_script(), encoding="utf-8")

    result = main(["--script-audit", "input/script.md"])

    report = json.loads((tmp_path / "data" / "script_audit_report.json").read_text(encoding="utf-8"))
    assert result == 0
    assert report["summary"]["input_path"] == "input/script.md"


def test_missing_script_writes_error_report_and_returns_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path / "config.yaml")

    result = main(["--script-audit"])

    captured = capsys.readouterr()
    report = json.loads((tmp_path / "data" / "script_audit_report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert "Script audit error" in captured.err
    assert report["status"] == "error"
    assert codes(report) == ["MISSING_SCRIPT"]


def test_directory_input_is_unreadable_script_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path / "config.yaml")
    (tmp_path / "input" / "script.md").mkdir(parents=True)

    result = main(["--script-audit", "input/script.md"])

    report = json.loads((tmp_path / "data" / "script_audit_report.json").read_text(encoding="utf-8"))
    assert result == 1
    assert report["status"] == "error"
    assert codes(report) == ["UNREADABLE_SCRIPT"]


def test_json_top_level_keys_are_stable(tmp_path):
    report = report_for(five_section_script(), tmp_path)

    assert list(report) == [
        "script_audit_schema_version",
        "status",
        "summary",
        "rules",
        "sections",
        "pause_tags",
        "findings",
        "suggested_cuts",
    ]


def test_markdown_headings_are_excluded_from_word_count(tmp_path):
    report = report_for("## 1. Hook\nspoken words only", tmp_path)

    assert report["summary"]["word_count"] == 3
    assert "SECTION_COUNT_MISMATCH" in codes(report)


def test_pause_tags_are_removed_before_word_count(tmp_path):
    report = report_for("## 1. Hook\nhello <#0.3#> world", tmp_path)

    assert report["summary"]["word_count"] == 2
    assert report["summary"]["pause_tag_count"] == 1
    assert report["summary"]["explicit_pause_seconds"] == 0.3


def test_voiceover_markers_limit_audited_text(tmp_path):
    report = report_for(
        "outside " * 100
        + "<!-- VOICEOVER_START -->\n## 1. Hook\ninside words\n<!-- VOICEOVER_END -->\n"
        + "outside " * 100,
        tmp_path,
    )

    assert report["summary"]["word_count"] == 2


def test_missing_voiceover_markers_fall_back_to_whole_file(tmp_path):
    report = report_for("outside words\ninside words", tmp_path)

    assert report["summary"]["word_count"] == 4


def test_invalid_pause_like_tags_warn_without_crashing(tmp_path):
    report = report_for("## 1. Hook\nhello <#abc#> world", tmp_path)

    assert report["status"] == "needs_cut"
    assert "INVALID_PAUSE_TAG" in codes(report)


def test_runtime_warning_and_hard_cap_status(tmp_path):
    warning_rules = ScriptAuditRules(wpm=60, warning_seconds=3, hard_cap_seconds=100, expected_sections=1)
    blocked_rules = ScriptAuditRules(wpm=60, warning_seconds=3, hard_cap_seconds=4, expected_sections=1)

    warning = report_for("one two three four", tmp_path, warning_rules)
    blocked = report_for("one two three four five", tmp_path, blocked_rules)

    assert warning["status"] == "needs_cut"
    assert "RUNTIME_WARNING" in codes(warning)
    assert blocked["status"] == "blocked_too_long"
    assert "RUNTIME_HARD_CAP_EXCEEDED" in codes(blocked)


def test_word_warning_and_block_status(tmp_path):
    warning = report_for("one two three", tmp_path, ScriptAuditRules(word_warning=2, word_block=10, expected_sections=1))
    blocked = report_for("one two three", tmp_path, ScriptAuditRules(word_warning=2, word_block=2, expected_sections=1))

    assert "WORD_COUNT_WARNING" in codes(warning)
    assert "WORD_COUNT_BLOCK" in codes(blocked)
    assert blocked["status"] == "blocked_too_long"


def test_pause_total_warning_and_block_status(tmp_path):
    warning = report_for("hello <#1.0#> world", tmp_path, ScriptAuditRules(pause_warning_seconds=0.5, pause_block_seconds=5, expected_sections=1))
    blocked = report_for("hello <#1.0#> world", tmp_path, ScriptAuditRules(pause_warning_seconds=0.5, pause_block_seconds=0.5, expected_sections=1))

    assert "PAUSE_SECONDS_WARNING" in codes(warning)
    assert "PAUSE_SECONDS_BLOCK" in codes(blocked)
    assert blocked["status"] == "blocked_too_long"


def test_single_pause_over_two_seconds_blocks(tmp_path):
    report = report_for("hello <#2.1#> world", tmp_path, ScriptAuditRules(expected_sections=1))

    assert report["status"] == "blocked_too_long"
    assert "LONG_PAUSE_TAG_BLOCK" in codes(report)


def test_pause_tags_within_twelve_spoken_words_warn(tmp_path):
    report = report_for("hello <#0.3#> close words <#0.3#> world", tmp_path, ScriptAuditRules(expected_sections=1))

    assert "PAUSE_TAGS_TOO_CLOSE" in codes(report)


def test_more_than_two_pause_tags_in_paragraph_warns(tmp_path):
    report = report_for("hello <#0.3#> one <#0.3#> two <#0.3#> world", tmp_path, ScriptAuditRules(expected_sections=1))

    assert "TOO_MANY_PAUSES_IN_PARAGRAPH" in codes(report)


def test_suggested_cuts_identify_most_over_budget_section(tmp_path):
    report = report_for(five_section_script(section_3_words=600), tmp_path)

    assert report["suggested_cuts"]
    assert report["suggested_cuts"][0]["section_number"] == 3
    assert report["suggested_cuts"][0]["seconds_to_cut"] > 0
    assert report["suggested_cuts"][0]["approx_words_to_cut"] > 0


def test_markdown_report_is_readable_not_raw_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path / "config.yaml")
    script = tmp_path / "input" / "script.md"
    script.parent.mkdir()
    script.write_text(five_section_script(), encoding="utf-8")

    result = main(["--script-audit"])

    markdown = (tmp_path / "data" / "script_audit_report.md").read_text(encoding="utf-8")
    assert result == 0
    assert markdown.startswith("# Script Audit Report")
    assert '"summary"' not in markdown


def test_script_audit_is_mutually_exclusive_with_existing_modes(tmp_path):
    write_config(tmp_path / "config.yaml")

    try:
        main(["--config", str(tmp_path / "config.yaml"), "--dry-run", "--script-audit"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject mutually exclusive modes")
