from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from .errors import PipelineError

SCRIPT_AUDIT_SCHEMA_VERSION = 1
JSON_REPORT_NAME = "script_audit_report.json"
MARKDOWN_REPORT_NAME = "script_audit_report.md"

VOICEOVER_START = "<!-- VOICEOVER_START -->"
VOICEOVER_END = "<!-- VOICEOVER_END -->"

VALID_PAUSE_RE = re.compile(r"<#(?P<seconds>\d+\.\d+)#>")
PAUSE_LIKE_RE = re.compile(r"<#[^>\s]*#>")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\'-][A-Za-z0-9]+)*")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<label>.+?)\s*$")
SECTION_MARKER_RE = re.compile(
    r"^\s*(?:(?:SECTION|Section)\s+(?P<section>\d+)|(?:PART|Part)\s+(?P<part>\d+))"
    r"(?:\s*[:.\-]\s*(?P<label>.*))?\s*$"
)
NUMBERED_HEADING_RE = re.compile(r"^\s*(?P<number>\d+)[.)]\s+(?P<label>.+?)\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
METADATA_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{0,40}:\s+\S+")

SEVERITY_ORDER = {
    "error": 0,
    "warning": 1,
    "info": 2,
    "recommendation": 3,
}

SECTION_BUDGETS = (
    ("Hook / opening tension", 45.0, 60.0),
    ("Setup / problem frame", 90.0, 120.0),
    ("Core idea / explanation", 150.0, 180.0),
    ("Deeper turn / consequence", 150.0, 180.0),
    ("Payoff / ending", 90.0, 120.0),
)


@dataclass(frozen=True)
class ScriptAuditRules:
    target_min_seconds: float = 600.0
    target_max_seconds: float = 660.0
    warning_seconds: float = 645.0
    hard_cap_seconds: float = 690.0
    wpm: int = 145
    word_warning: int = 1550
    word_block: int = 1700
    pause_warning_seconds: float = 60.0
    pause_block_seconds: float = 90.0
    pause_percentage_warning: float = 12.0
    pause_tag_warning: int = 30
    pause_tag_block: int = 45
    expected_sections: int = 5


@dataclass(frozen=True)
class ScriptAuditResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path

    @property
    def should_fail(self) -> bool:
        return self.report["status"] in {"blocked_too_long", "error"}


def run_script_audit(script_path: Path, data_dir: Path, base_dir: Path) -> ScriptAuditResult:
    rules = ScriptAuditRules()
    json_path = data_dir / JSON_REPORT_NAME
    markdown_path = data_dir / MARKDOWN_REPORT_NAME

    if not script_path.exists():
        report = _error_report(script_path, base_dir, rules, "MISSING_SCRIPT", f"Missing script file: {script_path}")
        return _write_reports(report, json_path, markdown_path)
    if not script_path.is_file():
        report = _error_report(script_path, base_dir, rules, "UNREADABLE_SCRIPT", f"Script path is not a file: {script_path}")
        return _write_reports(report, json_path, markdown_path)

    try:
        raw_text = script_path.read_text(encoding="utf-8")
    except OSError as exc:
        report = _error_report(script_path, base_dir, rules, "UNREADABLE_SCRIPT", f"Unreadable script file: {script_path} ({exc})")
        return _write_reports(report, json_path, markdown_path)
    except UnicodeDecodeError as exc:
        report = _error_report(script_path, base_dir, rules, "UNREADABLE_SCRIPT", f"Script file must be UTF-8 text: {script_path} ({exc})")
        return _write_reports(report, json_path, markdown_path)

    report = audit_script_text(raw_text, script_path, base_dir, rules)
    return _write_reports(report, json_path, markdown_path)


def audit_script_text(raw_text: str, script_path: Path, base_dir: Path, rules: ScriptAuditRules | None = None) -> dict[str, Any]:
    rules = rules or ScriptAuditRules()
    findings: list[dict[str, Any]] = []
    voiceover_text = _voiceover_region(raw_text)
    section_inputs = _detect_sections(voiceover_text)

    if len(section_inputs) != rules.expected_sections:
        findings.append(
            _finding(
                "warning",
                "SECTION_COUNT_MISMATCH",
                f"Detected {len(section_inputs)} section(s); expected {rules.expected_sections}.",
                recommendation="Use five explicit script sections before generating voiceover.",
            )
        )

    sections = []
    all_valid_pauses: list[dict[str, Any]] = []
    all_invalid_pauses: list[dict[str, Any]] = []
    excluded_line_count = 0
    for index, section in enumerate(section_inputs, start=1):
        analysis = _analyze_section(index, section["label"], section["text"], rules)
        sections.append(analysis["section"])
        all_valid_pauses.extend(analysis["valid_pauses"])
        all_invalid_pauses.extend(analysis["invalid_pauses"])
        findings.extend(analysis["findings"])
        excluded_line_count += analysis["excluded_line_count"]

    if excluded_line_count:
        findings.append(
            _finding(
                "info",
                "NON_SPOKEN_TEXT_EXCLUDED",
                f"Excluded {excluded_line_count} non-spoken Markdown/script-structure line(s) from word count.",
            )
        )

    word_count = sum(section["word_count"] for section in sections)
    explicit_pause_seconds = round(sum(section["explicit_pause_seconds"] for section in sections), 3)
    estimated_spoken_seconds = round(word_count / rules.wpm * 60, 3) if rules.wpm else 0.0
    estimated_total_seconds = round(estimated_spoken_seconds + explicit_pause_seconds, 3)
    pause_percentage = round((explicit_pause_seconds / estimated_total_seconds * 100), 1) if estimated_total_seconds else 0.0
    pause_tag_count = len(all_valid_pauses)
    longest_pause_seconds = round(max((pause["seconds"] for pause in all_valid_pauses), default=0.0), 3)
    average_pause_seconds = round(explicit_pause_seconds / pause_tag_count, 3) if pause_tag_count else 0.0

    _add_total_findings(
        findings=findings,
        rules=rules,
        estimated_total_seconds=estimated_total_seconds,
        word_count=word_count,
        explicit_pause_seconds=explicit_pause_seconds,
        pause_percentage=pause_percentage,
        pause_tag_count=pause_tag_count,
        longest_pause_seconds=longest_pause_seconds,
        average_pause_seconds=average_pause_seconds,
    )
    _add_pause_findings(findings, all_valid_pauses, all_invalid_pauses)
    _add_section_findings(findings, sections, rules)

    sorted_findings = _sort_findings(findings)
    status = _status(sorted_findings)
    report = {
        "script_audit_schema_version": SCRIPT_AUDIT_SCHEMA_VERSION,
        "status": status,
        "summary": {
            "input_path": _display_path(script_path, base_dir),
            "word_count": word_count,
            "estimated_spoken_seconds": estimated_spoken_seconds,
            "explicit_pause_seconds": explicit_pause_seconds,
            "estimated_total_seconds": estimated_total_seconds,
            "estimated_total_timecode": _timecode(estimated_total_seconds),
            "pause_percentage": pause_percentage,
            "pause_tag_count": pause_tag_count,
            "longest_pause_seconds": longest_pause_seconds,
            "section_count": len(sections),
            "wpm": rules.wpm,
        },
        "rules": asdict(rules),
        "sections": sections,
        "pause_tags": {
            "count": pause_tag_count,
            "total_seconds": explicit_pause_seconds,
            "longest_seconds": longest_pause_seconds,
            "average_seconds": average_pause_seconds,
            "invalid_count": len(all_invalid_pauses),
            "by_duration": _pause_duration_distribution(all_valid_pauses),
        },
        "findings": sorted_findings,
        "suggested_cuts": _suggested_cuts(sections, rules),
    }
    return report


def print_script_audit_summary(result: ScriptAuditResult, base_dir: Path) -> None:
    summary = result.report["summary"]
    print("Script audit complete")
    print(f"Status: {result.report['status']}")
    print(f"Estimated runtime: {summary['estimated_total_timecode']} ({summary['estimated_total_seconds']:.1f}s)")
    print(f"Words: {summary['word_count']}")
    print(f"Explicit pauses: {summary['explicit_pause_seconds']:.1f}s ({summary['pause_percentage']:.1f}%)")
    print("Reports written:")
    print(f"- {_display_path(result.json_path, base_dir)}")
    print(f"- {_display_path(result.markdown_path, base_dir)}")


def _error_report(script_path: Path, base_dir: Path, rules: ScriptAuditRules, code: str, message: str) -> dict[str, Any]:
    return {
        "script_audit_schema_version": SCRIPT_AUDIT_SCHEMA_VERSION,
        "status": "error",
        "summary": {
            "input_path": _display_path(script_path, base_dir),
            "word_count": 0,
            "estimated_spoken_seconds": 0.0,
            "explicit_pause_seconds": 0.0,
            "estimated_total_seconds": 0.0,
            "estimated_total_timecode": "00:00:00",
            "pause_percentage": 0.0,
            "pause_tag_count": 0,
            "longest_pause_seconds": 0.0,
            "section_count": 0,
            "wpm": rules.wpm,
        },
        "rules": asdict(rules),
        "sections": [],
        "pause_tags": {
            "count": 0,
            "total_seconds": 0.0,
            "longest_seconds": 0.0,
            "average_seconds": 0.0,
            "invalid_count": 0,
            "by_duration": [],
        },
        "findings": [_finding("error", code, message)],
        "suggested_cuts": [],
    }


def _voiceover_region(text: str) -> str:
    start = text.find(VOICEOVER_START)
    end = text.find(VOICEOVER_END)
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start + len(VOICEOVER_START) : end]


def _detect_sections(text: str) -> list[dict[str, str]]:
    heading_sections = _split_on_headings(text)
    if heading_sections:
        return heading_sections
    marker_sections = _split_on_markers(text)
    if marker_sections:
        return marker_sections
    return [{"label": "Script", "text": text}]


def _split_on_headings(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    sections: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if not match or not _sectionish_heading(match.group("label")):
            continue
        sections.append({"line_index": index, "label": _clean_label(match.group("label"))})
    return _sections_from_boundaries(lines, sections)


def _split_on_markers(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    sections: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        match = SECTION_MARKER_RE.match(line)
        if not match:
            continue
        number = match.group("section") or match.group("part")
        label = match.group("label") or f"Section {number}"
        sections.append({"line_index": index, "label": _clean_label(label)})
    return _sections_from_boundaries(lines, sections)


def _sections_from_boundaries(lines: list[str], boundaries: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not boundaries:
        return []
    sections = []
    for index, boundary in enumerate(boundaries):
        start = boundary["line_index"] + 1
        end = boundaries[index + 1]["line_index"] if index + 1 < len(boundaries) else len(lines)
        sections.append({"label": boundary["label"], "text": "\n".join(lines[start:end])})
    return sections


def _sectionish_heading(label: str) -> bool:
    clean = label.strip()
    return bool(
        re.search(r"\b(section|part)\b", clean, flags=re.IGNORECASE)
        or NUMBERED_HEADING_RE.match(clean)
    )


def _analyze_section(section_number: int, label: str, raw_text: str, rules: ScriptAuditRules) -> dict[str, Any]:
    cleaned_text, excluded_line_count = _spoken_text(raw_text)
    valid_pauses, invalid_pauses = _parse_pauses(cleaned_text, section_number)
    findings = _section_pause_findings(cleaned_text, section_number)
    text_without_pauses = VALID_PAUSE_RE.sub(" ", cleaned_text)
    text_without_pauses = PAUSE_LIKE_RE.sub(" ", text_without_pauses)
    word_count = _word_count(text_without_pauses)
    explicit_pause_seconds = round(sum(pause["seconds"] for pause in valid_pauses), 3)
    estimated_spoken_seconds = round(word_count / rules.wpm * 60, 3) if rules.wpm else 0.0
    estimated_total_seconds = round(estimated_spoken_seconds + explicit_pause_seconds, 3)
    budget = _section_budget(section_number)
    budget_min = budget[1] if budget else None
    budget_max = budget[2] if budget else None
    over_budget = round(max(0.0, estimated_total_seconds - budget_max), 3) if budget_max is not None else 0.0
    return {
        "section": {
            "section_number": section_number,
            "label": label or f"Section {section_number}",
            "word_count": word_count,
            "estimated_spoken_seconds": estimated_spoken_seconds,
            "explicit_pause_seconds": explicit_pause_seconds,
            "estimated_total_seconds": estimated_total_seconds,
            "estimated_total_timecode": _timecode(estimated_total_seconds),
            "pause_tag_count": len(valid_pauses),
            "budget_min_seconds": budget_min,
            "budget_max_seconds": budget_max,
            "over_budget_seconds": over_budget,
        },
        "valid_pauses": valid_pauses,
        "invalid_pauses": invalid_pauses,
        "findings": findings,
        "excluded_line_count": excluded_line_count,
    }


def _section_pause_findings(text: str, section_number: int) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for paragraph in re.split(r"\n\s*\n", text):
        pause_count = len(VALID_PAUSE_RE.findall(paragraph))
        if pause_count > 2:
            findings.append(
                _finding(
                    "warning",
                    "TOO_MANY_PAUSES_IN_PARAGRAPH",
                    f"A paragraph in section {section_number} contains {pause_count} pause tags.",
                    section_number,
                    "Use punctuation for most pauses and reserve pause tags for major beats.",
                )
            )

    matches = list(VALID_PAUSE_RE.finditer(text))
    for left, right in zip(matches, matches[1:]):
        between = text[left.end() : right.start()]
        spoken_words_between = _word_count(PAUSE_LIKE_RE.sub(" ", between))
        if spoken_words_between < 12:
            findings.append(
                _finding(
                    "warning",
                    "PAUSE_TAGS_TOO_CLOSE",
                    f"Two pause tags in section {section_number} are only {spoken_words_between} spoken word(s) apart.",
                    section_number,
                    "Remove one pause tag or let punctuation carry the beat.",
                )
            )
    return findings


def _spoken_text(text: str) -> tuple[str, int]:
    text = re.sub(r"```.*?```", "\n", text, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", "\n", text, flags=re.DOTALL)
    lines = text.splitlines()
    cleaned: list[str] = []
    excluded = 0
    in_metadata_block = True
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            in_metadata_block = False
            continue
        if HEADING_RE.match(line) or SECTION_MARKER_RE.match(line):
            excluded += 1
            continue
        if _is_markdown_table_line(stripped):
            excluded += 1
            continue
        if in_metadata_block and METADATA_RE.match(stripped):
            excluded += 1
            continue
        in_metadata_block = False
        cleaned.append(line)
    return "\n".join(cleaned), excluded


def _is_markdown_table_line(stripped: str) -> bool:
    if TABLE_SEPARATOR_RE.match(stripped):
        return True
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _parse_pauses(text: str, section_number: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = []
    valid_spans = []
    for match in VALID_PAUSE_RE.finditer(text):
        seconds = float(match.group("seconds"))
        valid.append(
            {
                "section_number": section_number,
                "seconds": seconds,
                "tag": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        )
        valid_spans.append(match.span())

    invalid = []
    for match in PAUSE_LIKE_RE.finditer(text):
        if any(start == match.start() and end == match.end() for start, end in valid_spans):
            continue
        invalid.append({"section_number": section_number, "tag": match.group(0), "start": match.start()})

    return valid, invalid


def _add_total_findings(
    findings: list[dict[str, Any]],
    rules: ScriptAuditRules,
    estimated_total_seconds: float,
    word_count: int,
    explicit_pause_seconds: float,
    pause_percentage: float,
    pause_tag_count: int,
    longest_pause_seconds: float,
    average_pause_seconds: float,
) -> None:
    if estimated_total_seconds > rules.hard_cap_seconds:
        findings.append(
            _finding(
                "error",
                "RUNTIME_HARD_CAP_EXCEEDED",
                f"Estimated runtime is {_timecode(estimated_total_seconds)}, above hard cap {_timecode(rules.hard_cap_seconds)}.",
                recommendation="Cut script length before generating voiceover.",
            )
        )
    elif estimated_total_seconds > rules.warning_seconds:
        findings.append(
            _finding(
                "warning",
                "RUNTIME_WARNING",
                f"Estimated runtime is {_timecode(estimated_total_seconds)}, above warning threshold {_timecode(rules.warning_seconds)}.",
                recommendation="Tighten the script before generating voiceover.",
            )
        )

    if word_count > rules.word_block:
        findings.append(_finding("error", "WORD_COUNT_BLOCK", f"Word count is {word_count}, above block threshold {rules.word_block}."))
    elif word_count > rules.word_warning:
        findings.append(_finding("warning", "WORD_COUNT_WARNING", f"Word count is {word_count}, above warning threshold {rules.word_warning}."))

    if explicit_pause_seconds > rules.pause_block_seconds:
        findings.append(
            _finding("error", "PAUSE_SECONDS_BLOCK", f"Explicit pause total is {explicit_pause_seconds:.1f}s, above block threshold {rules.pause_block_seconds:.1f}s.")
        )
    elif explicit_pause_seconds > rules.pause_warning_seconds:
        findings.append(
            _finding("warning", "PAUSE_SECONDS_WARNING", f"Explicit pause total is {explicit_pause_seconds:.1f}s, above warning threshold {rules.pause_warning_seconds:.1f}s.")
        )

    if pause_percentage > rules.pause_percentage_warning:
        findings.append(
            _finding("warning", "PAUSE_PERCENTAGE_WARNING", f"Explicit pauses are {pause_percentage:.1f}% of estimated runtime, above {rules.pause_percentage_warning:.1f}%.")
        )

    if pause_tag_count > rules.pause_tag_block:
        findings.append(_finding("error", "PAUSE_TAG_COUNT_BLOCK", f"Pause tag count is {pause_tag_count}, above block threshold {rules.pause_tag_block}."))
    elif pause_tag_count > rules.pause_tag_warning:
        findings.append(_finding("warning", "PAUSE_TAG_COUNT_WARNING", f"Pause tag count is {pause_tag_count}, above warning threshold {rules.pause_tag_warning}."))

    if longest_pause_seconds > 2.0:
        findings.append(_finding("error", "LONG_PAUSE_TAG_BLOCK", f"Longest pause tag is {longest_pause_seconds:.1f}s, above 2.0s."))
    elif longest_pause_seconds > 1.0:
        findings.append(_finding("warning", "LONG_PAUSE_TAG_WARNING", f"Longest pause tag is {longest_pause_seconds:.1f}s, above 1.0s."))

    if average_pause_seconds > 0.75:
        findings.append(_finding("warning", "LONG_PAUSE_TAG_WARNING", f"Average pause tag duration is {average_pause_seconds:.2f}s, above 0.75s."))


def _add_pause_findings(
    findings: list[dict[str, Any]],
    valid_pauses: list[dict[str, Any]],
    invalid_pauses: list[dict[str, Any]],
) -> None:
    for invalid in invalid_pauses:
        findings.append(
            _finding(
                "warning",
                "INVALID_PAUSE_TAG",
                f"Invalid pause-like tag {invalid['tag']} in section {invalid['section_number']}.",
                invalid["section_number"],
                "Use decimal pause tags such as <#0.3#>.",
            )
        )


def _add_section_findings(findings: list[dict[str, Any]], sections: list[dict[str, Any]], rules: ScriptAuditRules) -> None:
    full_section_budget_enabled = len(sections) == rules.expected_sections
    total_runtime = sum(section["estimated_total_seconds"] for section in sections)
    if not full_section_budget_enabled and total_runtime <= rules.target_max_seconds:
        return

    if not full_section_budget_enabled:
        section = max(sections, key=lambda item: item["estimated_total_seconds"], default=None)
        if section is None:
            return
        over_target = total_runtime - rules.target_max_seconds
        findings.append(
            _finding(
                "warning",
                "SECTION_OVER_BUDGET",
                f"Detected section structure is incomplete and total runtime is {over_target:.1f}s over target.",
                section["section_number"],
                "Add five sections and cut the longest section first.",
            )
        )
        return

    for section in sections:
        budget_max = section.get("budget_max_seconds")
        if budget_max is None:
            continue
        over_budget = section["estimated_total_seconds"] - budget_max
        if over_budget > 30:
            findings.append(
                _finding(
                    "warning",
                    "SECTION_FAR_OVER_BUDGET",
                    f"Section {section['section_number']} is {over_budget:.1f}s over its recommended budget.",
                    section["section_number"],
                    "Cut this section first if the full script needs shortening.",
                )
            )
        elif over_budget > 0:
            findings.append(
                _finding(
                    "warning",
                    "SECTION_OVER_BUDGET",
                    f"Section {section['section_number']} is {over_budget:.1f}s over its recommended budget.",
                    section["section_number"],
                    "Review this section for tightening.",
                )
            )


def _suggested_cuts(sections: list[dict[str, Any]], rules: ScriptAuditRules) -> list[dict[str, Any]]:
    candidates = []
    if len(sections) == rules.expected_sections:
        for section in sections:
            budget_max = section.get("budget_max_seconds")
            if budget_max is None:
                continue
            over = section["estimated_total_seconds"] - budget_max
            if over > 0:
                candidates.append((over, section))
    if not candidates:
        total = sum(section["estimated_total_seconds"] for section in sections)
        if total <= rules.target_max_seconds or not sections:
            return []
        section = max(sections, key=lambda item: item["estimated_total_seconds"])
        candidates.append((total - rules.target_max_seconds, section))

    seconds_to_cut, section = max(candidates, key=lambda item: (item[0], item[1]["word_count"]))
    words_to_cut = int(math.ceil(seconds_to_cut * rules.wpm / 60))
    issue = _cut_issue(section)
    return [
        {
            "section_number": section["section_number"],
            "seconds_to_cut": round(seconds_to_cut, 1),
            "approx_words_to_cut": words_to_cut,
            "pause_seconds_in_section": section["explicit_pause_seconds"],
            "issue": issue,
            "message": (
                f"Section {section['section_number']} is {seconds_to_cut:.1f}s over budget. "
                f"Cut about {words_to_cut} words or remove {min(seconds_to_cut, section['explicit_pause_seconds']):.1f}s of explicit pauses."
            ),
        }
    ]


def _cut_issue(section: dict[str, Any]) -> str:
    if section["word_count"] and section["explicit_pause_seconds"] >= 10:
        return "prose length and pause tags"
    if section["explicit_pause_seconds"] >= 10:
        return "pause tags"
    return "prose length"


def _pause_duration_distribution(pauses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[float, int] = {}
    for pause in pauses:
        key = round(pause["seconds"], 3)
        counts[key] = counts.get(key, 0) + 1
    return [{"seconds": key, "count": counts[key]} for key in sorted(counts)]


def _status(findings: list[dict[str, Any]]) -> str:
    error_codes = {finding["code"] for finding in findings if finding["severity"] == "error"}
    if error_codes & {"MISSING_SCRIPT", "UNREADABLE_SCRIPT"}:
        return "error"
    if error_codes:
        return "blocked_too_long"
    if any(finding["severity"] == "warning" for finding in findings):
        return "needs_cut"
    return "pass"


def _write_reports(report: dict[str, Any], json_path: Path, markdown_path: Path) -> ScriptAuditResult:
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    except OSError as exc:
        raise PipelineError(f"Unable to write script audit reports: {exc}") from exc
    return ScriptAuditResult(report=report, json_path=json_path, markdown_path=markdown_path)


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Script Audit Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Summary",
        "",
        f"- Input: `{summary['input_path']}`",
        f"- Estimated runtime: {summary['estimated_total_timecode']} ({summary['estimated_total_seconds']:.1f}s)",
        f"- Words: {summary['word_count']}",
        f"- Explicit pauses: {summary['explicit_pause_seconds']:.1f}s",
        f"- Pause percentage: {summary['pause_percentage']:.1f}%",
        f"- Pause tags: {summary['pause_tag_count']}",
        f"- Longest pause: {summary['longest_pause_seconds']:.1f}s",
        f"- Sections: {summary['section_count']}",
        f"- WPM: {summary['wpm']}",
        "",
        "## Section Timing",
        "",
    ]
    if report["sections"]:
        lines.extend(
            [
                "| Section | Label | Words | Runtime | Pauses | Budget | Over |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for section in report["sections"]:
            budget = "-"
            if section["budget_min_seconds"] is not None and section["budget_max_seconds"] is not None:
                budget = f"{section['budget_min_seconds']:.0f}-{section['budget_max_seconds']:.0f}s"
            lines.append(
                f"| {section['section_number']} | {section['label']} | {section['word_count']} | "
                f"{section['estimated_total_seconds']:.1f}s | {section['explicit_pause_seconds']:.1f}s | "
                f"{budget} | {section['over_budget_seconds']:.1f}s |"
            )
    else:
        lines.append("- No sections available.")
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        for finding in report["findings"]:
            section = f" section {finding['section_number']}:" if "section_number" in finding else ""
            lines.append(f"- `{finding['severity']}` `{finding['code']}`{section} {finding['message']}")
    else:
        lines.append("- None")
    lines.extend(["", "## Suggested Cuts", ""])
    if report["suggested_cuts"]:
        for cut in report["suggested_cuts"]:
            lines.append(f"- {cut['message']} Issue: {cut['issue']}.")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _section_budget(section_number: int) -> tuple[str, float, float] | None:
    if 1 <= section_number <= len(SECTION_BUDGETS):
        return SECTION_BUDGETS[section_number - 1]
    return None


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _timecode(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def _finding(
    severity: str,
    code: str,
    message: str,
    section_number: int | None = None,
    recommendation: str | None = None,
) -> dict[str, Any]:
    finding: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if section_number is not None:
        finding["section_number"] = section_number
    if recommendation is not None:
        finding["recommendation"] = recommendation
    return finding


def _sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        findings,
        key=lambda finding: (
            SEVERITY_ORDER.get(finding["severity"], 99),
            finding["code"],
            finding.get("section_number", 999999),
            finding["message"],
        ),
    )


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix().replace("\\", "/")


def _clean_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().strip("#")).strip()
