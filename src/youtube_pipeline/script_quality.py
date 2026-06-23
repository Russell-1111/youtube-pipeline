from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .errors import PipelineError

SCRIPT_QUALITY_SCHEMA_VERSION = 1
JSON_REPORT_NAME = "script_quality_report.json"
MARKDOWN_REPORT_NAME = "script_quality_report.md"

VOICEOVER_START = "<!-- VOICEOVER_START -->"
VOICEOVER_END = "<!-- VOICEOVER_END -->"
SCRIPT_BRIEF_RE = re.compile(r"<!--\s*SCRIPT_BRIEF(?P<body>.*?)-->", re.DOTALL | re.IGNORECASE)
FIELD_RE = re.compile(r"^\s*(?P<key>[A-Za-z][A-Za-z0-9_ -]*):\s*(?P<value>.*?)\s*$")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<label>.+?)\s*$")
SECTION_MARKER_RE = re.compile(
    r"^\s*(?:(?:SECTION|Section)\s+(?P<section>\d+)|(?:PART|Part)\s+(?P<part>\d+))"
    r"(?:\s*[:.\-]\s*(?P<label>.*))?\s*$"
)
NUMBERED_HEADING_RE = re.compile(r"^\s*(?P<number>\d+)[.)]\s+(?P<label>.+?)\s*$")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\'-][A-Za-z0-9]+)*")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")

REQUIRED_BRIEF_FIELDS = (
    "title",
    "thumbnail_promise",
    "viewer_problem",
    "central_tension",
    "central_question",
    "viewer_payoff",
    "open_loop",
    "final_takeaway",
    "target_viewer",
)
BLOCKING_BRIEF_FIELDS = {
    "title": "MISSING_TITLE",
    "thumbnail_promise": "MISSING_THUMBNAIL_PROMISE",
    "central_tension": "MISSING_CENTRAL_TENSION",
    "open_loop": "MISSING_OPEN_LOOP",
    "final_takeaway": "MISSING_FINAL_TAKEAWAY",
}
FIELD_CODES = {
    "title": "MISSING_TITLE",
    "thumbnail_promise": "MISSING_THUMBNAIL_PROMISE",
    "viewer_problem": "MISSING_VIEWER_PROBLEM",
    "central_tension": "MISSING_CENTRAL_TENSION",
    "central_question": "MISSING_CENTRAL_QUESTION",
    "viewer_payoff": "MISSING_VIEWER_PAYOFF",
    "open_loop": "MISSING_OPEN_LOOP",
    "final_takeaway": "MISSING_FINAL_TAKEAWAY",
    "target_viewer": "MISSING_TARGET_VIEWER",
}

SEVERITY_ORDER = {
    "error": 0,
    "warning": 1,
    "info": 2,
    "recommendation": 3,
}

RECOMMENDATION_PRIORITY = {
    "Add the required SCRIPT_BRIEF fields before voiceover review.": 0,
    "Write the first 30 seconds around a clear tension, contradiction, question, or specific promise.": 1,
    "Make the hook echo the title and thumbnail promise using concrete shared terms.": 2,
    "Use five explicit sections: hook, setup, core idea, deeper turn, and payoff.": 3,
    "Add meaningful turns that deepen the idea instead of restating it.": 4,
    "Replace generic phrasing with concrete modern-life images, consequences, and contradictions.": 5,
    "Rewrite the ending so it resolves or reframes the opening loop.": 6,
    "Run python -m youtube_pipeline --script-audit input/script.md for runtime compliance.": 99,
}

GENERIC_INTRO_PHRASES = (
    "in today's video",
    "in todays video",
    "we're going to talk about",
    "we are going to talk about",
    "today we are going to talk about",
    "today we're going to talk about",
    "let's dive in",
    "lets dive in",
)
WEAK_PHRASES = GENERIC_INTRO_PHRASES + (
    "at the end of the day",
    "in conclusion",
    "it is very important",
    "nowadays",
    "many people",
    "society today",
    "we all know",
    "this shows that",
)
GENERIC_SLOGANS = (
    "believe in yourself",
    "never give up",
    "follow your dreams",
    "work hard and you will succeed",
    "you can achieve anything",
    "be the best version of yourself",
)
TURN_TERMS = (
    "but",
    "yet",
    "however",
    "the strange thing is",
    "the problem is",
    "what nobody notices is",
    "worse",
    "deeper",
    "more importantly",
    "the real problem",
    "the hidden cost",
    "this is why",
    "that is the trap",
    "the point is",
    "the answer is not",
)
HOOK_REASON_TERMS = (
    "?",
    "but",
    "yet",
    "why",
    "how",
    "what if",
    "the strange thing",
    "the problem",
    "the trap",
    "hidden",
    "cost",
    "pressure",
    "promise",
    "promised",
    "less",
    "never",
)
THROAT_CLEARING_TERMS = (
    "before we begin",
    "first of all",
    "to understand this",
    "let us start with a definition",
    "let's start with a definition",
)
VAGUE_NOUNS = (
    "life",
    "society",
    "people",
    "things",
    "success",
    "mindset",
    "world",
    "time",
    "productivity",
    "happiness",
)
PURPOSE_KEYWORDS = {
    1: ("but", "why", "how", "what if", "problem", "trap", "promise", "hidden", "cost", "pressure"),
    2: ("problem", "because", "modern", "schedule", "pressure", "frame", "begins", "started"),
    3: ("because", "mechanism", "works", "this is why", "system", "explains", "turns"),
    4: ("but", "worse", "deeper", "hidden cost", "consequence", "real problem", "trap"),
    5: ("answer", "point", "finally", "takeaway", "this is why", "freedom", "means", "not to"),
}


@dataclass(frozen=True)
class ScriptQualityResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path

    @property
    def should_fail(self) -> bool:
        return self.report["status"] in {"blocked_weak_script", "error"}


def run_script_quality_audit(script_path: Path, data_dir: Path, base_dir: Path) -> ScriptQualityResult:
    json_path = data_dir / JSON_REPORT_NAME
    markdown_path = data_dir / MARKDOWN_REPORT_NAME

    if not script_path.exists():
        report = _error_report(script_path, base_dir, "MISSING_SCRIPT", f"Missing script file: {script_path}")
        return _write_reports(report, json_path, markdown_path)
    if not script_path.is_file():
        report = _error_report(script_path, base_dir, "UNREADABLE_SCRIPT", f"Script path is not a file: {script_path}")
        return _write_reports(report, json_path, markdown_path)

    try:
        raw_text = script_path.read_text(encoding="utf-8")
    except OSError as exc:
        report = _error_report(script_path, base_dir, "UNREADABLE_SCRIPT", f"Unreadable script file: {script_path} ({exc})")
        return _write_reports(report, json_path, markdown_path)
    except UnicodeDecodeError as exc:
        report = _error_report(script_path, base_dir, "UNREADABLE_SCRIPT", f"Script file must be UTF-8 text: {script_path} ({exc})")
        return _write_reports(report, json_path, markdown_path)

    report = audit_script_quality_text(raw_text, script_path, base_dir)
    return _write_reports(report, json_path, markdown_path)


def audit_script_quality_text(raw_text: str, script_path: Path, base_dir: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    score_breakdown: list[dict[str, Any]] = []
    blocked_reasons: list[str] = []
    recommendations: list[str] = ["Run python -m youtube_pipeline --script-audit input/script.md for runtime compliance."]

    brief, has_brief = _parse_script_brief(raw_text)
    voiceover_text = _voiceover_region(raw_text)
    spoken_text = _spoken_text(voiceover_text)
    sections = _analyze_sections(spoken_text)
    hook_text = sections[0]["text"] if sections else _first_words(spoken_text, 73)
    hook_30 = _first_words(hook_text, 73)

    if not has_brief:
        _add_finding(
            findings,
            "warning",
            "MISSING_SCRIPT_BRIEF",
            "Missing SCRIPT_BRIEF metadata block.",
            recommendation="Add SCRIPT_BRIEF metadata before the voiceover markers.",
        )
        _deduct(score_breakdown, "Missing SCRIPT_BRIEF metadata block", 12)
        _add_recommendation(recommendations, "Add the required SCRIPT_BRIEF fields before voiceover review.")

    for field in REQUIRED_BRIEF_FIELDS:
        if brief.get(field, "").strip():
            continue
        code = FIELD_CODES[field]
        severity = "error" if field in BLOCKING_BRIEF_FIELDS else "warning"
        message = f"Missing required script brief field: {field}."
        _add_finding(findings, severity, code, message, recommendation=f"Fill in SCRIPT_BRIEF field `{field}`.")
        deduction = 15 if field in BLOCKING_BRIEF_FIELDS else 5
        _deduct(score_breakdown, f"Missing {field.replace('_', ' ')}", deduction)
        if field in BLOCKING_BRIEF_FIELDS:
            blocked_reasons.append(message)
        _add_recommendation(recommendations, "Add the required SCRIPT_BRIEF fields before voiceover review.")

    if not spoken_text.strip():
        message = "Missing voiceover text to audit."
        _add_finding(findings, "error", "MISSING_VOICEOVER_TEXT", message, recommendation="Add spoken script text before running the quality gate.")
        _deduct(score_breakdown, "Missing voiceover text", 40)
        blocked_reasons.append(message)

    hook_analysis = _analyze_hook(hook_30, brief, findings, score_breakdown, recommendations)
    retention_structure = _analyze_retention_structure(spoken_text, sections, findings, score_breakdown, recommendations)
    _analyze_repetition(spoken_text, sections, findings, score_breakdown, recommendations)
    _analyze_specificity(spoken_text, findings, score_breakdown, recommendations)
    _analyze_ending(sections, brief, findings, score_breakdown, recommendations, blocked_reasons)

    if len(sections) != 5:
        _add_finding(
            findings,
            "warning",
            "SECTION_COUNT_MISMATCH",
            f"Detected {len(sections)} section(s); expected 5.",
            recommendation="Use five explicit sections before voiceover generation.",
        )
        _deduct(score_breakdown, "Five-section retention structure is incomplete", 10)
        _add_recommendation(recommendations, "Use five explicit sections: hook, setup, core idea, deeper turn, and payoff.")

    if not _has_payoff_section(sections):
        message = "Missing payoff or final section."
        _add_finding(
            findings,
            "error",
            "MISSING_PAYOFF_SECTION",
            message,
            recommendation="Add a final payoff section that resolves or reframes the opening loop.",
        )
        _deduct(score_breakdown, "Missing payoff or final section", 20)
        blocked_reasons.append(message)
        _add_recommendation(recommendations, "Rewrite the ending so it resolves or reframes the opening loop.")

    quality_score = max(0, 100 - sum(item["deduction"] for item in score_breakdown))
    if quality_score < 70:
        message = f"Script quality heuristic score is {quality_score}, below the 70-point block threshold."
        _add_finding(
            findings,
            "error",
            "LOW_QUALITY_SCORE",
            message,
            recommendation="Revise the script before generating voiceover.",
        )
        blocked_reasons.append(message)

    sorted_findings = _sort_findings(findings)
    sorted_recommendations = _sort_recommendations(recommendations)
    quality_label = _quality_label(quality_score)
    status = _status(quality_score, sorted_findings, blocked_reasons)

    report = {
        "script_quality_schema_version": SCRIPT_QUALITY_SCHEMA_VERSION,
        "status": status,
        "quality_score": quality_score,
        "quality_label": quality_label,
        "summary": {
            "input_path": _display_path(script_path, base_dir),
            "scoring_note": "Retention-risk score based on local structural heuristics; this is not a forecast of audience performance.",
            "voiceover_word_count": _word_count(spoken_text),
            "section_count": len(sections),
            "script_brief_present": has_brief,
            "voiceover_markers_present": _has_voiceover_markers(raw_text),
        },
        "required_brief_fields": {
            field: {
                "present": bool(brief.get(field, "").strip()),
                "blocking": field in BLOCKING_BRIEF_FIELDS,
            }
            for field in REQUIRED_BRIEF_FIELDS
        },
        "sections": [
            {
                "section_number": section["section_number"],
                "label": section["label"],
                "purpose": section["purpose"],
                "word_count": section["word_count"],
                "verdict": section["verdict"],
                "notes": section["notes"],
            }
            for section in sorted(sections, key=lambda item: item["section_number"])
        ],
        "hook_analysis": hook_analysis,
        "retention_structure": retention_structure,
        "score_breakdown": score_breakdown,
        "findings": sorted_findings,
        "recommendations": sorted_recommendations,
        "blocked_reasons": sorted(dict.fromkeys(blocked_reasons)),
    }
    return report


def print_script_quality_summary(result: ScriptQualityResult, base_dir: Path) -> None:
    print("Script quality audit complete")
    print(f"Status: {result.report['status']}")
    print(f"Retention-risk score: {result.report['quality_score']}/100 {result.report['quality_label']}")
    print("Reports written:")
    print(f"- {_display_path(result.json_path, base_dir)}")
    print(f"- {_display_path(result.markdown_path, base_dir)}")


def _error_report(script_path: Path, base_dir: Path, code: str, message: str) -> dict[str, Any]:
    finding = _finding("error", code, message, recommendation="Provide a readable UTF-8 Markdown script file.")
    return {
        "script_quality_schema_version": SCRIPT_QUALITY_SCHEMA_VERSION,
        "status": "error",
        "quality_score": 0,
        "quality_label": "reject",
        "summary": {
            "input_path": _display_path(script_path, base_dir),
            "scoring_note": "Retention-risk score based on local structural heuristics; this is not a forecast of audience performance.",
            "voiceover_word_count": 0,
            "section_count": 0,
            "script_brief_present": False,
            "voiceover_markers_present": False,
        },
        "required_brief_fields": {
            field: {"present": False, "blocking": field in BLOCKING_BRIEF_FIELDS}
            for field in REQUIRED_BRIEF_FIELDS
        },
        "sections": [],
        "hook_analysis": {
            "first_30_second_text": "",
            "has_reason_to_keep_watching": False,
            "title_promise_alignment": "unavailable",
            "shared_title_promise_terms": [],
            "central_tension_terms_in_hook": [],
            "generic_intro_phrases": [],
            "throat_clearing_phrases": [],
            "verdict": "error",
        },
        "retention_structure": {
            "turn_count": 0,
            "turn_density_per_100_words": 0.0,
            "open_loop_present": False,
            "final_takeaway_present": False,
            "verdict": "error",
        },
        "score_breakdown": [{"reason": message, "deduction": 100}],
        "findings": [finding],
        "recommendations": ["Provide a readable UTF-8 Markdown script file."],
        "blocked_reasons": [message],
    }


def _parse_script_brief(raw_text: str) -> tuple[dict[str, str], bool]:
    match = SCRIPT_BRIEF_RE.search(raw_text)
    if not match:
        return {}, False
    fields: dict[str, str] = {}
    for line in match.group("body").splitlines():
        field_match = FIELD_RE.match(line)
        if not field_match:
            continue
        key = field_match.group("key").strip().lower().replace(" ", "_").replace("-", "_")
        fields[key] = field_match.group("value").strip()
    return fields, True


def _voiceover_region(text: str) -> str:
    start = text.find(VOICEOVER_START)
    end = text.find(VOICEOVER_END)
    if start != -1 and end != -1 and end > start:
        return text[start + len(VOICEOVER_START) : end]
    return SCRIPT_BRIEF_RE.sub("\n", text)


def _has_voiceover_markers(text: str) -> bool:
    start = text.find(VOICEOVER_START)
    end = text.find(VOICEOVER_END)
    return start != -1 and end != -1 and end > start


def _spoken_text(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "\n", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "\n", text, flags=re.DOTALL)
    return text.strip()


def _analyze_sections(text: str) -> list[dict[str, Any]]:
    sections = _split_on_headings(text) or _split_on_markers(text)
    if not sections and text.strip():
        sections = [{"label": "Script", "text": text}]

    analyzed = []
    for index, section in enumerate(sections, start=1):
        clean_text = section["text"].strip()
        notes = []
        purpose = _section_purpose(index)
        purpose_terms = PURPOSE_KEYWORDS.get(index, ())
        if purpose_terms and not _contains_any(clean_text, purpose_terms):
            notes.append("Purpose keywords are weak or absent.")
        if _word_count(clean_text) < 20:
            notes.append("Section may be too thin to carry its retention purpose.")
        verdict = "review" if notes else "clear"
        analyzed.append(
            {
                "section_number": index,
                "label": section["label"] or f"Section {index}",
                "text": clean_text,
                "purpose": purpose,
                "word_count": _word_count(clean_text),
                "verdict": verdict,
                "notes": notes,
            }
        )
    return analyzed


def _split_on_headings(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    boundaries: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if not match:
            continue
        label = _clean_label(match.group("label"))
        if _sectionish_heading(label):
            boundaries.append({"line_index": index, "label": label})
    return _sections_from_boundaries(lines, boundaries)


def _split_on_markers(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    boundaries: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        match = SECTION_MARKER_RE.match(line)
        if not match:
            continue
        number = match.group("section") or match.group("part")
        label = match.group("label") or f"Section {number}"
        boundaries.append({"line_index": index, "label": _clean_label(label)})
    return _sections_from_boundaries(lines, boundaries)


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
        re.search(r"\b(section|part|hook|setup|core|turn|payoff|ending|conclusion)\b", clean, flags=re.IGNORECASE)
        or NUMBERED_HEADING_RE.match(clean)
    )


def _analyze_hook(
    hook_text: str,
    brief: dict[str, str],
    findings: list[dict[str, Any]],
    score_breakdown: list[dict[str, Any]],
    recommendations: list[str],
) -> dict[str, Any]:
    lower_hook = hook_text.lower()
    generic_phrases = _matched_phrases(lower_hook, GENERIC_INTRO_PHRASES)
    throat_clearing = _matched_phrases(lower_hook, THROAT_CLEARING_TERMS)
    has_reason = _contains_any(lower_hook, HOOK_REASON_TERMS)
    title_terms = _important_terms(brief.get("title", ""))
    promise_terms = _important_terms(brief.get("thumbnail_promise", ""))
    alignment_terms = sorted((set(title_terms) | set(promise_terms)) & set(_important_terms(hook_text)))
    alignment_available = bool(title_terms or promise_terms)
    alignment_risk = alignment_available and not alignment_terms

    if not has_reason:
        _add_finding(
            findings,
            "warning",
            "WEAK_HOOK",
            "The first 30 seconds may not contain a clear reason to keep watching.",
            section_number=1,
            recommendation="Add tension, contradiction, a high-stakes question, or a specific promise immediately.",
        )
        _deduct(score_breakdown, "Weak first 30-second hook", 20)
        _add_recommendation(recommendations, "Write the first 30 seconds around a clear tension, contradiction, question, or specific promise.")

    if alignment_risk:
        _add_finding(
            findings,
            "warning",
            "HOOK_TITLE_MISMATCH_RISK",
            "Potential title-hook mismatch risk: title/thumbnail keywords do not appear in the opening.",
            section_number=1,
            recommendation="Make the opening echo the title and thumbnail promise with concrete shared terms.",
        )
        _deduct(score_breakdown, "Potential title-hook mismatch risk", 6)
        _add_recommendation(recommendations, "Make the hook echo the title and thumbnail promise using concrete shared terms.")

    if throat_clearing:
        _add_finding(
            findings,
            "warning",
            "HOOK_THROAT_CLEARING",
            "The hook may spend time on setup before tension.",
            section_number=1,
            recommendation="Open with the tension or consequence before background.",
        )
        _deduct(score_breakdown, "Hook throat-clearing risk", 8)

    if generic_phrases:
        _add_finding(
            findings,
            "warning",
            "GENERIC_INTRO_PHRASE",
            f"Generic intro phrase detected in the hook: {', '.join(generic_phrases)}.",
            section_number=1,
            recommendation="Replace generic channel-language with immediate tension.",
        )
        _deduct(score_breakdown, "Generic intro phrase detected", 8)
        _add_recommendation(recommendations, "Replace generic phrasing with concrete modern-life images, consequences, and contradictions.")

    central_terms = _important_terms(brief.get("central_tension", ""))
    central_terms_in_hook = sorted(set(central_terms) & set(_important_terms(hook_text)))
    if central_terms and not central_terms_in_hook:
        _add_finding(
            findings,
            "warning",
            "CENTRAL_TENSION_NOT_IN_HOOK",
            "Central tension keywords do not appear in the opening.",
            section_number=1,
            recommendation="Echo the central tension in the first 30 seconds.",
        )
        _deduct(score_breakdown, "Central tension keywords absent from hook", 6)

    verdict = "strong" if has_reason and not generic_phrases and not throat_clearing else "review"
    return {
        "first_30_second_text": _preview(hook_text, 500),
        "has_reason_to_keep_watching": has_reason,
        "title_promise_alignment": "risk" if alignment_risk else "aligned_or_unavailable",
        "shared_title_promise_terms": alignment_terms,
        "central_tension_terms_in_hook": central_terms_in_hook,
        "generic_intro_phrases": generic_phrases,
        "throat_clearing_phrases": throat_clearing,
        "verdict": verdict,
    }


def _analyze_retention_structure(
    text: str,
    sections: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    score_breakdown: list[dict[str, Any]],
    recommendations: list[str],
) -> dict[str, Any]:
    turn_count = _phrase_count(text, TURN_TERMS)
    words = max(_word_count(text), 1)
    density = round(turn_count / words * 100, 2)
    flat_sections = []

    for section in sections:
        if section["verdict"] != "clear":
            flat_sections.append(section["section_number"])
            _add_finding(
                findings,
                "warning",
                "SECTION_PURPOSE_WEAK",
                f"Section {section['section_number']} may not clearly serve its retention purpose.",
                section_number=section["section_number"],
                recommendation="Clarify how this section advances the hook, problem, mechanism, turn, or payoff.",
            )
            _deduct(score_breakdown, f"Section {section['section_number']} purpose is weak", 4)

    if density < 0.45 and words >= 80:
        _add_finding(
            findings,
            "warning",
            "LOW_RETENTION_TURN_DENSITY",
            "The script has low retention-turn density for a structural quality gate.",
            recommendation="Add meaningful turns that deepen, complicate, or resolve the idea.",
        )
        _deduct(score_breakdown, "Low retention-turn density", 10)
        _add_recommendation(recommendations, "Add meaningful turns that deepen the idea instead of restating it.")

    return {
        "turn_count": turn_count,
        "turn_density_per_100_words": density,
        "flat_section_numbers": flat_sections,
        "open_loop_present": True,
        "final_takeaway_present": True,
        "verdict": "review" if flat_sections or density < 0.45 else "clear",
    }


def _analyze_repetition(
    text: str,
    sections: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    score_breakdown: list[dict[str, Any]],
    recommendations: list[str],
) -> None:
    lower_text = text.lower()
    weak_matches = _matched_phrases(lower_text, WEAK_PHRASES + GENERIC_SLOGANS)
    repeated_weak = [phrase for phrase in weak_matches if lower_text.count(phrase) > 1]
    if repeated_weak:
        _add_finding(
            findings,
            "warning",
            "GENERIC_PHRASE_REPETITION",
            f"Repeated generic phrase risk: {', '.join(repeated_weak)}.",
            recommendation="Replace repeated generic phrasing with specific observations or consequences.",
        )
        _deduct(score_breakdown, "Repeated generic phrases", 8)
        _add_recommendation(recommendations, "Replace generic phrasing with concrete modern-life images, consequences, and contradictions.")
    elif weak_matches:
        _add_finding(
            findings,
            "warning",
            "GENERIC_INTRO_PHRASE",
            f"Generic or motivational phrase detected: {', '.join(weak_matches)}.",
            recommendation="Review generic phrasing before voiceover.",
        )
        _deduct(score_breakdown, "Generic or motivational phrase detected", 5)

    starts = [_sentence_start(sentence) for sentence in SENTENCE_RE.findall(text) if _sentence_start(sentence)]
    start_counts = Counter(starts)
    repetitive_starts = sorted(start for start, count in start_counts.items() if count >= 4 and start in {"you", "but", "and", "the problem"})
    if repetitive_starts:
        _add_finding(
            findings,
            "warning",
            "REPETITIVE_SENTENCE_STARTS",
            f"Repetitive sentence starts detected: {', '.join(repetitive_starts)}.",
            recommendation="Vary sentence openings to avoid a flat cadence.",
        )
        _deduct(score_breakdown, "Repetitive sentence starts", 6)

    for left, right in zip(sections, sections[1:]):
        overlap = set(_important_terms(left["text"])) & set(_important_terms(right["text"]))
        smaller = max(1, min(len(set(_important_terms(left["text"]))), len(set(_important_terms(right["text"])))))
        if len(overlap) >= 6 and len(overlap) / smaller >= 0.6:
            _add_finding(
                findings,
                "warning",
                "SECTION_REPETITION_RISK",
                f"Sections {left['section_number']} and {right['section_number']} may repeat similar ideas.",
                section_number=right["section_number"],
                recommendation="Make this section turn, deepen, or resolve the previous idea.",
            )
            _deduct(score_breakdown, f"Section {right['section_number']} repetition risk", 6)
            _add_recommendation(recommendations, "Add meaningful turns that deepen the idea instead of restating it.")


def _analyze_specificity(
    text: str,
    findings: list[dict[str, Any]],
    score_breakdown: list[dict[str, Any]],
    recommendations: list[str],
) -> None:
    words = [word.lower() for word in WORD_RE.findall(text)]
    if not words:
        return
    vague_count = sum(1 for word in words if word in VAGUE_NOUNS)
    concrete_signals = _phrase_count(
        text,
        ("calendar", "phone", "screen", "alarm", "meeting", "deadline", "notification", "commute", "inbox", "schedule"),
    )
    vague_ratio = vague_count / len(words) * 100
    if vague_count >= 16 and vague_ratio >= 3.0 and concrete_signals < 4:
        _add_finding(
            findings,
            "warning",
            "EXCESSIVE_VAGUE_LANGUAGE",
            "The script may lean too heavily on vague nouns without enough concrete framing.",
            recommendation="Ground abstract ideas in specific modern-life images or consequences.",
        )
        _deduct(score_breakdown, "Excessive vague language", 8)
        _add_recommendation(recommendations, "Replace generic phrasing with concrete modern-life images, consequences, and contradictions.")


def _analyze_ending(
    sections: list[dict[str, Any]],
    brief: dict[str, str],
    findings: list[dict[str, Any]],
    score_breakdown: list[dict[str, Any]],
    recommendations: list[str],
    blocked_reasons: list[str],
) -> None:
    if not sections:
        return
    final = sections[-1]
    final_text = final["text"].lower()
    weak_ending_phrases = _matched_phrases(final_text, ("in conclusion", "thanks for watching", "to summarize", "that is all"))
    takeaway_terms = _important_terms(brief.get("final_takeaway", ""))
    open_loop_terms = _important_terms(brief.get("open_loop", ""))
    final_terms = set(_important_terms(final_text))
    takeaway_overlap = sorted(set(takeaway_terms) & final_terms)
    open_loop_overlap = sorted(set(open_loop_terms) & final_terms)

    if weak_ending_phrases or (takeaway_terms and not takeaway_overlap):
        _add_finding(
            findings,
            "warning",
            "FLAT_ENDING",
            "Final section may not clearly resolve or reframe the opening loop.",
            section_number=final["section_number"],
            recommendation="End with a strong final idea instead of a recap or generic closure.",
        )
        _deduct(score_breakdown, "Flat ending or weak final takeaway connection", 10)
        _add_recommendation(recommendations, "Rewrite the ending so it resolves or reframes the opening loop.")

    final["notes"].extend(
        note
        for note in (
            "Final section has weak final_takeaway keyword overlap." if takeaway_terms and not takeaway_overlap else "",
            "Final section has weak open_loop keyword overlap." if open_loop_terms and not open_loop_overlap else "",
        )
        if note
    )


def _has_payoff_section(sections: list[dict[str, Any]]) -> bool:
    if len(sections) < 5:
        return False
    final_label = sections[-1]["label"].lower()
    final_text = sections[-1]["text"].lower()
    return bool(re.search(r"\b(payoff|ending|final|conclusion|takeaway|resolve)\b", final_label))


def _status(quality_score: int, findings: list[dict[str, Any]], blocked_reasons: list[str]) -> str:
    if any(finding["code"] in {"MISSING_SCRIPT", "UNREADABLE_SCRIPT"} for finding in findings):
        return "error"
    if blocked_reasons or quality_score < 70 or any(finding["severity"] == "error" for finding in findings):
        return "blocked_weak_script"
    if quality_score < 85 or any(finding["severity"] == "warning" for finding in findings):
        return "needs_rewrite"
    return "pass"


def _quality_label(score: int) -> str:
    if score >= 85:
        return "strong_script"
    if score >= 70:
        return "usable_but_revise"
    if score >= 50:
        return "weak_script"
    return "reject"


def _write_reports(report: dict[str, Any], json_path: Path, markdown_path: Path) -> ScriptQualityResult:
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    except OSError as exc:
        raise PipelineError(f"Unable to write script quality reports: {exc}") from exc
    return ScriptQualityResult(report=report, json_path=json_path, markdown_path=markdown_path)


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Script Quality Report",
        "",
        f"Status: `{report['status']}`",
        f"Retention-risk score: {report['quality_score']}/100 `{report['quality_label']}`",
        "",
        "This script quality heuristic uses local structural checks only. It is a pre-voiceover quality gate, not a forecast of audience performance.",
        "",
        "## Hook Verdict",
        "",
        f"- Verdict: {report['hook_analysis']['verdict']}",
        f"- Reason to keep watching: {report['hook_analysis']['has_reason_to_keep_watching']}",
        f"- Title-promise alignment: {report['hook_analysis']['title_promise_alignment']}",
        "",
        "## Central Tension",
        "",
        f"- Central tension present: {report['required_brief_fields']['central_tension']['present']}",
        f"- Terms in hook: {', '.join(report['hook_analysis']['central_tension_terms_in_hook']) or '-'}",
        "",
        "## Section Purpose",
        "",
    ]
    if report["sections"]:
        lines.extend(["| Section | Label | Purpose | Words | Verdict | Notes |", "| ---: | --- | --- | ---: | --- | --- |"])
        for section in report["sections"]:
            notes = "; ".join(section["notes"]) or "-"
            lines.append(
                f"| {section['section_number']} | {section['label']} | {section['purpose']} | "
                f"{section['word_count']} | {section['verdict']} | {notes} |"
            )
    else:
        lines.append("- No sections available.")

    lines.extend(["", "## Score Deductions", ""])
    if report["score_breakdown"]:
        for item in report["score_breakdown"]:
            lines.append(f"- -{item['deduction']}: {item['reason']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Top Weak Points", ""])
    warnings = [finding for finding in report["findings"] if finding["severity"] in {"error", "warning"}]
    if warnings:
        for finding in warnings[:10]:
            section = f" section {finding['section_number']}:" if "section_number" in finding else ":"
            lines.append(f"- `{finding['code']}`{section} {finding['message']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Generic Phrase Warnings", ""])
    lines.extend(_findings_markdown(report, {"GENERIC_INTRO_PHRASE", "GENERIC_PHRASE_REPETITION"}))
    lines.extend(["", "## Repetition Warnings", ""])
    lines.extend(_findings_markdown(report, {"REPETITIVE_SENTENCE_STARTS", "SECTION_REPETITION_RISK"}))
    lines.extend(["", "## Specificity Warnings", ""])
    lines.extend(_findings_markdown(report, {"EXCESSIVE_VAGUE_LANGUAGE"}))
    lines.extend(["", "## Ending / Payoff Verdict", ""])
    lines.extend(_findings_markdown(report, {"MISSING_PAYOFF_SECTION", "FLAT_ENDING"}))
    if not any(finding["code"] in {"MISSING_PAYOFF_SECTION", "FLAT_ENDING"} for finding in report["findings"]):
        lines.append("- Ending/payoff passed local heuristic review.")

    lines.extend(["", "## Next Rewrite Instructions", ""])
    lines.extend(f"- {recommendation}" for recommendation in report["recommendations"])
    if not report["recommendations"]:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _findings_markdown(report: dict[str, Any], codes: set[str]) -> list[str]:
    items = [finding for finding in report["findings"] if finding["code"] in codes]
    if not items:
        return ["- None"]
    return [f"- `{finding['code']}`: {finding['message']}" for finding in items]


def _section_purpose(section_number: int) -> str:
    purposes = {
        1: "Hook: earns the click",
        2: "Setup: frames the problem",
        3: "Core: explains the mechanism",
        4: "Turn: deepens or complicates the idea",
        5: "Payoff: resolves the opening loop",
    }
    return purposes.get(section_number, "Extra section: review for retention purpose")


def _deduct(score_breakdown: list[dict[str, Any]], reason: str, deduction: int) -> None:
    if deduction <= 0:
        return
    score_breakdown.append({"reason": reason, "deduction": deduction})


def _add_recommendation(recommendations: list[str], recommendation: str) -> None:
    if recommendation not in recommendations:
        recommendations.append(recommendation)


def _sort_recommendations(recommendations: list[str]) -> list[str]:
    return sorted(dict.fromkeys(recommendations), key=lambda item: (RECOMMENDATION_PRIORITY.get(item, 50), item))


def _add_finding(
    findings: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    section_number: int | None = None,
    recommendation: str | None = None,
) -> None:
    findings.append(_finding(severity, code, message, section_number, recommendation))


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


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _first_words(text: str, count: int) -> str:
    words = WORD_RE.findall(text)
    if len(words) <= count:
        return " ".join(words)
    return " ".join(words[:count])


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def _matched_phrases(text: str, phrases: tuple[str, ...]) -> list[str]:
    lower = text.lower()
    return sorted({phrase for phrase in phrases if phrase in lower})


def _phrase_count(text: str, phrases: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(lower.count(phrase) for phrase in phrases)


def _important_terms(text: str) -> list[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "this",
        "to",
        "we",
        "what",
        "when",
        "why",
        "with",
        "you",
        "your",
    }
    return [
        word
        for word in (match.lower() for match in WORD_RE.findall(text))
        if len(word) >= 4 and word not in stop_words
    ]


def _sentence_start(sentence: str) -> str:
    words = WORD_RE.findall(sentence.strip())
    if not words:
        return ""
    first = words[0].lower()
    if first == "the" and len(words) > 1 and words[1].lower() == "problem":
        return "the problem"
    return first


def _preview(text: str, max_chars: int = 100) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _clean_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().strip("#")).strip()


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix().replace("\\", "/")
