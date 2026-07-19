"""Grounded transcript cleanup, map-reduce orchestration, and citation validation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from murmur.artifacts import ArtifactStore, fingerprint_file

PROMPT_VERSION = 2
DEFAULT_CHUNK_CHARACTERS = 24_000
CLAIM_FIELDS = ("executive_summary", "topics", "decisions", "open_questions")
UNCERTAIN_TEXT = re.compile(r"\b(?:inaudible|unclear|unintelligible)\b|\?{2,}", re.IGNORECASE)

Generator = Callable[[str, str, dict[str, str]], dict[str, Any]]


def _clock(seconds: float) -> str:
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_source_transcript(input_path: str | Path) -> tuple[ArtifactStore, dict[str, Any], Path]:
    """Load canonical JSON when available, with a legacy text fallback."""
    candidate = Path(input_path).expanduser().resolve()
    store = ArtifactStore.for_input(candidate)
    canonical_path = store.path("transcript.json")
    source_path = canonical_path if canonical_path.is_file() else candidate
    if source_path.suffix == ".json":
        payload = json.loads(source_path.read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
            raise ValueError(f"Transcript JSON has an invalid schema: {source_path}")
        return store, payload, source_path

    text = source_path.read_text()
    segments = [
        {
            "id": f"legacy-{index:06d}",
            "start": float(index - 1),
            "end": float(index),
            "speaker": "unknown",
            "text": line.strip(),
        }
        for index, line in enumerate(text.splitlines(), 1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return (
        store,
        {
            "schema_version": 1,
            "source": str(source_path),
            "provider": "legacy-text",
            "segments": segments,
            "text": " ".join(segment["text"] for segment in segments),
        },
        source_path,
    )


def clean_transcript(source: dict[str, Any]) -> dict[str, Any]:
    """Normalize whitespace without rewriting source words or speaker identity."""
    cleaned = []
    for index, source_segment in enumerate(source.get("segments", []), 1):
        raw_text = str(source_segment.get("text", ""))
        text = " ".join(raw_text.split())
        if not text:
            continue
        speaker = str(source_segment.get("speaker") or "unknown")
        reasons = []
        if speaker == "unknown" or speaker.startswith("unknown:"):
            reasons.append("unresolved_speaker")
        if UNCERTAIN_TEXT.search(text):
            reasons.append("uncertain_transcription")
        cleaned.append(
            {
                "id": str(source_segment.get("id") or f"segment-{index:06d}"),
                "start": round(float(source_segment.get("start", 0.0)), 3),
                "end": round(float(source_segment.get("end", 0.0)), 3),
                "speaker": speaker,
                "side": source_segment.get("side", "unknown"),
                "text": text,
                "raw_text": raw_text,
                "uncertain": bool(reasons),
                "uncertainty_reasons": reasons,
            }
        )
    return {
        "schema_version": 1,
        "kind": "cleaned_transcript",
        "source_provider": source.get("provider"),
        "source_model": source.get("model"),
        "segments": cleaned,
        "text": " ".join(segment["text"] for segment in cleaned),
    }


def render_cleaned_transcript(cleaned: dict[str, Any]) -> str:
    lines = ["# Cleaned transcript", ""]
    for segment in cleaned["segments"]:
        marker = " ⚠" if segment["uncertain"] else ""
        lines.extend(
            [
                f'<a id="{segment["id"]}"></a>',
                f"**[{_clock(segment['start'])}] {segment['speaker']}**{marker}: "
                f"{segment['text']}",
                "",
            ]
        )
    return "\n".join(lines)


def persist_cleaned_transcript(
    store: ArtifactStore, cleaned: dict[str, Any], source_path: Path
) -> tuple[Path, Path]:
    cleaned_payload = dict(cleaned)
    cleaned_payload["source_transcript"] = str(source_path)
    cleaned_payload["source_fingerprint"] = fingerprint_file(source_path)
    json_path = store.write_json("transcript.cleaned.json", cleaned_payload)
    markdown_path = store.write_text(
        "transcript.cleaned.md", render_cleaned_transcript(cleaned_payload)
    )
    provenance = {
        "operation": "whitespace_normalization_and_uncertainty_annotation",
        "source_transcript": str(source_path),
        "source_sha256": cleaned_payload["source_fingerprint"]["sha256"],
    }
    store.register_artifact(
        "cleaned_transcript_json", json_path, kind="cleaned_transcript", provenance=provenance
    )
    store.register_artifact(
        "cleaned_transcript_markdown",
        markdown_path,
        kind="cleaned_transcript_markdown",
        provenance=provenance,
    )
    return json_path, markdown_path


def _segment_text(segment: dict[str, Any]) -> str:
    return (
        f"[{segment['id']} {_clock(segment['start'])}-{_clock(segment['end'])}] "
        f"{segment['speaker']}: {segment['text']}"
    )


def chunk_segments(
    segments: list[dict[str, Any]], max_characters: int = DEFAULT_CHUNK_CHARACTERS
) -> list[list[dict[str, Any]]]:
    if max_characters < 100:
        raise ValueError("Summary chunk size must be at least 100 characters.")
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 0
    for segment in segments:
        size = len(_segment_text(segment)) + 1
        if current and current_size + size > max_characters:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(segment)
        current_size += size
    if current:
        chunks.append(current)
    return chunks


def _empty_summary() -> dict[str, Any]:
    return {
        "title": "No source content",
        "attendees": [],
        "executive_summary": [],
        "topics": [],
        "decisions": [],
        "open_questions": [],
        "action_items": [],
    }


def _citation(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": segment["id"],
        "start": segment["start"],
        "end": segment["end"],
        "timestamp": _clock(segment["start"]),
        "speaker": segment["speaker"],
        "excerpt": segment["text"][:240],
        "href": f"transcript.cleaned.md#{segment['id']}",
    }


def _ground_item(
    item: Any, index: dict[str, dict[str, Any]], rejected: list[dict[str, Any]], field: str
) -> dict[str, Any] | None:
    if isinstance(item, str):
        item = {"text": item, "segment_ids": []}
    if not isinstance(item, dict):
        rejected.append({"field": field, "reason": "invalid_schema", "value": str(item)})
        return None
    requested = item.get("segment_ids", [])
    valid_ids = [segment_id for segment_id in requested if segment_id in index]
    if not valid_ids:
        rejected.append(
            {
                "field": field,
                "reason": "no_valid_source_citation",
                "value": item.get("text") or item.get("task") or "",
            }
        )
        return None
    grounded = dict(item)
    grounded["segment_ids"] = valid_ids
    grounded["citations"] = [_citation(index[segment_id]) for segment_id in valid_ids]
    if field in ("decisions", "action_items"):
        grounded["commitment"] = (
            item.get("commitment")
            if item.get("commitment") in ("explicit", "inferred")
            else "inferred"
        )
        try:
            grounded["confidence"] = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
        except TypeError, ValueError:
            grounded["confidence"] = 0.5
    if field == "action_items":
        grounded["source_excerpts"] = [citation["excerpt"] for citation in grounded["citations"]]
    return grounded


def validate_and_ground_summary(
    candidate: dict[str, Any], cleaned: dict[str, Any]
) -> dict[str, Any]:
    """Drop unsupported claims and resolve every retained citation from source data."""
    index = {segment["id"]: segment for segment in cleaned["segments"]}
    if not index:
        empty = _empty_summary()
        empty["rejected_claims"] = []
        empty["uncertainties"] = []
        return empty
    rejected: list[dict[str, Any]] = []
    grounded: dict[str, Any] = {
        "title": str(candidate.get("title") or "Meeting summary"),
    }
    for field in ("attendees", *CLAIM_FIELDS, "action_items"):
        items = candidate.get(field, [])
        if not isinstance(items, list):
            items = []
        grounded[field] = [
            result
            for item in items
            if (result := _ground_item(item, index, rejected, field)) is not None
        ]
    known_attendees = {
        item.get("name") or item.get("text")
        for item in grounded["attendees"]
        if item.get("name") or item.get("text")
    }
    for speaker in sorted({segment["speaker"] for segment in index.values()}):
        if speaker == "unknown" or speaker.startswith("unknown:") or speaker in known_attendees:
            continue
        source = next(segment for segment in index.values() if segment["speaker"] == speaker)
        grounded["attendees"].append(
            {"name": speaker, "segment_ids": [source["id"]], "citations": [_citation(source)]}
        )
    grounded["uncertainties"] = [
        {
            "text": ", ".join(segment["uncertainty_reasons"]),
            "segment_ids": [segment["id"]],
            "citations": [_citation(segment)],
        }
        for segment in index.values()
        if segment["uncertain"]
    ]
    grounded["rejected_claims"] = rejected
    return grounded


def generate_grounded_summary(
    cleaned: dict[str, Any],
    generator: Generator,
    *,
    glossary: dict[str, str] | None = None,
    max_characters: int = DEFAULT_CHUNK_CHARACTERS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one-pass or map-reduce generation and return summary plus run metadata."""
    segments = cleaned["segments"]
    chunks = chunk_segments(segments, max_characters) if segments else []
    calls = []
    if not chunks:
        return validate_and_ground_summary(_empty_summary(), cleaned), {
            "strategy": "no-source",
            "map_calls": 0,
            "reduce_calls": 0,
        }
    glossary = glossary or {}
    if len(chunks) == 1:
        content = "\n".join(_segment_text(segment) for segment in chunks[0])
        candidate = generator("final", content, glossary)
        calls.append({"stage": "final", "segment_count": len(chunks[0])})
        strategy = "single-pass"
        reduce_calls = 0
    else:
        partials = []
        for chunk in chunks:
            content = "\n".join(_segment_text(segment) for segment in chunk)
            partials.append(generator("map", content, glossary))
            calls.append({"stage": "map", "segment_count": len(chunk)})
        reduce_calls = 0
        while len(partials) > 1:
            groups: list[list[dict[str, Any]]] = []
            current: list[dict[str, Any]] = []
            current_size = 2
            for partial in partials:
                partial_size = len(json.dumps(partial, sort_keys=True)) + 1
                if current and current_size + partial_size > max_characters:
                    groups.append(current)
                    current = []
                    current_size = 2
                current.append(partial)
                current_size += partial_size
            if current:
                groups.append(current)
            if len(groups) == len(partials):
                groups = [partials[index : index + 2] for index in range(0, len(partials), 2)]
            reduced = []
            for group in groups:
                reduced.append(generator("reduce", json.dumps(group, sort_keys=True), glossary))
                calls.append({"stage": "reduce", "partial_count": len(group)})
                reduce_calls += 1
            partials = reduced
        candidate = partials[0]
        strategy = "map-reduce"
    return validate_and_ground_summary(candidate, cleaned), {
        "strategy": strategy,
        "map_calls": len(chunks) if len(chunks) > 1 else 0,
        "reduce_calls": reduce_calls,
        "calls": calls,
    }


def _cite(item: dict[str, Any]) -> str:
    return " ".join(
        f"[{citation['segment_id']} @ {citation['timestamp']}]({citation['href']})"
        for citation in item.get("citations", [])
    )


def render_summary(summary: dict[str, Any]) -> str:
    lines = [f"# {summary['title']}", ""]
    sections = (
        ("Attendees", "attendees"),
        ("Executive summary", "executive_summary"),
        ("Topics", "topics"),
        ("Decisions", "decisions"),
        ("Open questions", "open_questions"),
    )
    for title, field in sections:
        lines.extend([f"## {title}", ""])
        for item in summary.get(field, []):
            value = item.get("name") or item.get("text") or ""
            qualifier = ""
            if field == "decisions":
                qualifier = f" ({item['commitment']}, {item['confidence']:.0%})"
            lines.append(f"- {value}{qualifier} {_cite(item)}".rstrip())
        if not summary.get(field):
            lines.append("- None supported by the transcript.")
        lines.append("")

    lines.extend(["## Action items", ""])
    for item in summary.get("action_items", []):
        owner = item.get("owner") or "Unassigned"
        deadline = item.get("deadline") or "no stated deadline"
        lines.append(
            f"- **{item.get('task', '')}** - {owner}; {deadline}; "
            f"{item['commitment']}; {item['confidence']:.0%} {_cite(item)}".rstrip()
        )
        for excerpt in item.get("source_excerpts", []):
            lines.append(f"  - Source: “{excerpt}”")
    if not summary.get("action_items"):
        lines.append("- None supported by the transcript.")
    lines.append("")

    if summary.get("uncertainties"):
        lines.extend(["## Transcript uncertainties", ""])
        for item in summary["uncertainties"]:
            lines.append(f"- {item['text']} {_cite(item)}")
        lines.append("")
    return "\n".join(lines)


def generation_timestamp() -> str:
    return datetime.now(UTC).isoformat()
