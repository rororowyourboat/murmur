"""Murmur plugin: structured meeting summarization via DSPy + LiteLLM."""

# NOTE: no `from __future__ import annotations` — DSPy needs real type objects
# in Signature fields, not ForwardRef strings.

import json
from datetime import UTC
from pathlib import Path

import click
from rich.console import Console

from murmur import hooks
from murmur.artifacts import ArtifactStore, fingerprint_file
from murmur.config import get_section
from murmur.grounded_summary import (
    DEFAULT_CHUNK_CHARACTERS,
    PROMPT_VERSION,
    clean_transcript,
    generate_grounded_summary,
    generation_timestamp,
    load_source_transcript,
    persist_cleaned_transcript,
    render_summary,
)

console = Console()

DEFAULT_MODEL = "gemini/gemini-3-flash-preview"

SYSTEM_PROMPT = """\
You are an expert meeting analyst. Your job is to produce structured, \
actionable meeting summaries from raw transcripts.

Rules:
- Be concise and factual — no filler or pleasantries.
- Attribute action items to specific people when mentioned.
- Use exact names from the transcript; do not invent attendees.
- If a deadline or date is mentioned, include it.
- Preserve technical terms and project names exactly as spoken.
- If the transcript is unclear or low quality, note it rather than guessing.
- Every attendee, claim, decision, question, and action item must include one or
  more exact segment_ids from the supplied transcript. Unsupported output is discarded.
- Label decisions and action items as explicit commitments or inferred suggestions.
"""


def _check_dep() -> bool:
    try:
        import dspy  # noqa: F401
        import litellm  # noqa: F401

        return True
    except ImportError:
        console.print(
            "[red]dspy and litellm are not installed.[/red]\n"
            "Install with: [cyan]uv pip install murmur[ai][/cyan]"
        )
        return False


# ---------------------------------------------------------------------------
# Pydantic schemas + DSPy signature (built lazily, cached)
# ---------------------------------------------------------------------------

_summarizer_cache = None


def _build_summarizer():
    """Build and return a DSPy MeetingSummarizer module (cached)."""
    global _summarizer_cache
    if _summarizer_cache is not None:
        return _summarizer_cache

    import dspy
    import pydantic

    class GroundedClaim(pydantic.BaseModel):
        text: str = pydantic.Field(description="Factual claim supported by the source")
        segment_ids: list[str] = pydantic.Field(description="Exact supporting segment IDs")

    class Attendee(pydantic.BaseModel):
        name: str = pydantic.Field(description="Exact participant display name")
        segment_ids: list[str] = pydantic.Field(description="Segments where this person speaks")

    class Decision(GroundedClaim):
        commitment: str = pydantic.Field(description="explicit or inferred")
        confidence: float = pydantic.Field(description="Grounding confidence from 0 to 1")

    class ActionItem(pydantic.BaseModel):
        task: str = pydantic.Field(description="What needs to be done")
        owner: str = pydantic.Field(description="Person responsible, or 'Unassigned' if unclear")
        deadline: str = pydantic.Field(
            default="", description="Due date if mentioned, otherwise empty"
        )
        priority: str = pydantic.Field(default="normal", description="high, normal, or low")
        commitment: str = pydantic.Field(description="explicit or inferred")
        confidence: float = pydantic.Field(description="Grounding confidence from 0 to 1")
        segment_ids: list[str] = pydantic.Field(description="Exact supporting segment IDs")

    class MeetingSummary(pydantic.BaseModel):
        title: str = pydantic.Field(
            description="Short descriptive title for the meeting (5-10 words)"
        )
        attendees: list[Attendee] = pydantic.Field(
            default_factory=list, description="People supported as present by speaking segments"
        )
        executive_summary: list[GroundedClaim] = pydantic.Field(
            default_factory=list,
            description="Two or three grounded overview claims",
        )
        topics: list[GroundedClaim] = pydantic.Field(
            default_factory=list,
            description="Important grounded topics",
        )
        decisions: list[Decision] = pydantic.Field(
            default_factory=list,
            description="Decisions made or suggestions inferred",
        )
        open_questions: list[GroundedClaim] = pydantic.Field(
            default_factory=list,
            description="Unresolved questions supported by source segments",
        )
        action_items: list[ActionItem] = pydantic.Field(
            default_factory=list,
            description="Tasks assigned, volunteered, or clearly suggested",
        )

    # Resolve annotations so DSPy sees real types, not ForwardRef
    MeetingSummary.model_rebuild()

    class SummarizeTranscript(dspy.Signature):
        """Analyze a meeting transcript and produce a structured summary."""

        stage: str = dspy.InputField(desc="final, map, or reduce")
        transcript: str = dspy.InputField(desc="Segment-labelled source or partial summaries")
        glossary: str = dspy.InputField(desc="Authoritative term mappings as JSON")
        summary: MeetingSummary = dspy.OutputField(
            desc=("Structured meeting summary with decisions, action items, and discussion points")
        )

    class MeetingSummarizer(dspy.Module):
        def __init__(self):
            self.summarize = dspy.ChainOfThought(SummarizeTranscript)

        def forward(self, stage, transcript, glossary):
            return self.summarize(stage=stage, transcript=transcript, glossary=glossary)

    _summarizer_cache = MeetingSummarizer()
    return _summarizer_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_env():
    """Load .env file from the project root if it exists."""
    import os

    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _get_calendar_context(file_path):
    """Try to match a recording to a calendar event and return context string."""
    try:
        from murmur.plugins.calendar import event_to_context, match_recording_to_event
    except ImportError:
        return None

    # Extract timestamp from the recording's metadata
    meta_path = Path(file_path).with_suffix(".json")
    if not meta_path.exists():
        return None

    try:
        import json
        from datetime import datetime

        meta = json.loads(meta_path.read_text())
        started_at = meta.get("started_at")
        if not started_at:
            return None
        recording_time = datetime.fromisoformat(started_at)
        if recording_time.tzinfo is None:
            recording_time = recording_time.replace(tzinfo=UTC)
        event = match_recording_to_event(recording_time)
        if event:
            return event_to_context(event)
    except Exception:  # noqa: S110
        pass
    return None


def _get_task_context():
    """Load task_context.md if it exists and export_context is enabled."""
    cfg = get_section("tasks")
    if not cfg.get("export_context"):
        return None

    task_context_path = Path.home() / ".config" / "murmur" / "task_context.md"
    if not task_context_path.exists():
        return None

    content = task_context_path.read_text().strip()
    return content if content else None


def _get_system_prompt(file_path=None):
    """Build system prompt from base + memory + calendar + task context."""
    from murmur.plugins.memory import load_memory

    parts = [SYSTEM_PROMPT]
    memory = load_memory()
    if memory:
        parts.append(
            "The user has provided the following personal context. "
            "Use it to tailor the summary (e.g. highlight their action items, "
            "use their preferred format):\n\n" + memory
        )

    if file_path:
        cal_context = _get_calendar_context(file_path)
        if cal_context:
            parts.append(
                "The following meeting metadata was retrieved from the user's "
                "calendar. Use it to identify speakers and understand the "
                "meeting's purpose:\n\n" + cal_context
            )

    # Append task context if configured
    task_context = _get_task_context()
    if task_context:
        parts.append(
            "The following open tasks are currently tracked. Use this context "
            "to identify which tasks were discussed, updated, or resolved "
            "during the meeting. Flag any new tasks that should be created:\n\n" + task_context
        )

    return "\n\n".join(parts)


def _render_markdown(summary):
    """Render a validated grounded summary to Markdown."""
    payload = summary.model_dump(mode="json") if hasattr(summary, "model_dump") else summary
    return render_summary(payload)


def _llm_generate(model, stage, transcript, glossary, file_path=None):
    """Run one DSPy map, reduce, or final generation call and return JSON."""
    import dspy

    _load_env()
    lm = dspy.LM(model, system_prompt=_get_system_prompt(file_path))
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

    summarizer = _build_summarizer()
    result = summarizer(stage=stage, transcript=transcript, glossary=json.dumps(glossary))
    return result.summary.model_dump(mode="json")


def _find_transcript(file_path):
    """Find canonical JSON first, with derived and legacy text fallbacks."""
    file_path = Path(file_path)
    if file_path.suffix in (".json", ".md", ".txt"):
        return file_path

    store = ArtifactStore(file_path)
    for transcript_path in (
        store.path("transcript.json"),
        store.path("transcript.md"),
        store.path("transcript.txt"),
    ):
        if transcript_path.exists():
            return transcript_path

    legacy_path = file_path.with_suffix(".txt")
    if legacy_path.exists():
        return legacy_path

    console.print(
        f"[red]No transcript found for {file_path.name}.[/red]\n"
        f"Expected a canonical transcript under: [cyan]{store.directory}[/cyan]\n"
        f"Run [cyan]murmur transcribe {file_path}[/cyan] first."
    )
    raise SystemExit(1)


def _summarize_file(
    transcript_path: Path,
    model_name: str,
    *,
    glossary: dict[str, str] | None = None,
    max_characters: int = DEFAULT_CHUNK_CHARACTERS,
) -> Path:
    """Generate grounded JSON/Markdown while preserving source and cleaned layers."""
    store, source, source_path = load_source_transcript(transcript_path)
    cleaned = clean_transcript(source)
    cleaned_json, cleaned_markdown = persist_cleaned_transcript(store, cleaned, source_path)
    source_fingerprint = fingerprint_file(source_path)
    glossary = glossary or {}
    parameters = {
        "prompt_version": PROMPT_VERSION,
        "source_sha256": source_fingerprint["sha256"],
        "glossary": glossary,
        "max_chunk_characters": max_characters,
    }
    summary_path = store.path("summary.md")
    summary_json_path = store.path("summary.json")
    previous = store.jobs().get("jobs", {}).get("summarize:litellm", {})
    compatible_resume = bool(
        previous.get("model") == model_name and previous.get("parameters") == parameters
    )
    _, should_run = store.begin_job(
        "summarize",
        "litellm",
        model=model_name,
        parameters=parameters,
        input_paths=[source_path, cleaned_json],
        output_paths=[summary_json_path, summary_path, cleaned_json, cleaned_markdown],
        output_artifacts=[
            "summary_json",
            "summary_markdown",
            "cleaned_transcript_json",
            "cleaned_transcript_markdown",
        ],
        resume=compatible_resume,
    )
    if not should_run:
        return summary_path

    try:
        summary, run = generate_grounded_summary(
            cleaned,
            lambda stage, content, terms: _llm_generate(
                model_name,
                stage,
                content,
                terms,
                file_path=str(source_path),
            ),
            glossary=glossary,
            max_characters=max_characters,
        )
        generated_at = generation_timestamp()
        provenance = {
            "provider": "litellm",
            "model": model_name,
            "prompt_version": PROMPT_VERSION,
            "source_transcript": str(source_path),
            "source_sha256": source_fingerprint["sha256"],
            "glossary": glossary,
            "generated_at": generated_at,
            "generation": run,
        }
        summary["metadata"] = provenance
        summary_json_path = store.write_json("summary.json", summary)
        summary_path = store.write_text("summary.md", render_summary(summary))
        store.register_artifact(
            "summary_json",
            summary_json_path,
            kind="grounded_meeting_summary",
            provenance=provenance,
        )
        store.register_artifact(
            "summary_markdown", summary_path, kind="meeting_summary", provenance=provenance
        )
        store.complete_job("summarize", "litellm")
        return summary_path
    except Exception as error:
        store.fail_job("summarize", "litellm", error)
        raise


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def _parse_glossary(configured, entries) -> dict[str, str]:
    glossary = dict(configured) if isinstance(configured, dict) else {}
    for entry in entries:
        source, separator, canonical = entry.partition("=")
        if not separator or not source.strip() or not canonical.strip():
            raise click.ClickException("Glossary entries must use SPOKEN=CANONICAL format.")
        glossary[source.strip()] = canonical.strip()
    return glossary


def register(cli):
    """Register the summarize command and hook."""

    @cli.command()
    @click.argument("file", type=click.Path(exists=True))
    @click.option(
        "-m",
        "--model",
        default=None,
        help=f"LLM model (default: {DEFAULT_MODEL}). Any litellm model string.",
    )
    @click.option(
        "--glossary",
        "glossary_entries",
        multiple=True,
        help="Authoritative SPOKEN=CANONICAL term mapping (repeatable).",
    )
    @click.option(
        "--max-chars",
        type=click.IntRange(min=100),
        default=None,
        help="Maximum transcript characters per map call.",
    )
    def summarize(file, model, glossary_entries, max_chars):
        """Summarize a transcript using an LLM.

        FILE can be canonical transcript JSON, derived text, or an audio file.

        Uses DSPy to produce a grounded, citation-validated summary with
        attendees, topics, decisions, open questions, and action items.

        Requires: uv pip install murmur[ai]
        """
        if not _check_dep():
            raise SystemExit(1)

        cfg = get_section("summarize")
        model_name = model or cfg.get("model", DEFAULT_MODEL)
        glossary = _parse_glossary(cfg.get("glossary", {}), glossary_entries)
        chunk_characters = max_chars or cfg.get("max_chunk_characters", DEFAULT_CHUNK_CHARACTERS)

        transcript_path = _find_transcript(file)

        console.print(f"[bold]Summarizing[/bold] {transcript_path} (model={model_name})")

        summary_path = _summarize_file(
            transcript_path,
            model_name,
            glossary=glossary,
            max_characters=chunk_characters,
        )
        summary = summary_path.read_text()

        console.print(f"\n[bold green]Summary saved:[/bold green] {summary_path}")
        console.print(f"\n{summary}")
        hooks.emit("summary_complete", summary_path=str(summary_path), source_file=file)

    # Auto-summarize on transcription_complete if configured
    cfg = get_section("summarize")
    if cfg.get("auto"):

        def _auto_summarize(transcript_path, **kwargs):
            if not _check_dep():
                return
            model_name = cfg.get("model", DEFAULT_MODEL)
            transcript_path = Path(transcript_path)

            console.print(f"\n[bold]Auto-summarizing[/bold] {transcript_path}")
            summary_path = _summarize_file(
                transcript_path,
                model_name,
                glossary=cfg.get("glossary", {}),
                max_characters=cfg.get("max_chunk_characters", DEFAULT_CHUNK_CHARACTERS),
            )
            console.print(f"[bold green]Auto-summary saved:[/bold green] {summary_path}")
            hooks.emit(
                "summary_complete",
                summary_path=str(summary_path),
                source_file=kwargs.get("audio_path"),
            )

        hooks.on("transcription_complete", _auto_summarize)
