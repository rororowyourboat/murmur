"""Murmur plugin: structured meeting summarization via DSPy + LiteLLM."""

# NOTE: no `from __future__ import annotations` — DSPy needs real type objects
# in Signature fields, not ForwardRef strings.

from datetime import UTC
from pathlib import Path

import click
from rich.console import Console

from murmur import hooks
from murmur.config import get_section

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

    class ActionItem(pydantic.BaseModel):
        task: str = pydantic.Field(description="What needs to be done")
        owner: str = pydantic.Field(description="Person responsible, or 'Unassigned' if unclear")
        deadline: str = pydantic.Field(
            default="", description="Due date if mentioned, otherwise empty"
        )
        priority: str = pydantic.Field(default="normal", description="high, normal, or low")

    class MeetingSummary(pydantic.BaseModel):
        title: str = pydantic.Field(
            description="Short descriptive title for the meeting (5-10 words)"
        )
        executive_summary: str = pydantic.Field(
            description="2-3 sentence overview of what the meeting was about and its outcome"
        )
        key_decisions: list[str] = pydantic.Field(
            default_factory=list,
            description="Decisions that were made, each as a concise bullet",
        )
        action_items: list[ActionItem] = pydantic.Field(
            default_factory=list,
            description="Tasks assigned or volunteered during the meeting",
        )
        discussion_points: list[str] = pydantic.Field(
            default_factory=list,
            description="Important topics discussed, grouped logically",
        )
        open_questions: list[str] = pydantic.Field(
            default_factory=list,
            description="Unresolved questions or topics deferred to later",
        )
        attendees: list[str] = pydantic.Field(
            default_factory=list,
            description="People who spoke or were mentioned as present",
        )

    # Resolve annotations so DSPy sees real types, not ForwardRef
    MeetingSummary.model_rebuild()

    class SummarizeTranscript(dspy.Signature):
        """Analyze a meeting transcript and produce a structured summary."""

        transcript: str = dspy.InputField(desc="Raw meeting transcript text")
        summary: MeetingSummary = dspy.OutputField(
            desc=("Structured meeting summary with decisions, action items, and discussion points")
        )

    class MeetingSummarizer(dspy.Module):
        def __init__(self):
            self.summarize = dspy.ChainOfThought(SummarizeTranscript)

        def forward(self, transcript):
            return self.summarize(transcript=transcript)

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
    except Exception:
        pass
    return None


def _get_system_prompt(file_path=None):
    """Build system prompt from base + memory + calendar context."""
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

    return "\n\n".join(parts)


def _render_markdown(summary):
    """Render a MeetingSummary to clean markdown."""
    lines = [f"# {summary.title}", "", summary.executive_summary, ""]

    if summary.attendees:
        lines += ["## Attendees", ""]
        lines += [f"- {a}" for a in summary.attendees]
        lines += [""]

    if summary.key_decisions:
        lines += ["## Key Decisions", ""]
        lines += [f"- {d}" for d in summary.key_decisions]
        lines += [""]

    if summary.action_items:
        lines += [
            "## Action Items",
            "",
            "| Task | Owner | Deadline | Priority |",
            "|------|-------|----------|----------|",
        ]
        for item in summary.action_items:
            deadline = item.deadline or "\u2014"
            lines.append(f"| {item.task} | {item.owner} | {deadline} | {item.priority} |")
        lines += [""]

    if summary.discussion_points:
        lines += ["## Discussion Points", ""]
        lines += [f"- {p}" for p in summary.discussion_points]
        lines += [""]

    if summary.open_questions:
        lines += ["## Open Questions", ""]
        lines += [f"- {q}" for q in summary.open_questions]
        lines += [""]

    return "\n".join(lines)


def _llm_generate(model, transcript, file_path=None):
    """Run the DSPy summarizer and return rendered markdown."""
    import dspy

    _load_env()
    lm = dspy.LM(model, system_prompt=_get_system_prompt(file_path))
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

    summarizer = _build_summarizer()
    result = summarizer(transcript=transcript)
    return _render_markdown(result.summary)


def _find_transcript(file_path):
    """Given an audio or transcript file, find the transcript .txt file."""
    file_path = Path(file_path)
    if file_path.suffix == ".txt":
        return file_path

    transcript_path = file_path.with_suffix(".txt")
    if transcript_path.exists():
        return transcript_path

    console.print(
        f"[red]No transcript found for {file_path.name}.[/red]\n"
        f"Expected: [cyan]{transcript_path}[/cyan]\n"
        f"Run [cyan]murmur transcribe {file_path}[/cyan] first."
    )
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


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
    def summarize(file, model):
        """Summarize a transcript using an LLM.

        FILE can be a transcript (.txt) or an audio file — if given an audio
        file, the command looks for a sibling .txt transcript automatically.

        Uses DSPy with chain-of-thought reasoning to produce structured output:
        title, executive summary, key decisions, action items (with owner,
        deadline, priority), discussion points, and open questions.

        Requires: uv pip install murmur[ai]
        """
        if not _check_dep():
            raise SystemExit(1)

        cfg = get_section("summarize")
        model_name = model or cfg.get("model", DEFAULT_MODEL)

        transcript_path = _find_transcript(file)
        transcript = transcript_path.read_text()
        if not transcript.strip():
            console.print("[yellow]Transcript is empty, nothing to summarize.[/yellow]")
            return

        console.print(f"[bold]Summarizing[/bold] {transcript_path} (model={model_name})")

        summary = _llm_generate(model_name, transcript, file_path=str(transcript_path))

        summary_path = transcript_path.with_suffix(".summary.md")
        summary_path.write_text(summary)

        console.print(f"\n[bold green]Summary saved:[/bold green] {summary_path}")
        console.print(f"\n{summary}")

    # Auto-summarize on transcription_complete if configured
    cfg = get_section("summarize")
    if cfg.get("auto"):

        def _auto_summarize(transcript_path, **kwargs):
            if not _check_dep():
                return
            model_name = cfg.get("model", DEFAULT_MODEL)
            transcript = Path(transcript_path).read_text()
            if not transcript.strip():
                return

            console.print(f"\n[bold]Auto-summarizing[/bold] {transcript_path}")
            summary = _llm_generate(model_name, transcript, file_path=transcript_path)

            summary_path = Path(transcript_path).with_suffix(".summary.md")
            summary_path.write_text(summary)
            console.print(f"[bold green]Auto-summary saved:[/bold green] {summary_path}")

        hooks.on("transcription_complete", _auto_summarize)
