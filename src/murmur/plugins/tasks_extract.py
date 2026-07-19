"""Murmur plugin: DSPy-based task extraction from meeting transcripts/summaries."""

# NOTE: no `from __future__ import annotations` — DSPy needs real type objects
# in Signature fields, not ForwardRef strings.

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console

from murmur.artifacts import ArtifactStore, fingerprint_file
from murmur.config import get_section

console = Console()

DEFAULT_MODEL = "gemini/gemini-3-flash-preview"

SYSTEM_PROMPT = """\
You are an expert meeting analyst specializing in task extraction. Your job is \
to identify actionable tasks, blockers, and commitments from meeting transcripts \
and summaries.

Rules:
- Extract concrete, actionable tasks — not vague discussion points.
- Attribute tasks to specific people when mentioned by name.
- Use exact names from the transcript; do not invent assignees.
- If a deadline or date is mentioned, include it.
- Distinguish between new tasks and existing blockers being raised or resolved.
- Set priority based on urgency cues (ASAP, urgent, critical = high; nice to have = low).
- If confidence is low (ambiguous language), set confidence below 0.7.
- Preserve technical terms and project names exactly as spoken.
- Classify each item as an explicit commitment or inferred suggestion.
- Include exact source segment IDs. Do not create uncited tasks.
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

_extractor_cache = None


def _build_extractor():
    """Build and return a DSPy TaskExtractor module (cached)."""
    global _extractor_cache
    if _extractor_cache is not None:
        return _extractor_cache

    import dspy
    import pydantic

    class ExtractedTask(pydantic.BaseModel):
        title: str = pydantic.Field(description="Concise description of the task")
        owner: str = pydantic.Field(
            default="Unassigned", description="Person responsible, or 'Unassigned' if unclear"
        )
        deadline: str = pydantic.Field(
            default="", description="Due date if mentioned, otherwise empty"
        )
        priority: str = pydantic.Field(
            default="normal", description="critical, high, normal, or low"
        )
        project: str = pydantic.Field(
            default="", description="Project or workstream name if mentioned"
        )
        source_excerpt: str = pydantic.Field(
            default="", description="Brief excerpt from transcript where this task was mentioned"
        )
        confidence: float = pydantic.Field(
            default=1.0, description="Confidence score 0.0-1.0 for ambiguous tasks"
        )
        commitment: str = pydantic.Field(default="inferred", description="explicit or inferred")
        source_segment_ids: list[str] = pydantic.Field(
            default_factory=list, description="Exact transcript segment IDs supporting this task"
        )

    class MeetingTaskAnalysis(pydantic.BaseModel):
        new_tasks: list[ExtractedTask] = pydantic.Field(
            default_factory=list, description="Tasks identified from the meeting"
        )
        blockers_raised: list[str] = pydantic.Field(
            default_factory=list, description="Blockers or issues raised during the meeting"
        )
        blockers_resolved: list[str] = pydantic.Field(
            default_factory=list, description="Blockers or issues that were resolved"
        )

    # Resolve annotations so DSPy sees real types, not ForwardRef
    MeetingTaskAnalysis.model_rebuild()

    class ExtractTasks(dspy.Signature):
        """Analyze a meeting transcript and extract actionable tasks and blockers."""

        transcript: str = dspy.InputField(desc="Meeting transcript or summary text")
        existing_tasks: str = dspy.InputField(desc="Current open tasks as context")
        analysis: MeetingTaskAnalysis = dspy.OutputField(
            desc="Extracted tasks, blockers raised, and blockers resolved"
        )

    class TaskExtractor(dspy.Module):
        def __init__(self):
            self.extract = dspy.ChainOfThought(ExtractTasks)

        def forward(self, transcript, existing_tasks):
            return self.extract(transcript=transcript, existing_tasks=existing_tasks)

    _extractor_cache = TaskExtractor()
    return _extractor_cache


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

    meta_path = Path(file_path).with_suffix(".json")
    if not meta_path.exists():
        return None

    try:
        import json
        from datetime import UTC, datetime

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


def _get_system_prompt(file_path=None):
    """Build system prompt from base + memory + calendar context."""
    from murmur.plugins.memory import load_memory

    parts = [SYSTEM_PROMPT]
    memory = load_memory()
    if memory:
        parts.append(
            "The user has provided the following personal context. "
            "Use it to tailor task extraction (e.g. identify their tasks, "
            "use their project names):\n\n" + memory
        )

    if file_path:
        cal_context = _get_calendar_context(file_path)
        if cal_context:
            parts.append(
                "The following meeting metadata was retrieved from the user's "
                "calendar. Use it to identify task owners and understand the "
                "meeting's purpose:\n\n" + cal_context
            )

    return "\n\n".join(parts)


def _find_input_file(file_path):
    """Given a file path, find the best text input for task extraction.

    Prefers .summary.md, then .txt transcript, then looks for siblings of audio files.
    """
    file_path = Path(file_path)

    # Resolve rendered canonical summaries back to grounded JSON when available.
    if file_path.name == "summary.md":
        canonical_json = ArtifactStore.for_input(file_path).path("summary.json")
        if canonical_json.exists():
            return canonical_json

    # If it's already another summary or transcript, use it directly.
    if file_path.suffix in (".json", ".md", ".txt"):
        if file_path.exists():
            return file_path
        raise click.ClickException(f"File not found: {file_path}")

    store = ArtifactStore(file_path)
    for canonical in (
        store.path("summary.json"),
        store.path("summary.md"),
        store.path("transcript.md"),
        store.path("transcript.txt"),
    ):
        if canonical.exists():
            return canonical

    # Backward compatibility for sibling artifacts from older Murmur versions.
    summary_path = file_path.with_suffix(".summary.md")
    if summary_path.exists():
        return summary_path

    transcript_path = file_path.with_suffix(".txt")
    if transcript_path.exists():
        return transcript_path

    raise click.ClickException(
        f"No transcript or summary found for {file_path.name}.\n"
        f"Expected: {summary_path} or {transcript_path}\n"
        f"Run 'murmur transcribe {file_path}' first."
    )


def _format_existing_tasks(tasks):
    """Format open tasks as a context string for the LLM."""
    if not tasks:
        return "No existing open tasks."

    lines = []
    for t in tasks:
        parts = [f"- [{t.priority}] {t.title}"]
        if t.owner:
            parts.append(f"(@{t.owner})")
        if t.project:
            parts.append(f"(+{t.project})")
        if t.deadline:
            parts.append(f"(due:{t.deadline})")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _source_segment_ids(payload):
    found = set()
    if isinstance(payload, dict):
        if isinstance(payload.get("id"), str) and "text" in payload and "start" in payload:
            found.add(payload["id"])
        segment_id = payload.get("segment_id")
        if isinstance(segment_id, str):
            found.add(segment_id)
        segment_ids = payload.get("segment_ids", [])
        if isinstance(segment_ids, list):
            found.update(item for item in segment_ids if isinstance(item, str))
        for value in payload.values():
            found.update(_source_segment_ids(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_source_segment_ids(value))
    return found


def _preview_payload(input_path, model_name, analysis, rejected_count=0):
    tasks = [
        {
            "title": task.title,
            "owner": task.owner,
            "deadline": task.deadline,
            "priority": task.priority,
            "project": task.project,
            "source_excerpt": task.source_excerpt,
            "confidence": task.confidence,
            "commitment": task.commitment
            if task.commitment in ("explicit", "inferred")
            else "inferred",
            "source_segment_ids": list(task.source_segment_ids),
        }
        for task in analysis.new_tasks
    ]
    identity = json.dumps(
        {"source": str(input_path), "model": model_name, "tasks": tasks},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": 1,
        "kind": "task_change_preview",
        "preview_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
        "source": str(input_path),
        "source_fingerprint": fingerprint_file(input_path),
        "model": model_name,
        "generated_at": datetime.now(UTC).isoformat(),
        "approval_required": True,
        "applied_at": None,
        "new_tasks": tasks,
        "rejected_uncited_tasks": rejected_count,
        "blockers_raised": list(analysis.blockers_raised),
        "blockers_resolved": list(analysis.blockers_resolved),
    }


def _extract_tasks(file_path, model=None):
    """Extract and persist a reviewable preview without mutating a task backend."""
    import dspy

    from murmur.plugins.tasks import load_tasks

    cfg = get_section("tasks")
    model_name = model or cfg.get("model", DEFAULT_MODEL)

    # Find and read input file
    input_path = _find_input_file(file_path)
    text = input_path.read_text()
    if not text.strip():
        raise click.ClickException(f"File is empty: {input_path}")

    # Load existing open tasks as context
    all_tasks = load_tasks()
    open_tasks = [t for t in all_tasks if t.status not in ("done", "dropped")]
    existing_context = _format_existing_tasks(open_tasks)

    # Configure DSPy and run extraction
    _load_env()
    lm = dspy.LM(model_name, system_prompt=_get_system_prompt(file_path=str(input_path)))
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

    extractor = _build_extractor()
    result = extractor(transcript=text, existing_tasks=existing_context)
    analysis = result.analysis
    allowed_ids = set()
    if input_path.suffix == ".json":
        try:
            allowed_ids = _source_segment_ids(json.loads(text))
        except json.JSONDecodeError:
            allowed_ids = set()
    rejected_count = 0
    if input_path.suffix == ".json":
        supported = []
        for task in analysis.new_tasks:
            if set(task.source_segment_ids) & allowed_ids:
                task.source_segment_ids = [
                    segment_id
                    for segment_id in task.source_segment_ids
                    if segment_id in allowed_ids
                ]
                supported.append(task)
            else:
                rejected_count += 1
        analysis.new_tasks = supported

    store = ArtifactStore.for_input(input_path)
    preview = _preview_payload(input_path, model_name, analysis, rejected_count)
    preview_path = store.write_json("tasks.preview.json", preview)
    store.register_artifact(
        "task_change_preview",
        preview_path,
        kind="task_change_preview",
        provenance={
            "model": model_name,
            "source_sha256": preview["source_fingerprint"]["sha256"],
            "approval_required": True,
        },
    )

    return analysis


def _apply_task_preview(file_path):
    """Apply the exact persisted preview after an explicit second-step approval."""
    from murmur.plugins.tasks import Task, load_tasks, save_tasks

    input_path = _find_input_file(file_path)
    store = ArtifactStore.for_input(input_path)
    preview_path = store.path("tasks.preview.json")
    try:
        preview = json.loads(preview_path.read_text())
    except FileNotFoundError as error:
        raise click.ClickException(
            "No task preview exists. Run `murmur tasks ingest FILE` first."
        ) from error
    if preview.get("source") != str(input_path):
        raise click.ClickException("Task preview belongs to a different source file.")
    current_sha = fingerprint_file(input_path)["sha256"]
    if preview.get("source_fingerprint", {}).get("sha256") != current_sha:
        raise click.ClickException("Source changed after preview; generate a new task preview.")
    if preview.get("applied_at"):
        return 0
    all_tasks = load_tasks()
    for proposed in preview.get("new_tasks", []):
        task = Task.new(
            proposed["title"],
            owner=proposed.get("owner", "") if proposed.get("owner") != "Unassigned" else "",
            priority=proposed.get("priority")
            if proposed.get("priority") in ("critical", "high", "normal", "low")
            else "normal",
            project=proposed.get("project", ""),
            deadline=proposed.get("deadline", ""),
            source_file=str(input_path),
            tags=["murmur", proposed.get("commitment", "inferred")],
        )
        all_tasks.append(task)
    save_tasks(all_tasks)
    preview["applied_at"] = datetime.now(UTC).isoformat()
    preview["approval_required"] = False
    preview_path = store.write_json("tasks.preview.json", preview)
    store.register_artifact(
        "task_change_preview",
        preview_path,
        kind="task_change_preview_applied",
        provenance={"preview_id": preview["preview_id"], "approved": True},
    )
    return len(preview.get("new_tasks", []))


# ---------------------------------------------------------------------------
# Phase 3: Cross-meeting task matching
# ---------------------------------------------------------------------------

_matcher_cache = None


def _build_matcher():
    """Build and return a DSPy MatchTaskMention module (cached)."""
    global _matcher_cache
    if _matcher_cache is not None:
        return _matcher_cache

    import dspy
    import pydantic

    class TaskStatusUpdate(pydantic.BaseModel):
        task_id: str = pydantic.Field(
            default="", description="ID of the matched existing task, or empty if no match"
        )
        new_status: str = pydantic.Field(
            default="", description="Updated status if mentioned (done, active, etc.)"
        )
        new_deadline: str = pydantic.Field(default="", description="Updated deadline if mentioned")
        discussion_context: str = pydantic.Field(
            default="", description="Brief summary of what was said about this task"
        )
        confidence: float = pydantic.Field(
            default=1.0, description="Confidence that this is the right match (0.0-1.0)"
        )

    class MatchTaskMention(dspy.Signature):
        """Match a task mention from a meeting to an existing task."""

        mention: str = dspy.InputField(desc="What was said about a task in the meeting")
        candidates: str = dspy.InputField(desc="Existing tasks as JSON")
        match: TaskStatusUpdate = dspy.OutputField(
            desc="Matched task update, or empty task_id if no match"
        )

    class TaskMatcher(dspy.Module):
        def __init__(self):
            self.match = dspy.ChainOfThought(MatchTaskMention)

        def forward(self, mention, candidates):
            return self.match(mention=mention, candidates=candidates)

    _matcher_cache = TaskMatcher()
    return _matcher_cache


def _match_extracted_to_existing(extracted_tasks, existing_tasks, model=None):
    """Try to match extracted tasks against existing ones using DSPy.

    Returns (new_tasks, updates) where new_tasks are truly new and
    updates are (existing_task, update_info) pairs.
    """
    import json

    import dspy

    if not existing_tasks or not extracted_tasks:
        return extracted_tasks, []

    cfg = get_section("tasks")
    model_name = model or cfg.get("model", DEFAULT_MODEL)

    _load_env()
    lm = dspy.LM(model_name, system_prompt=_get_system_prompt())
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

    candidates_json = json.dumps(
        [
            {"id": t.id, "title": t.title, "owner": t.owner, "project": t.project}
            for t in existing_tasks
        ]
    )

    matcher = _build_matcher()
    new_tasks = []
    updates = []

    for extracted in extracted_tasks:
        try:
            result = matcher(mention=extracted.title, candidates=candidates_json)
            update = result.match
            if update.task_id and update.confidence >= 0.7:
                # Find the matched existing task
                matched = None
                for t in existing_tasks:
                    if t.id.startswith(update.task_id) or update.task_id.startswith(t.id):
                        matched = t
                        break
                if matched:
                    updates.append((matched, update))
                    continue
        except Exception:  # noqa: S110
            pass
        new_tasks.append(extracted)

    return new_tasks, updates


def _write_tasks_json(file_path, analysis, updates=None):
    """Write a .tasks.json sidecar file next to the recording."""
    import json

    sidecar = Path(file_path).with_suffix(".tasks.json")
    data = {
        "source": str(file_path),
        "new_tasks": [
            {
                "title": t.title,
                "owner": t.owner,
                "deadline": t.deadline,
                "priority": t.priority,
                "project": t.project,
                "confidence": t.confidence,
                "commitment": t.commitment,
                "source_segment_ids": t.source_segment_ids,
                "source_excerpt": t.source_excerpt,
            }
            for t in analysis.new_tasks
        ],
        "blockers_raised": analysis.blockers_raised,
        "blockers_resolved": analysis.blockers_resolved,
    }
    if updates:
        data["task_updates"] = [
            {
                "task_id": task.id,
                "title": task.title,
                "new_status": update.new_status,
                "new_deadline": update.new_deadline,
                "context": update.discussion_context,
            }
            for task, update in updates
        ]
    sidecar.write_text(json.dumps(data, indent=2) + "\n")
    return sidecar


def _auto_extract(summary_path, source_file=None, **_kwargs):
    """Hook handler: create a task preview, never mutate a backend automatically."""
    if not _check_dep():
        return

    try:
        analysis = _extract_tasks(summary_path)
        console.print(
            f"[green]Prepared {len(analysis.new_tasks)} task change(s) for review[/green] "
            f"from {Path(summary_path).name}; run `murmur tasks ingest {summary_path} --approve` "
            "to apply the saved preview."
        )
    except Exception as exc:
        console.print(f"[yellow]Task auto-extraction failed:[/yellow] {exc}")


def register_hooks():
    """Register auto-extraction hooks if enabled in config."""
    cfg = get_section("tasks")
    if cfg.get("auto"):
        from murmur import hooks

        hooks.on("summary_complete", _auto_extract)
