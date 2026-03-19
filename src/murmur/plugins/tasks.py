"""Murmur plugin: task management with todo.txt and TaskWarrior backends."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from murmur.config import get_section

console = Console()

# Priority mapping: A=critical, B=high, C=normal, D=low
PRIORITY_TO_LETTER = {"critical": "A", "high": "B", "normal": "C", "low": "D"}
LETTER_TO_PRIORITY = {v: k for k, v in PRIORITY_TO_LETTER.items()}

VALID_STATUSES = ("inbox", "next", "active", "waiting", "done", "dropped")
VALID_PRIORITIES = ("critical", "high", "normal", "low")

STATUS_STYLES = {
    "inbox": "dim",
    "next": "cyan",
    "active": "bold green",
    "waiting": "yellow",
    "done": "dim strikethrough",
    "dropped": "dim red",
}

DEFAULT_TASKS_FILE = Path.home() / ".local" / "share" / "murmur" / "tasks.txt"
TASK_CONTEXT_PATH = Path.home() / ".config" / "murmur" / "task_context.md"


def _tasks_path() -> Path:
    """Get the tasks file path from config or default."""
    cfg = get_section("tasks")
    raw = cfg.get("file", str(DEFAULT_TASKS_FILE))
    return Path(raw).expanduser()


@dataclass
class Task:
    """A single task."""

    id: str
    title: str
    status: str = "inbox"
    priority: str = "normal"
    owner: str = ""
    project: str = ""
    deadline: str = ""
    created_at: str = ""
    source_file: str = ""
    tags: list[str] = field(default_factory=list)

    @staticmethod
    def new(
        title: str,
        *,
        status: str = "inbox",
        priority: str = "normal",
        owner: str = "",
        project: str = "",
        deadline: str = "",
        source_file: str = "",
        tags: list[str] | None = None,
    ) -> Task:
        """Create a new task with generated id and timestamp."""
        return Task(
            id=uuid.uuid4().hex[:8],
            title=title,
            status=status,
            priority=priority,
            owner=owner,
            project=project,
            deadline=deadline,
            created_at=datetime.now(UTC).strftime("%Y-%m-%d"),
            source_file=source_file,
            tags=tags or [],
        )


# ---------------------------------------------------------------------------
# todo.txt serialization
# ---------------------------------------------------------------------------


def task_to_line(task: Task) -> str:
    """Serialize a Task to a todo.txt line."""
    parts: list[str] = []

    # Done/dropped prefix
    if task.status == "done":
        parts.append("x")
        parts.append(datetime.now(UTC).strftime("%Y-%m-%d"))

    # Priority
    letter = PRIORITY_TO_LETTER.get(task.priority, "C")
    parts.append(f"({letter})")

    # Created date
    if task.created_at:
        parts.append(task.created_at)

    # Title
    parts.append(task.title)

    # Project
    if task.project:
        parts.append(f"+{task.project}")

    # Owner
    if task.owner:
        parts.append(f"@{task.owner}")

    # Tags
    for tag in task.tags:
        parts.append(f"+{tag}")

    # Key:value metadata
    parts.append(f"id:{task.id}")
    parts.append(f"status:{task.status}")

    if task.deadline:
        parts.append(f"due:{task.deadline}")

    if task.source_file:
        parts.append(f"src:{task.source_file}")

    return " ".join(parts)


def line_to_task(line: str) -> Task | None:
    """Parse a todo.txt line into a Task. Returns None for blank/comment lines."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    tokens = line.split()
    idx = 0
    is_done = False

    # Check for done prefix
    if tokens[idx] == "x":
        is_done = True
        idx += 1
        # Skip completion date
        if idx < len(tokens) and _is_date(tokens[idx]):
            idx += 1

    # Priority
    priority = "normal"
    if (
        idx < len(tokens)
        and len(tokens[idx]) == 3
        and tokens[idx][0] == "("
        and tokens[idx][2] == ")"
    ):
        letter = tokens[idx][1]
        priority = LETTER_TO_PRIORITY.get(letter, "normal")
        idx += 1

    # Created date
    created_at = ""
    if idx < len(tokens) and _is_date(tokens[idx]):
        created_at = tokens[idx]
        idx += 1

    # Parse remaining tokens
    title_parts: list[str] = []
    task_id = ""
    status = "done" if is_done else "inbox"
    owner = ""
    project = ""
    deadline = ""
    source_file = ""
    tags: list[str] = []

    for token in tokens[idx:]:
        if token.startswith("id:"):
            task_id = token[3:]
        elif token.startswith("status:"):
            status = token[7:]
        elif token.startswith("due:"):
            deadline = token[4:]
        elif token.startswith("src:"):
            source_file = token[4:]
        elif token.startswith("@"):
            owner = token[1:]
        elif token.startswith("+"):
            # First +tag is the project, rest are tags
            if not project:
                project = token[1:]
            else:
                tags.append(token[1:])
        else:
            title_parts.append(token)

    if not task_id:
        task_id = uuid.uuid4().hex[:8]

    return Task(
        id=task_id,
        title=" ".join(title_parts),
        status=status,
        priority=priority,
        owner=owner,
        project=project,
        deadline=deadline,
        created_at=created_at,
        source_file=source_file,
        tags=tags,
    )


def _is_date(s: str) -> bool:
    """Check if a string looks like YYYY-MM-DD."""
    if len(s) != 10:
        return False
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Backend: todo.txt
# ---------------------------------------------------------------------------


def _todo_load() -> list[Task]:
    """Load all tasks from the todo.txt file."""
    path = _tasks_path()
    if not path.exists():
        return []

    tasks = []
    for line in path.read_text().splitlines():
        task = line_to_task(line)
        if task is not None:
            tasks.append(task)
    return tasks


def _todo_save(tasks: list[Task]) -> None:
    """Write all tasks to the todo.txt file."""
    path = _tasks_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [task_to_line(t) for t in tasks]
    path.write_text("\n".join(lines) + "\n" if lines else "")


def _todo_add(task: Task) -> None:
    """Add a single task to the todo.txt file."""
    tasks = _todo_load()
    tasks.append(task)
    _todo_save(tasks)


def _todo_update(task: Task) -> None:
    """Update a task in the todo.txt file (matched by id)."""
    tasks = _todo_load()
    for i, t in enumerate(tasks):
        if t.id == task.id:
            tasks[i] = task
            break
    _todo_save(tasks)


def _todo_find(task_id: str) -> Task | None:
    """Find a task by id or prefix in the todo.txt file."""
    for t in _todo_load():
        if t.id == task_id or t.id.startswith(task_id):
            return t
    return None


def _todo_backend() -> dict:
    """Return the todo.txt backend as a dict of callables."""
    return {
        "load": _todo_load,
        "save": _todo_save,
        "add": _todo_add,
        "update": _todo_update,
        "find": _todo_find,
    }


# ---------------------------------------------------------------------------
# Backend: TaskWarrior
# ---------------------------------------------------------------------------

# GTD status -> TaskWarrior mapping: (tw_status, extra_tags)
_STATUS_TO_TW = {
    "inbox": ("pending", set()),
    "next": ("pending", {"next"}),
    "active": ("started", set()),
    "waiting": ("waiting", set()),
    "done": ("completed", set()),
    "dropped": ("deleted", set()),
}

# TaskWarrior status -> GTD status (default; "next" tag overrides pending)
_TW_TO_STATUS = {
    "pending": "inbox",
    "started": "active",
    "waiting": "waiting",
    "completed": "done",
    "deleted": "dropped",
}

# Priority: murmur -> TaskWarrior
_PRIORITY_TO_TW = {"critical": "H", "high": "H", "normal": "M", "low": "L"}
_TW_TO_PRIORITY = {"H": "high", "M": "normal", "L": "low"}


def _tw_task_to_task(tw_task) -> Task:
    """Convert a tasklib Task object to our Task dataclass."""
    tw_status = str(tw_task["status"])
    tw_tags = set(tw_task["tags"] or [])

    # Determine GTD status
    status = _TW_TO_STATUS.get(tw_status, "inbox")
    if tw_status == "pending" and "next" in tw_tags:
        status = "next"

    # Priority
    tw_pri = tw_task["priority"] or ""
    priority = _TW_TO_PRIORITY.get(tw_pri, "normal")

    # Deadline
    deadline = ""
    if tw_task["due"]:
        deadline = tw_task["due"].strftime("%Y-%m-%d")

    # Created date
    created_at = ""
    if tw_task["entry"]:
        created_at = tw_task["entry"].strftime("%Y-%m-%d")

    # Tags: exclude internal ones (next, default_tags)
    cfg = get_section("tasks")
    tw_cfg = cfg.get("taskwarrior", {})
    default_tags = set(tw_cfg.get("default_tags", ["murmur"]))
    user_tags = sorted(tw_tags - {"next"} - default_tags)

    # Use the TaskWarrior UUID short form as our id
    task_id = str(tw_task["uuid"])[:8]

    return Task(
        id=task_id,
        title=str(tw_task["description"]),
        status=status,
        priority=priority,
        owner=str(tw_task.get("owner") or tw_task.get("assigned") or ""),
        project=str(tw_task["project"] or ""),
        deadline=deadline,
        created_at=created_at,
        source_file=str(tw_task.get("source") or ""),
        tags=user_tags,
    )


def _task_to_tw_kwargs(task: Task) -> dict:
    """Convert our Task to keyword arguments for tasklib."""
    from datetime import datetime as dt

    cfg = get_section("tasks")
    tw_cfg = cfg.get("taskwarrior", {})
    default_project = tw_cfg.get("default_project", "meetings")
    default_tags = list(tw_cfg.get("default_tags", ["murmur"]))

    kwargs: dict = {
        "description": task.title,
        "project": task.project or default_project,
        "priority": _PRIORITY_TO_TW.get(task.priority),
    }

    # Tags
    tags = list(task.tags) + default_tags
    _, extra_tags = _STATUS_TO_TW.get(task.status, ("pending", set()))
    tags.extend(extra_tags)
    kwargs["tags"] = sorted(set(tags))

    # Deadline
    if task.deadline:
        kwargs["due"] = dt.strptime(task.deadline, "%Y-%m-%d").replace(tzinfo=UTC)

    return kwargs


def _get_tw():
    """Lazy import and return tasklib.TaskWarrior instance."""
    try:
        from tasklib import TaskWarrior
    except ImportError as err:
        raise click.ClickException(
            "tasklib is not installed. Install with: uv pip install murmur[tasks-tw]"
        ) from err
    return TaskWarrior()


def _tw_load() -> list[Task]:
    """Load tasks from TaskWarrior."""
    tw = _get_tw()
    cfg = get_section("tasks")
    tw_cfg = cfg.get("taskwarrior", {})
    default_tags = tw_cfg.get("default_tags", ["murmur"])

    # Only load tasks with our default tags
    tasks = tw.tasks.filter(tags__contains=default_tags)
    return [_tw_task_to_task(t) for t in tasks]


def _tw_save(tasks: list[Task]) -> None:
    """Bulk save is not supported with the TaskWarrior backend."""
    raise click.ClickException(
        "Bulk save is not supported with the TaskWarrior backend. "
        "Use add/update operations instead."
    )


def _tw_add(task: Task) -> None:
    """Add a single task to TaskWarrior."""
    from tasklib import Task as TWTask

    tw = _get_tw()
    kwargs = _task_to_tw_kwargs(task)
    tw_task = TWTask(tw, **kwargs)
    tw_task.save()

    # Handle status transitions
    tw_status, _ = _STATUS_TO_TW.get(task.status, ("pending", set()))
    if tw_status == "started":
        tw_task.start()
    elif tw_status == "completed":
        tw_task.done()
    elif tw_status == "deleted":
        tw_task.delete()


def _tw_update(task: Task) -> None:
    """Update an existing task in TaskWarrior (matched by id prefix on UUID)."""
    tw = _get_tw()
    matches = [t for t in tw.tasks.all() if str(t["uuid"]).startswith(task.id)]
    if not matches:
        raise click.ClickException(f"TaskWarrior task '{task.id}' not found.")

    tw_task = matches[0]
    kwargs = _task_to_tw_kwargs(task)
    for key, val in kwargs.items():
        tw_task[key] = val
    tw_task.save()

    # Handle status transitions
    tw_status, _ = _STATUS_TO_TW.get(task.status, ("pending", set()))
    current_tw_status = str(tw_task["status"])
    if tw_status == "completed" and current_tw_status != "completed":
        tw_task.done()
    elif tw_status == "deleted" and current_tw_status != "deleted":
        tw_task.delete()
    elif tw_status == "started" and current_tw_status != "started":
        tw_task.start()


def _tw_find(task_id: str) -> Task | None:
    """Find a task in TaskWarrior by UUID prefix."""
    tw = _get_tw()
    cfg = get_section("tasks")
    tw_cfg = cfg.get("taskwarrior", {})
    default_tags = tw_cfg.get("default_tags", ["murmur"])

    for t in tw.tasks.filter(tags__contains=default_tags):
        if str(t["uuid"]).startswith(task_id):
            return _tw_task_to_task(t)
    return None


def _tw_backend() -> dict:
    """Return the TaskWarrior backend as a dict of callables."""
    return {
        "load": _tw_load,
        "save": _tw_save,
        "add": _tw_add,
        "update": _tw_update,
        "find": _tw_find,
    }


# ---------------------------------------------------------------------------
# Backend dispatcher
# ---------------------------------------------------------------------------


def _get_backend() -> dict:
    """Get the configured backend as a dict of callables."""
    cfg = get_section("tasks")
    backend = cfg.get("backend", "todo")
    if backend == "taskwarrior":
        return _tw_backend()
    return _todo_backend()


# Public API (used by CLI commands and other plugins)
def load_tasks() -> list[Task]:
    """Load all tasks from the configured backend."""
    return _get_backend()["load"]()


def save_tasks(tasks: list[Task]) -> None:
    """Save all tasks to the configured backend."""
    _get_backend()["save"](tasks)


def find_task(task_id: str, tasks: list[Task] | None = None) -> Task | None:
    """Find a task by its ID (or prefix)."""
    if tasks is not None:
        for t in tasks:
            if t.id == task_id or t.id.startswith(task_id):
                return t
        return None
    return _get_backend()["find"](task_id)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _export_task_context() -> Path:
    """Export open tasks to task_context.md, grouped by project and owner."""
    tasks = load_tasks()
    open_tasks = [t for t in tasks if t.status not in ("done", "dropped")]

    # Group by project, then by owner
    by_project: dict[str, dict[str, list[Task]]] = {}
    for t in open_tasks:
        proj = t.project or ""
        owner = t.owner or ""
        by_project.setdefault(proj, {}).setdefault(owner, []).append(t)

    lines = ["# Open Tasks", ""]

    # Named projects first, then unassigned project
    named_projects = sorted(k for k in by_project if k)
    has_unassigned = "" in by_project

    for proj in named_projects:
        lines.append(f"## Project: {proj}")
        owners = by_project[proj]
        for owner in sorted(owners):
            if owner:
                lines.append(f"### Owner: {owner}")
            else:
                lines.append("### Unassigned Owner")
            for t in owners[owner]:
                lines.append(f"- {_format_task_line(t)}")
        lines.append("")

    if has_unassigned:
        owners = by_project[""]
        lines.append("## Unassigned")
        for owner in sorted(owners):
            if owner:
                lines.append(f"### Owner: {owner}")
            for t in owners[owner]:
                lines.append(f"- {_format_task_line(t)}")
        lines.append("")

    TASK_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TASK_CONTEXT_PATH.write_text("\n".join(lines))
    return TASK_CONTEXT_PATH


def _format_task_line(task: Task) -> str:
    """Format a single task as a markdown list item body."""
    parts = [f"[{task.status}] {task.title}"]
    if task.deadline:
        parts.append(f"(due: {task.deadline})")
    for tag in task.tags:
        parts.append(f"#{tag}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_EM_DASH = "\u2014"


def _render_table(tasks: list[Task], *, title: str = "Tasks") -> Table:
    """Build a Rich table from a list of tasks."""
    table = Table(title=title, show_lines=False)
    table.add_column("ID", style="dim", width=10)
    table.add_column("Pri", width=4)
    table.add_column("Status", width=10)
    table.add_column("Title")
    table.add_column("Owner", style="blue")
    table.add_column("Project", style="magenta")
    table.add_column("Due", style="red")

    pri_style = {"critical": "bold red", "high": "red", "normal": "", "low": "dim"}

    for task in tasks:
        style = STATUS_STYLES.get(task.status, "")
        p_letter = PRIORITY_TO_LETTER.get(task.priority, "C")
        table.add_row(
            task.id,
            f"[{pri_style.get(task.priority, '')}]{p_letter}[/]",
            f"[{style}]{task.status}[/]",
            f"[{style}]{task.title}[/]",
            task.owner,
            task.project,
            task.deadline,
        )

    return table


def _show_detail(task: Task) -> None:
    """Print detailed view of a single task."""
    style = STATUS_STYLES.get(task.status, "")
    console.print(f"[bold]Task {task.id}[/bold]")
    console.print(f"  Title:    [{style}]{task.title}[/]")
    console.print(f"  Status:   [{style}]{task.status}[/]")
    console.print(f"  Priority: {task.priority}")
    console.print(f"  Owner:    {task.owner or _EM_DASH}")
    console.print(f"  Project:  {task.project or _EM_DASH}")
    console.print(f"  Deadline: {task.deadline or _EM_DASH}")
    console.print(f"  Created:  {task.created_at}")
    if task.source_file:
        console.print(f"  Source:   {task.source_file}")
    if task.tags:
        console.print(f"  Tags:     {', '.join(task.tags)}")


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

_BULLET = "\u2022"


def register(cli: click.Group) -> None:
    """Register the tasks command group."""

    @cli.group(invoke_without_command=True)
    @click.pass_context
    def tasks(ctx):
        """Manage tasks extracted from meetings."""
        if ctx.invoked_subcommand is not None:
            return

        # Default: list non-done, non-dropped tasks
        all_tasks = load_tasks()
        active = [t for t in all_tasks if t.status not in ("done", "dropped")]
        if active:
            console.print(_render_table(active))
        else:
            console.print("[dim]No active tasks.[/dim]")
            console.print('Run [cyan]murmur tasks add "title"[/cyan] to create one.')

    @tasks.command()
    @click.argument("title")
    @click.option("-o", "--owner", default="", help="Person responsible.")
    @click.option("-p", "--project", default="", help="Project or workstream.")
    @click.option(
        "--priority",
        type=click.Choice(VALID_PRIORITIES, case_sensitive=False),
        default="normal",
        help="Task priority.",
    )
    @click.option(
        "--status",
        type=click.Choice(VALID_STATUSES, case_sensitive=False),
        default="inbox",
        help="Initial status.",
    )
    @click.option("--deadline", default="", help="Due date (YYYY-MM-DD).")
    @click.option("--source", default="", help="Source recording file.")
    @click.option("-t", "--tag", multiple=True, help="Tags (repeatable).")
    def add(title, owner, project, priority, status, deadline, source, tag):
        """Add a new task."""
        task = Task.new(
            title,
            status=status,
            priority=priority,
            owner=owner,
            project=project,
            deadline=deadline,
            source_file=source,
            tags=list(tag),
        )
        backend = _get_backend()
        backend["add"](task)
        console.print(f"[green]Added[/green] task [bold]{task.id}[/bold]: {task.title}")

    @tasks.command("list")
    @click.option("-s", "--status", default=None, help="Filter by status.")
    @click.option("-o", "--owner", default=None, help="Filter by owner.")
    @click.option("-p", "--project", default=None, help="Filter by project.")
    @click.option("--all", "show_all", is_flag=True, help="Include done/dropped tasks.")
    def list_tasks(status, owner, project, show_all):
        """List tasks with optional filters."""
        all_tasks = load_tasks()

        filtered = all_tasks
        if status:
            filtered = [t for t in filtered if t.status == status.lower()]
        elif not show_all:
            filtered = [t for t in filtered if t.status not in ("done", "dropped")]

        if owner:
            filtered = [t for t in filtered if t.owner.lower() == owner.lower()]
        if project:
            filtered = [t for t in filtered if t.project.lower() == project.lower()]

        if filtered:
            console.print(_render_table(filtered))
        else:
            console.print("[dim]No matching tasks.[/dim]")

    @tasks.command()
    @click.argument("task_id")
    def done(task_id):
        """Mark a task as done."""
        backend = _get_backend()
        task = find_task(task_id)
        if not task:
            console.print(f"[red]Task '{task_id}' not found.[/red]")
            raise SystemExit(1)
        task.status = "done"
        backend["update"](task)
        console.print(f"[green]Done:[/green] [dim strikethrough]{task.title}[/]")

    @tasks.command()
    @click.argument("task_id")
    def drop(task_id):
        """Mark a task as dropped."""
        backend = _get_backend()
        task = find_task(task_id)
        if not task:
            console.print(f"[red]Task '{task_id}' not found.[/red]")
            raise SystemExit(1)
        task.status = "dropped"
        backend["update"](task)
        console.print(f"[dim red]Dropped:[/dim red] {task.title}")

    @tasks.command()
    @click.argument("task_id")
    @click.argument("new_status", type=click.Choice(VALID_STATUSES, case_sensitive=False))
    def move(task_id, new_status):
        """Change a task's status."""
        backend = _get_backend()
        task = find_task(task_id)
        if not task:
            console.print(f"[red]Task '{task_id}' not found.[/red]")
            raise SystemExit(1)
        old = task.status
        task.status = new_status.lower()
        backend["update"](task)
        style = STATUS_STYLES.get(task.status, "")
        console.print(
            f"Moved [bold]{task.id}[/bold]: "
            f"[{STATUS_STYLES.get(old, '')}]{old}[/] -> [{style}]{task.status}[/]"
        )

    @tasks.command()
    @click.argument("task_id")
    def show(task_id):
        """Show details of a task."""
        task = find_task(task_id)
        if not task:
            console.print(f"[red]Task '{task_id}' not found.[/red]")
            raise SystemExit(1)
        _show_detail(task)

    @tasks.command()
    @click.argument("file", type=click.Path(exists=True))
    @click.option(
        "--dry-run",
        is_flag=True,
        default=False,
        help="Preview extracted tasks without saving.",
    )
    @click.option(
        "-m",
        "--model",
        default=None,
        help="LLM model override. Any litellm model string.",
    )
    def ingest(file, dry_run, model):
        """Extract tasks from a meeting transcript or summary.

        FILE can be a .summary.md, .txt transcript, or an audio file -- if given
        an audio file, the command looks for a sibling summary or transcript.

        Requires: uv pip install murmur[tasks]
        """
        from murmur.plugins.tasks_extract import _check_dep, _extract_tasks

        if not _check_dep():
            raise SystemExit(1)

        mode = "DRY RUN -- " if dry_run else ""
        console.print(f"[bold]{mode}Extracting tasks from[/bold] {file}")

        analysis = _extract_tasks(file, model=model, dry_run=dry_run)

        # Display extracted tasks
        if analysis.new_tasks:
            table = Table(title="Extracted Tasks", show_lines=False)
            table.add_column("#", style="dim", width=4)
            table.add_column("Title")
            table.add_column("Owner", style="blue")
            table.add_column("Priority", width=8)
            table.add_column("Project", style="magenta")
            table.add_column("Deadline", style="red")
            table.add_column("Conf", style="dim", width=5)

            pri_style = {
                "critical": "bold red",
                "high": "red",
                "normal": "",
                "low": "dim",
            }

            for i, extracted in enumerate(analysis.new_tasks, 1):
                style = pri_style.get(extracted.priority, "")
                conf = f"{extracted.confidence:.0%}" if extracted.confidence < 1.0 else ""
                table.add_row(
                    str(i),
                    extracted.title,
                    extracted.owner,
                    f"[{style}]{extracted.priority}[/]",
                    extracted.project,
                    extracted.deadline,
                    conf,
                )
            console.print(table)
        else:
            console.print("[dim]No tasks extracted.[/dim]")

        # Display blockers
        if analysis.blockers_raised:
            console.print("\n[bold yellow]Blockers Raised:[/bold yellow]")
            for b in analysis.blockers_raised:
                console.print(f"  [yellow]![/yellow] {b}")

        if analysis.blockers_resolved:
            console.print("\n[bold green]Blockers Resolved:[/bold green]")
            for b in analysis.blockers_resolved:
                console.print(f"  [green]\u2713[/green] {b}")

        # Summary
        count = len(analysis.new_tasks)
        if dry_run:
            console.print(f"\n[dim]Dry run: {count} task(s) found, none saved.[/dim]")
        elif count:
            console.print(f"\n[bold green]{count} task(s) created.[/bold green]")

    @tasks.command()
    @click.argument("task_id")
    @click.option("--title", default=None, help="New title.")
    @click.option("-o", "--owner", default=None, help="New owner.")
    @click.option("-p", "--project", default=None, help="New project.")
    @click.option(
        "--priority",
        type=click.Choice(VALID_PRIORITIES, case_sensitive=False),
        default=None,
        help="New priority.",
    )
    @click.option("--deadline", default=None, help="New deadline.")
    @click.option("-t", "--tag", multiple=True, help="Replace tags (repeatable).")
    def edit(task_id, title, owner, project, priority, deadline, tag):
        """Edit fields of an existing task."""
        backend = _get_backend()
        task = find_task(task_id)
        if not task:
            console.print(f"[red]Task '{task_id}' not found.[/red]")
            raise SystemExit(1)

        if title is not None:
            task.title = title
        if owner is not None:
            task.owner = owner
        if project is not None:
            task.project = project
        if priority is not None:
            task.priority = priority.lower()
        if deadline is not None:
            task.deadline = deadline
        if tag:
            task.tags = list(tag)

        backend["update"](task)
        console.print(f"[green]Updated[/green] task [bold]{task.id}[/bold]")
        _show_detail(task)

    @tasks.command()
    def agenda():
        """Pre-meeting briefing: tasks relevant to next calendar event."""
        try:
            from murmur.plugins.calendar import get_next_event
        except ImportError as err:
            raise click.ClickException(
                "Calendar plugin not installed. Install with: uv sync --extra calendar"
            ) from err

        event = get_next_event()
        if not event:
            console.print("[dim]No upcoming events found.[/dim]")
            return

        attendees = [a.lower() for a in event.get("attendees", [])]
        title = event.get("title", "Unknown")
        start = event.get("start", "")

        attendee_str = ", ".join(event.get("attendees", [])) or _EM_DASH
        console.print(
            Panel(
                f"[bold]{title}[/bold]\nTime: {start}\nAttendees: {attendee_str}",
                title="Next Meeting",
            )
        )

        all_tasks = load_tasks()
        open_tasks = [t for t in all_tasks if t.status not in ("done", "dropped")]

        relevant = []
        for t in open_tasks:
            owner_match = t.owner and t.owner.lower() in " ".join(attendees)
            if owner_match:
                relevant.append(t)

        if relevant:
            console.print(_render_table(relevant, title="Relevant Tasks"))
        else:
            console.print("[dim]No tasks matched to attendees.[/dim]")

    @tasks.command()
    @click.option("--days", default=1, help="Lookback period in days.", show_default=True)
    def standup(days):
        """Generate standup from recent task activity."""
        all_tasks = load_tasks()
        cutoff = (date.today() - timedelta(days=days)).isoformat()

        done_recently = [t for t in all_tasks if t.status == "done" and t.created_at >= cutoff]
        working = [t for t in all_tasks if t.status in ("active", "next")]
        blocked = [t for t in all_tasks if t.status == "waiting"]

        if done_recently:
            console.print(f"\n[bold green]Done (last {days}d)[/bold green]")
            for t in done_recently:
                owner = f" @{t.owner}" if t.owner else ""
                console.print(f"  [dim]{_BULLET}[/dim] {t.title}{owner}")
        else:
            console.print(f"\n[dim]Nothing completed in the last {days}d.[/dim]")

        if working:
            console.print("\n[bold cyan]Working On[/bold cyan]")
            for t in working:
                pri = f" [{t.priority}]" if t.priority != "normal" else ""
                owner = f" @{t.owner}" if t.owner else ""
                console.print(f"  [dim]{_BULLET}[/dim] {t.title}{pri}{owner}")

        if blocked:
            console.print("\n[bold yellow]Blocked / Waiting[/bold yellow]")
            for t in blocked:
                owner = f" @{t.owner}" if t.owner else ""
                console.print(f"  [dim]{_BULLET}[/dim] {t.title}{owner}")

        if not done_recently and not working and not blocked:
            console.print("[dim]No task activity to report.[/dim]")

    @tasks.command()
    def review():
        """GTD weekly review: stale, overdue, and untriaged tasks."""
        all_tasks = load_tasks()
        today = date.today()
        two_days_ago = (today - timedelta(days=2)).isoformat()
        seven_days_ago = (today - timedelta(days=7)).isoformat()
        today_str = today.isoformat()

        inbox_stale = [
            t
            for t in all_tasks
            if t.status == "inbox" and t.created_at and t.created_at <= two_days_ago
        ]
        stale = [
            t
            for t in all_tasks
            if t.status in ("active", "next") and t.created_at and t.created_at <= seven_days_ago
        ]
        overdue = [
            t
            for t in all_tasks
            if t.deadline and t.deadline < today_str and t.status not in ("done", "dropped")
        ]
        waiting = [t for t in all_tasks if t.status == "waiting"]

        counts = {
            "Inbox (needs triage)": len(inbox_stale),
            "Stale (>7d)": len(stale),
            "Overdue": len(overdue),
            "Blocked/Waiting": len(waiting),
        }
        total = sum(counts.values())

        console.print(f"\n[bold]Weekly Review[/bold] {_EM_DASH} {total} items need attention\n")
        for label, count in counts.items():
            style = "red" if count > 0 else "green"
            console.print(f"  [{style}]{count}[/{style}] {label}")

        if inbox_stale:
            console.print(
                _render_table(
                    inbox_stale,
                    title=f"Inbox {_EM_DASH} Needs Triage (>2d)",
                )
            )
        if stale:
            console.print(_render_table(stale, title=f"Stale {_EM_DASH} No Updates (>7d)"))
        if overdue:
            console.print(_render_table(overdue, title="Overdue"))
        if waiting:
            console.print(_render_table(waiting, title="Blocked / Waiting"))

    @tasks.command("export")
    def export_tasks():
        """Export open tasks to task_context.md for summarizer integration."""
        path = _export_task_context()
        open_count = len([t for t in load_tasks() if t.status not in ("done", "dropped")])
        console.print(f"[green]Exported[/green] {open_count} open tasks to [cyan]{path}[/cyan]")
