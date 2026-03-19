"""Murmur plugin: task management with todo.txt backend."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console
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
# Storage operations
# ---------------------------------------------------------------------------


def load_tasks() -> list[Task]:
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


def save_tasks(tasks: list[Task]) -> None:
    """Write all tasks to the todo.txt file."""
    path = _tasks_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [task_to_line(t) for t in tasks]
    path.write_text("\n".join(lines) + "\n" if lines else "")


def find_task(task_id: str, tasks: list[Task] | None = None) -> Task | None:
    """Find a task by its ID (or prefix)."""
    if tasks is None:
        tasks = load_tasks()
    for t in tasks:
        if t.id == task_id or t.id.startswith(task_id):
            return t
    return None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


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
    console.print(f"  Owner:    {task.owner or '—'}")
    console.print(f"  Project:  {task.project or '—'}")
    console.print(f"  Deadline: {task.deadline or '—'}")
    console.print(f"  Created:  {task.created_at}")
    if task.source_file:
        console.print(f"  Source:   {task.source_file}")
    if task.tags:
        console.print(f"  Tags:     {', '.join(task.tags)}")


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


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
        all_tasks = load_tasks()
        all_tasks.append(task)
        save_tasks(all_tasks)
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
        all_tasks = load_tasks()
        task = find_task(task_id, all_tasks)
        if not task:
            console.print(f"[red]Task '{task_id}' not found.[/red]")
            raise SystemExit(1)
        task.status = "done"
        save_tasks(all_tasks)
        console.print(f"[green]Done:[/green] [dim strikethrough]{task.title}[/]")

    @tasks.command()
    @click.argument("task_id")
    def drop(task_id):
        """Mark a task as dropped."""
        all_tasks = load_tasks()
        task = find_task(task_id, all_tasks)
        if not task:
            console.print(f"[red]Task '{task_id}' not found.[/red]")
            raise SystemExit(1)
        task.status = "dropped"
        save_tasks(all_tasks)
        console.print(f"[dim red]Dropped:[/dim red] {task.title}")

    @tasks.command()
    @click.argument("task_id")
    @click.argument("new_status", type=click.Choice(VALID_STATUSES, case_sensitive=False))
    def move(task_id, new_status):
        """Change a task's status."""
        all_tasks = load_tasks()
        task = find_task(task_id, all_tasks)
        if not task:
            console.print(f"[red]Task '{task_id}' not found.[/red]")
            raise SystemExit(1)
        old = task.status
        task.status = new_status.lower()
        save_tasks(all_tasks)
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
        all_tasks = load_tasks()
        task = find_task(task_id, all_tasks)
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

        save_tasks(all_tasks)
        console.print(f"[green]Updated[/green] task [bold]{task.id}[/bold]")
        _show_detail(task)
