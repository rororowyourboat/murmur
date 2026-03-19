"""Murmur plugin: persistent memory for LLM context."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown

console = Console()

MEMORY_PATH = Path.home() / ".config" / "murmur" / "memory.md"

DEFAULT_MEMORY = """\
# About me
<!-- Your name, role, and what you work on -->

# Team
<!-- Key people you meet with regularly — name and role -->

# Projects
<!-- Current projects, codenames, or workstreams for context -->

# Summary preferences
<!-- How you like meeting summaries formatted -->
- Highlight action items assigned to me
- Use bullet points, keep summaries concise
- Note key decisions and who made them
- Flag deadlines and dates explicitly
- If I didn't speak much, focus on what's relevant to my work
"""


def load_memory() -> str | None:
    """Load memory content. Returns None if no memory file exists."""
    if not MEMORY_PATH.exists():
        return None
    text = MEMORY_PATH.read_text().strip()
    return text if text else None


def register(cli: click.Group) -> None:
    """Register the memory command group."""

    @cli.group(invoke_without_command=True)
    @click.pass_context
    def memory(ctx):
        """View and manage your personal context for LLM summaries."""
        if ctx.invoked_subcommand is not None:
            return

        # Default: show current memory
        content = load_memory()
        if content:
            console.print(Markdown(content))
        else:
            console.print("[dim]No memory set up yet.[/dim]")
            console.print("Run [cyan]murmur memory edit[/cyan] to create one.")

    @memory.command()
    def edit():
        """Open memory file in your editor."""
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not MEMORY_PATH.exists():
            MEMORY_PATH.write_text(DEFAULT_MEMORY)
            console.print(f"[green]Created[/green] {MEMORY_PATH} with template.")

        editor = os.environ.get("EDITOR", "nano")
        subprocess.run([editor, str(MEMORY_PATH)])  # noqa: S603, S607

    @memory.command()
    def path():
        """Print the memory file path."""
        console.print(str(MEMORY_PATH))

    @memory.command()
    def show():
        """Display the current memory contents."""
        content = load_memory()
        if content:
            console.print(Markdown(content))
        else:
            console.print("[dim]No memory set up yet.[/dim]")

    @memory.command()
    def reset():
        """Reset memory to the default template."""
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_PATH.write_text(DEFAULT_MEMORY)
        console.print(f"[green]Reset[/green] {MEMORY_PATH} to default template.")
