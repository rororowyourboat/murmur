"""Murmur plugin: Rich Live TUI dashboard."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime

import click
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from murmur.recorder import (
    _default_output_dir,
    is_recording,
    make_output_path,
    notify,
    record_background,
    resolve_sink,
)

console = Console()


def _get_audio_levels() -> str | None:
    """Try to get current audio levels from PipeWire via pw-top (one-shot)."""
    try:
        result = subprocess.run(
            ["pw-top", "-b", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse first line with audio data for a rough level indicator
            for line in result.stdout.splitlines()[1:]:  # skip header
                parts = line.split()
                if len(parts) >= 6 and "running" in line.lower():
                    return parts[5] if parts[5] != "-" else None
    except subprocess.TimeoutExpired, FileNotFoundError:
        pass
    return None


def _level_bar(level_db: float, width: int = 30) -> Text:
    """Render a visual level bar from dB value."""
    # Rough mapping: -60dB = silent, 0dB = max
    normalized = max(0.0, min(1.0, (level_db + 60) / 60))
    filled = int(normalized * width)
    bar = Text()
    if filled < width * 0.7:
        color = "green"
    elif filled < width * 0.9:
        color = "yellow"
    else:
        color = "red"
    bar.append("\u2588" * filled, style=color)
    bar.append("\u2591" * (width - filled), style="dim")
    return bar


def _build_dashboard(recording_meta: dict | None = None) -> Layout:
    """Build the live dashboard layout."""
    layout = Layout()

    pid = is_recording()

    # Status panel
    if pid and recording_meta:
        started = datetime.fromisoformat(recording_meta["started_at"])
        elapsed = datetime.now() - started
        mins, secs = divmod(int(elapsed.total_seconds()), 60)
        hours, mins = divmod(mins, 60)
        elapsed_str = f"{hours}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"

        status_text = Text()
        status_text.append("\u25cf REC ", style="bold red")
        status_text.append(elapsed_str, style="bold white")

        info = Table.grid(padding=(0, 2))
        info.add_column(style="dim", width=10)
        info.add_column(style="white")
        info.add_row("File", recording_meta.get("output", "unknown"))
        info.add_row("Source", recording_meta.get("source", "unknown"))
        info.add_row("Format", recording_meta.get("format", "unknown"))
        if recording_meta.get("dual_channel"):
            info.add_row("Mic", recording_meta.get("mic_source", "unknown"))

        # Audio level
        level_str = _get_audio_levels()
        if level_str:
            with contextlib.suppress(ValueError):
                info.add_row("Level", _level_bar(float(level_str)))

        status_panel = Panel(info, title=status_text, border_style="red")
    else:
        status_panel = Panel(
            "[dim]Not recording. Press [bold]r[/bold] to start.[/dim]",
            title="Murmur",
            border_style="dim",
        )

    # Recent recordings
    recordings_dir = _default_output_dir()
    recent_table = Table(title="Recent Recordings", expand=True)
    recent_table.add_column("File", style="cyan")
    recent_table.add_column("Size", style="white", justify="right")
    recent_table.add_column("Date", style="green")

    if recordings_dir.exists():
        recent = sorted(recordings_dir.glob("meeting*.*"), reverse=True)
        recent = [r for r in recent if r.suffix != ".json"][:8]
        for rec in recent:
            size_mb = rec.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(rec.stat().st_mtime)
            recent_table.add_row(
                rec.name,
                f"{size_mb:.1f} MB",
                mtime.strftime("%Y-%m-%d %H:%M"),
            )

    layout.split_column(
        Layout(status_panel, name="status", size=9),
        Layout(recent_table, name="recent"),
        Layout(
            Panel(
                "[dim]r[/dim] start recording  [dim]q[/dim] quit  [dim]s[/dim] stop recording",
                border_style="dim",
            ),
            name="keys",
            size=3,
        ),
    )

    return layout


def register(cli: click.Group) -> None:
    """Register the tui command."""

    @cli.command()
    @click.option(
        "-f",
        "--format",
        "audio_format",
        type=click.Choice(["flac", "mp3", "wav", "ogg"]),
        default="flac",
        help="Audio format for recordings started from TUI.",
    )
    def tui(audio_format: str):
        """Live dashboard showing recording status and recent files."""
        import select
        import termios
        import tty

        # Get current recording metadata if any
        recording_meta = None
        pid = is_recording()
        if pid:
            recordings_dir = _default_output_dir()
            meta_files = sorted(recordings_dir.glob("meeting*.json"), reverse=True)
            if meta_files:
                recording_meta = json.loads(meta_files[0].read_text())

        # Set up non-blocking key input
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())

            with Live(
                _build_dashboard(recording_meta),
                refresh_per_second=2,
                console=console,
                screen=True,
            ) as live:
                while True:
                    # Check for keypress (non-blocking)
                    if select.select([sys.stdin], [], [], 0.5)[0]:
                        key = sys.stdin.read(1)

                        if key == "q":
                            # Stop recording if active, then quit
                            pid = is_recording()
                            if pid:
                                os.kill(pid, signal.SIGTERM)
                            break

                        elif key == "s":
                            # Stop recording
                            pid = is_recording()
                            if pid:
                                os.kill(pid, signal.SIGTERM)
                                notify("Recording stopped", "Meeting recording saved.")
                                recording_meta = None

                        elif key == "r":
                            # Start recording
                            if not is_recording():
                                sink_id, monitor_name = resolve_sink(None)
                                output_path = make_output_path(None, audio_format, None)
                                record_background(
                                    output_path,
                                    monitor_name,
                                    sink_id,
                                    audio_format,
                                )
                                notify("Recording started", f"Saving to {output_path.name}")
                                # Load the metadata we just wrote
                                meta_path = output_path.with_suffix(".json")
                                if meta_path.exists():
                                    recording_meta = json.loads(meta_path.read_text())

                    # Refresh display
                    # Re-check recording state in case it stopped externally
                    if recording_meta and not is_recording():
                        recording_meta = None
                    live.update(_build_dashboard(recording_meta))

        except KeyboardInterrupt:
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
