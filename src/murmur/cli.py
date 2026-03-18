"""Murmur CLI — click.Group subclass with plugin auto-discovery."""

from __future__ import annotations

import json
import os
import signal
from datetime import datetime
from importlib.metadata import entry_points
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from murmur import hooks
from murmur.recorder import (
    get_pipewire_sinks,
    get_pipewire_sources,
    is_recording,
    make_output_path,
    notify,
    record_background,
    record_foreground,
    resolve_sink,
    resolve_source,
)

console = Console()


class MurmurCLI(click.Group):
    """Click group that auto-discovers plugins via entry_points."""

    _plugins_loaded = False

    def _load_plugins(self) -> None:
        if self._plugins_loaded:
            return
        self._plugins_loaded = True

        eps = entry_points(group="murmur.plugins")
        for ep in eps:
            try:
                register = ep.load()
                register(self)
            except Exception as e:
                console.print(f"[yellow]Warning: failed to load plugin '{ep.name}': {e}[/yellow]")

    def list_commands(self, ctx: click.Context) -> list[str]:
        self._load_plugins()
        return super().list_commands(ctx)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        self._load_plugins()
        return super().get_command(ctx, cmd_name)


@click.group(cls=MurmurCLI)
@click.version_option(package_name="murmur")
def cli():
    """Murmur — record system audio from meetings (Zoom, Google Meet, etc.)."""


@cli.command()
def devices():
    """List available audio devices (outputs and inputs)."""
    sinks = get_pipewire_sinks()
    sources = get_pipewire_sources()

    if sinks:
        table = Table(title="Audio Output Devices (Sinks)")
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Name", style="white")
        table.add_column("Default", style="green")

        for sink in sinks:
            table.add_row(
                str(sink["id"]),
                sink["name"],
                "\u2713" if sink["default"] else "",
            )
        console.print(table)
    else:
        console.print("[yellow]No audio sinks found.[/yellow]")

    if sources:
        console.print()
        table = Table(title="Audio Input Devices (Sources)")
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Name", style="white")
        table.add_column("Default", style="green")

        for source in sources:
            table.add_row(
                str(source["id"]),
                source["name"],
                "\u2713" if source["default"] else "",
            )
        console.print(table)
    else:
        console.print("\n[yellow]No audio sources (mics) found.[/yellow]")

    console.print(
        "\n[dim]Change default: wpctl set-default <ID>\nRecord with mic: murmur start --mic[/dim]"
    )


@cli.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path. Auto-generated if not provided.",
)
@click.option(
    "-f",
    "--format",
    "audio_format",
    type=click.Choice(["flac", "mp3", "wav", "ogg"]),
    default="flac",
    help="Audio format (default: flac).",
)
@click.option(
    "-d",
    "--device",
    type=int,
    default=None,
    help="Sink ID to record from (default: current default sink).",
)
@click.option(
    "-t",
    "--tag",
    default=None,
    help="Tag for the recording filename (e.g., 'standup', 'sprint-review').",
)
@click.option(
    "--mic",
    is_flag=True,
    default=False,
    help="Also capture microphone input (dual-channel).",
)
@click.option(
    "--mic-device",
    type=int,
    default=None,
    help="Source ID for mic input (default: default source).",
)
def start(
    output: str | None,
    audio_format: str,
    device: int | None,
    tag: str | None,
    mic: bool,
    mic_device: int | None,
):
    """Start recording system audio."""
    sink_id, monitor_name = resolve_sink(device)
    output_path = make_output_path(output, audio_format, tag)

    mic_source = None
    mic_id = None
    if mic:
        mic_id, mic_source = resolve_source(mic_device)

    console.print("[bold green]Recording started[/bold green]")
    console.print(f"  System: [cyan]{monitor_name}.monitor[/cyan]")
    if mic_source:
        console.print(f"  Mic:    [cyan]{mic_source}[/cyan]")
    console.print(f"  Output: [cyan]{output_path}[/cyan]")
    console.print(f"  Format: [cyan]{audio_format}[/cyan]")
    console.print("\n[yellow]Press Ctrl+C to stop recording.[/yellow]\n")

    record_foreground(
        output_path,
        monitor_name,
        sink_id,
        audio_format,
        mic_source=mic_source,
        mic_id=mic_id,
    )


@cli.command(name="list")
def list_recordings():
    """List saved recordings."""
    from murmur.recorder import _default_output_dir

    recordings_dir = _default_output_dir()
    if not recordings_dir.exists():
        console.print("[yellow]No recordings directory found.[/yellow]")
        return

    recordings = sorted(recordings_dir.glob("meeting*.*"), reverse=True)
    recordings = [r for r in recordings if r.suffix != ".json"]

    if not recordings:
        console.print("[yellow]No recordings found.[/yellow]")
        return

    table = Table(title=f"Recordings in {recordings_dir}")
    table.add_column("File", style="cyan")
    table.add_column("Size", style="white", justify="right")
    table.add_column("Date", style="green")
    table.add_column("Duration", style="yellow")

    for rec in recordings[:20]:
        size_mb = rec.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(rec.stat().st_mtime)

        duration = "\u2014"
        meta_path = rec.with_suffix(".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if "started_at" in meta and "stopped_at" in meta:
                    start_dt = datetime.fromisoformat(meta["started_at"])
                    stop_dt = datetime.fromisoformat(meta["stopped_at"])
                    delta = stop_dt - start_dt
                    mins, secs = divmod(int(delta.total_seconds()), 60)
                    hours, mins = divmod(mins, 60)
                    duration = f"{hours}h {mins}m {secs}s" if hours else f"{mins}m {secs}s"
            except json.JSONDecodeError, KeyError:
                pass

        table.add_row(
            rec.name,
            f"{size_mb:.1f} MB",
            mtime.strftime("%Y-%m-%d %H:%M"),
            duration,
        )

    console.print(table)


@cli.command()
@click.option(
    "-f",
    "--format",
    "audio_format",
    type=click.Choice(["flac", "mp3", "wav", "ogg"]),
    default="flac",
    help="Audio format (default: flac).",
)
@click.option(
    "-t",
    "--tag",
    default=None,
    help="Tag for the recording filename.",
)
@click.option(
    "--mic",
    is_flag=True,
    default=False,
    help="Also capture microphone input (dual-channel).",
)
@click.option(
    "--mic-device",
    type=int,
    default=None,
    help="Source ID for mic input (default: default source).",
)
def toggle(audio_format: str, tag: str | None, mic: bool, mic_device: int | None):
    """Toggle recording on/off. Designed for keyboard shortcuts."""
    pid = is_recording()

    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
            notify("Recording stopped", "Meeting recording saved.")
            console.print(f"[yellow]Stopped recording (PID {pid}).[/yellow]")
        except ProcessLookupError:
            from murmur.recorder import PID_FILE

            PID_FILE.unlink(missing_ok=True)
            console.print("[yellow]Recording was already stopped.[/yellow]")
        return

    sink_id, monitor_name = resolve_sink(None)
    output_path = make_output_path(None, audio_format, tag)

    mic_source = None
    mic_id = None
    if mic:
        mic_id, mic_source = resolve_source(mic_device)

    pid = record_background(
        output_path,
        monitor_name,
        sink_id,
        audio_format,
        mic_source=mic_source,
        mic_id=mic_id,
    )

    notify("Recording started", f"Saving to {output_path.name}")
    console.print(f"[bold green]Recording started[/bold green] (PID {pid})")
    console.print(f"  Output: [cyan]{output_path}[/cyan]")


@cli.command()
def status():
    """Check if a recording is currently in progress."""
    from murmur.recorder import _default_output_dir

    pid = is_recording()
    if pid:
        recordings_dir = _default_output_dir()
        meta_files = sorted(recordings_dir.glob("meeting*.json"), reverse=True)
        if meta_files:
            meta = json.loads(meta_files[0].read_text())
            started = datetime.fromisoformat(meta["started_at"])
            elapsed = datetime.now() - started
            mins, secs = divmod(int(elapsed.total_seconds()), 60)
            console.print(f"[bold green]Recording in progress[/bold green] (PID {pid})")
            console.print(f"  File: [cyan]{meta.get('output', 'unknown')}[/cyan]")
            console.print(f"  Elapsed: [yellow]{mins}m {secs}s[/yellow]")
            if meta.get("dual_channel"):
                console.print(f"  Mic: [cyan]{meta.get('mic_source', 'unknown')}[/cyan]")
        else:
            console.print(f"[bold green]Recording in progress[/bold green] (PID {pid})")
    else:
        console.print("[dim]Not recording.[/dim]")


@cli.command(name="import")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "-t",
    "--tag",
    default=None,
    help="Tag for the imported filename.",
)
def import_audio(file: str, tag: str | None):
    """Import an external audio file into Murmur for processing by plugins."""
    import shutil

    from murmur.recorder import _default_output_dir

    source = Path(file)
    recordings_dir = _default_output_dir()
    recordings_dir.mkdir(parents=True, exist_ok=True)

    if tag:
        dest = recordings_dir / f"meeting_{tag}_{source.stem}{source.suffix}"
    else:
        dest = recordings_dir / source.name

    if dest.exists():
        console.print(f"[yellow]File already exists: {dest}[/yellow]")
        return

    shutil.copy2(source, dest)

    meta = {
        "source": "import",
        "format": source.suffix.lstrip("."),
        "imported_at": datetime.now().isoformat(),
        "original_path": str(source),
        "output": str(dest),
    }
    meta_path = dest.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))

    hooks.emit(
        "recording_saved",
        output_path=str(dest),
        meta_path=str(meta_path),
        duration_secs=0,
    )

    console.print(f"[bold green]Imported:[/bold green] {dest}")
