"""Record system audio from meetings via PipeWire + FFmpeg."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()

RECORDINGS_DIR = Path.home() / "Recordings" / "meetings"
PID_FILE = Path.home() / ".cache" / "recorder" / "recorder.pid"


def get_pipewire_sinks() -> list[dict]:
    """List available PipeWire audio sinks (outputs) that can be monitored."""
    result = subprocess.run(
        ["wpctl", "status"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print("[red]Error: wpctl not available. Is PipeWire running?[/red]")
        sys.exit(1)

    sinks: list[dict] = []
    in_sinks = False
    for line in result.stdout.splitlines():
        # Strip tree-drawing chars: │ ├ └ ─
        cleaned_line = line.replace("│", " ").replace("├", " ").replace("└", " ").replace("─", " ")
        stripped = cleaned_line.strip()

        if stripped == "Sinks:":
            in_sinks = True
            continue
        if in_sinks:
            if not stripped or stripped.endswith(":"):
                # Empty line or next section header
                in_sinks = False
                continue
            # Parse lines like: "*   91. BRAVIA Theatre U  [vol: 0.97]"
            # or:                "    58. Meteor Lake-P ... [vol: 1.03]"
            is_default = "*" in stripped
            cleaned = stripped.lstrip("*").strip()
            if "." in cleaned:
                dot_idx = cleaned.index(".")
                try:
                    sink_id = int(cleaned[:dot_idx].strip())
                except ValueError:
                    continue
                rest = cleaned[dot_idx + 1 :].strip()
                # Extract name (everything before the last [...])
                bracket_idx = rest.rfind("[")
                name = rest[:bracket_idx].strip() if bracket_idx != -1 else rest
                sinks.append({"id": sink_id, "name": name, "default": is_default})

    return sinks


def get_default_sink_id() -> int | None:
    """Get the ID of the default PipeWire sink."""
    sinks = get_pipewire_sinks()
    for sink in sinks:
        if sink["default"]:
            return sink["id"]
    return sinks[0]["id"] if sinks else None


def get_monitor_node_name(sink_id: int) -> str | None:
    """Get the PipeWire node name for a sink's monitor source."""
    result = subprocess.run(
        ["wpctl", "inspect", str(sink_id)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        cleaned = line.strip().lstrip("*").strip()
        if cleaned.startswith("node.name"):
            # node.name = "alsa_output.pci-..."
            parts = cleaned.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip('"')
    return None


def build_ffmpeg_cmd(
    output_path: Path,
    monitor_source: str,
    audio_format: str = "flac",
) -> list[str]:
    """Build the FFmpeg command for recording audio from a PipeWire monitor source."""
    # Use PipeWire's PulseAudio compatibility layer
    # The ".monitor" suffix captures audio playing through that sink
    source = f"{monitor_source}.monitor"

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "pulse",
        "-i",
        source,
    ]

    if audio_format == "flac":
        cmd += ["-c:a", "flac"]
    elif audio_format == "mp3":
        cmd += ["-c:a", "libmp3lame", "-q:a", "2"]
    elif audio_format == "wav":
        cmd += ["-c:a", "pcm_s16le"]
    elif audio_format == "ogg":
        cmd += ["-c:a", "libvorbis", "-q:a", "6"]

    cmd.append(str(output_path))
    return cmd


@click.group()
def cli():
    """Record system audio from meetings (Zoom, Google Meet, etc.)."""
    pass


@cli.command()
def devices():
    """List available audio output devices."""
    sinks = get_pipewire_sinks()
    if not sinks:
        console.print("[yellow]No audio sinks found.[/yellow]")
        return

    table = Table(title="Audio Output Devices (Sinks)")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Name", style="white")
    table.add_column("Default", style="green")

    for sink in sinks:
        table.add_row(
            str(sink["id"]),
            sink["name"],
            "✓" if sink["default"] else "",
        )

    console.print(table)
    console.print(
        "\n[dim]The recorder captures audio from the default sink's monitor.\n"
        "Change default sink with: wpctl set-default <ID>[/dim]"
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
def start(
    output: str | None,
    audio_format: str,
    device: int | None,
    tag: str | None,
):
    """Start recording system audio."""
    # Resolve which sink to record
    sink_id = device or get_default_sink_id()
    if sink_id is None:
        console.print("[red]No audio sinks found. Is PipeWire running?[/red]")
        sys.exit(1)

    monitor_name = get_monitor_node_name(sink_id)
    if monitor_name is None:
        console.print(f"[red]Could not resolve node name for sink {sink_id}.[/red]")
        sys.exit(1)

    # Build output path
    if output:
        output_path = Path(output)
    else:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tag_part = f"_{tag}" if tag else ""
        output_path = RECORDINGS_DIR / f"meeting{tag_part}_{timestamp}.{audio_format}"

    # Build and run ffmpeg
    cmd = build_ffmpeg_cmd(output_path, monitor_name, audio_format)

    console.print(f"[bold green]Recording started[/bold green]")
    console.print(f"  Source: [cyan]{monitor_name}.monitor[/cyan]")
    console.print(f"  Output: [cyan]{output_path}[/cyan]")
    console.print(f"  Format: [cyan]{audio_format}[/cyan]")
    console.print(f"\n[yellow]Press Ctrl+C to stop recording.[/yellow]\n")

    # Save metadata alongside the recording
    meta = {
        "source": f"{monitor_name}.monitor",
        "sink_id": sink_id,
        "format": audio_format,
        "started_at": datetime.now().isoformat(),
        "output": str(output_path),
    }

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Write PID file so toggle/status can find us
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))

    def handle_stop(signum, frame):
        proc.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    _, stderr = proc.communicate()

    PID_FILE.unlink(missing_ok=True)
    meta["stopped_at"] = datetime.now().isoformat()

    if output_path.exists():
        meta["file_size_mb"] = round(output_path.stat().st_size / (1024 * 1024), 2)
        # Save metadata
        meta_path = output_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2))

        console.print(f"\n[bold green]Recording saved:[/bold green] {output_path}")
        console.print(f"[dim]Metadata: {meta_path}[/dim]")
    else:
        console.print(f"\n[red]Recording failed.[/red]")
        if stderr:
            console.print(f"[dim]{stderr.decode()[-500:]}[/dim]")


@cli.command(name="list")
def list_recordings():
    """List saved recordings."""
    if not RECORDINGS_DIR.exists():
        console.print("[yellow]No recordings directory found.[/yellow]")
        return

    recordings = sorted(RECORDINGS_DIR.glob("meeting*.*"), reverse=True)
    # Filter out metadata files
    recordings = [r for r in recordings if r.suffix != ".json"]

    if not recordings:
        console.print("[yellow]No recordings found.[/yellow]")
        return

    table = Table(title=f"Recordings in {RECORDINGS_DIR}")
    table.add_column("File", style="cyan")
    table.add_column("Size", style="white", justify="right")
    table.add_column("Date", style="green")
    table.add_column("Duration", style="yellow")

    for rec in recordings[:20]:
        size_mb = rec.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(rec.stat().st_mtime)

        # Try to get duration from metadata
        duration = "—"
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
                    if hours:
                        duration = f"{hours}h {mins}m {secs}s"
                    else:
                        duration = f"{mins}m {secs}s"
            except (json.JSONDecodeError, KeyError):
                pass

        table.add_row(
            rec.name,
            f"{size_mb:.1f} MB",
            mtime.strftime("%Y-%m-%d %H:%M"),
            duration,
        )

    console.print(table)


def _notify(title: str, body: str, urgency: str = "normal"):
    """Send a desktop notification via notify-send."""
    subprocess.run(
        ["notify-send", f"--urgency={urgency}", "--app-name=Recorder", title, body],
        capture_output=True,
    )


def _is_recording() -> int | None:
    """Check if a recording is in progress. Returns PID if running, else None."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # check if process exists
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return None


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
def toggle(audio_format: str, tag: str | None):
    """Toggle recording on/off. Designed for keyboard shortcuts."""
    pid = _is_recording()

    if pid is not None:
        # Stop the running recording
        try:
            os.kill(pid, signal.SIGTERM)
            _notify("Recording stopped", "Meeting recording saved.")
            console.print(f"[yellow]Stopped recording (PID {pid}).[/yellow]")
        except ProcessLookupError:
            PID_FILE.unlink(missing_ok=True)
            console.print("[yellow]Recording was already stopped.[/yellow]")
        return

    # Start a new recording in the background
    sink_id = get_default_sink_id()
    if sink_id is None:
        _notify("Recording failed", "No audio sinks found.", urgency="critical")
        sys.exit(1)

    monitor_name = get_monitor_node_name(sink_id)
    if monitor_name is None:
        _notify("Recording failed", f"Could not resolve sink {sink_id}.", urgency="critical")
        sys.exit(1)

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag_part = f"_{tag}" if tag else ""
    output_path = RECORDINGS_DIR / f"meeting{tag_part}_{timestamp}.{audio_format}"

    cmd = build_ffmpeg_cmd(output_path, monitor_name, audio_format)

    # Save metadata
    meta = {
        "source": f"{monitor_name}.monitor",
        "sink_id": sink_id,
        "format": audio_format,
        "started_at": datetime.now().isoformat(),
        "output": str(output_path),
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))

    # Launch ffmpeg as a detached background process
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    PID_FILE.write_text(str(proc.pid))

    _notify("Recording started", f"Saving to {output_path.name}")
    console.print(f"[bold green]Recording started[/bold green] (PID {proc.pid})")
    console.print(f"  Output: [cyan]{output_path}[/cyan]")


@cli.command()
def status():
    """Check if a recording is currently in progress."""
    pid = _is_recording()
    if pid:
        # Read metadata to show details
        meta_files = sorted(RECORDINGS_DIR.glob("meeting*.json"), reverse=True)
        if meta_files:
            meta = json.loads(meta_files[0].read_text())
            started = datetime.fromisoformat(meta["started_at"])
            elapsed = datetime.now() - started
            mins, secs = divmod(int(elapsed.total_seconds()), 60)
            console.print(f"[bold green]Recording in progress[/bold green] (PID {pid})")
            console.print(f"  File: [cyan]{meta.get('output', 'unknown')}[/cyan]")
            console.print(f"  Elapsed: [yellow]{mins}m {secs}s[/yellow]")
        else:
            console.print(f"[bold green]Recording in progress[/bold green] (PID {pid})")
    else:
        console.print("[dim]Not recording.[/dim]")


if __name__ == "__main__":
    cli()
