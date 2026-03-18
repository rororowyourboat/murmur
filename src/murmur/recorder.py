"""PipeWire discovery and FFmpeg recording logic."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

from murmur import hooks
from murmur.config import get_section

console = Console()

PID_FILE = Path.home() / ".cache" / "murmur" / "murmur.pid"


def _default_output_dir() -> Path:
    cfg = get_section("recording")
    raw = cfg.get("output_dir", "~/Recordings/meetings")
    return Path(raw).expanduser()


def _default_format() -> str:
    return get_section("recording").get("format", "flac")


def _parse_wpctl_section(output: str, section_name: str) -> list[dict]:
    """Parse a section (Sinks or Sources) from wpctl status output."""
    items: list[dict] = []
    in_section = False
    for line in output.splitlines():
        cleaned_line = (
            line.replace("\u2502", " ")
            .replace("\u251c", " ")
            .replace("\u2514", " ")
            .replace("\u2500", " ")
        )
        stripped = cleaned_line.strip()

        if stripped == f"{section_name}:":
            in_section = True
            continue
        if in_section:
            if not stripped or stripped.endswith(":"):
                in_section = False
                continue
            is_default = "*" in stripped
            cleaned = stripped.lstrip("*").strip()
            if "." in cleaned:
                dot_idx = cleaned.index(".")
                try:
                    item_id = int(cleaned[:dot_idx].strip())
                except ValueError:
                    continue
                rest = cleaned[dot_idx + 1 :].strip()
                bracket_idx = rest.rfind("[")
                name = rest[:bracket_idx].strip() if bracket_idx != -1 else rest
                items.append({"id": item_id, "name": name, "default": is_default})

    return items


def _get_wpctl_status() -> str:
    """Run wpctl status and return stdout. Exits on failure."""
    result = subprocess.run(["wpctl", "status"], capture_output=True, text=True)
    if result.returncode != 0:
        console.print("[red]Error: wpctl not available. Is PipeWire running?[/red]")
        sys.exit(1)
    return result.stdout


def get_pipewire_sinks() -> list[dict]:
    """List available PipeWire audio sinks (outputs) that can be monitored."""
    return _parse_wpctl_section(_get_wpctl_status(), "Sinks")


def get_pipewire_sources() -> list[dict]:
    """List available PipeWire audio sources (inputs/mics)."""
    return _parse_wpctl_section(_get_wpctl_status(), "Sources")


def get_default_sink_id() -> int | None:
    """Get the ID of the default PipeWire sink."""
    sinks = get_pipewire_sinks()
    for sink in sinks:
        if sink["default"]:
            return sink["id"]
    return sinks[0]["id"] if sinks else None


def get_default_source_id() -> int | None:
    """Get the ID of the default PipeWire source (mic)."""
    sources = get_pipewire_sources()
    for source in sources:
        if source["default"]:
            return source["id"]
    return sources[0]["id"] if sources else None


def get_node_name(node_id: int) -> str | None:
    """Get the PipeWire node name for any node ID (sink or source)."""
    result = subprocess.run(
        ["wpctl", "inspect", str(node_id)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        cleaned = line.strip().lstrip("*").strip()
        if cleaned.startswith("node.name"):
            parts = cleaned.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip('"')
    return None


# Keep old name as alias for backward compat in tests
get_monitor_node_name = get_node_name


_CODEC_MAP: dict[str, list[str]] = {
    "flac": ["-c:a", "flac"],
    "mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
    "wav": ["-c:a", "pcm_s16le"],
    "ogg": ["-c:a", "libvorbis", "-q:a", "6"],
}


def _codec_args(audio_format: str) -> list[str]:
    """Return FFmpeg codec arguments for a given format."""
    return list(_CODEC_MAP.get(audio_format, []))


def build_ffmpeg_cmd(
    output_path: Path,
    monitor_source: str,
    audio_format: str = "flac",
    mic_source: str | None = None,
) -> list[str]:
    """Build FFmpeg command for recording.

    If mic_source is provided, mixes system audio (left channel) and mic
    (right channel) into a stereo file. This keeps compatibility with all
    audio formats and lets diarization split channels later.
    """
    system_source = f"{monitor_source}.monitor"

    cmd = ["ffmpeg", "-y"]

    # Input 0: system audio
    cmd += ["-f", "pulse", "-i", system_source]

    if mic_source:
        # Input 1: microphone
        cmd += ["-f", "pulse", "-i", mic_source]
        # Mix into stereo: system=left, mic=right
        cmd += [
            "-filter_complex",
            "[0:a][1:a]amerge=inputs=2[out]",
            "-map",
            "[out]",
            "-ac",
            "2",
        ]

    cmd += _codec_args(audio_format)
    cmd.append(str(output_path))
    return cmd


def notify(title: str, body: str, urgency: str = "normal"):
    """Send a desktop notification via notify-send."""
    subprocess.run(
        ["notify-send", f"--urgency={urgency}", "--app-name=Murmur", title, body],
        capture_output=True,
    )


def is_recording() -> int | None:
    """Check if a recording is in progress. Returns PID if running, else None."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except ValueError, ProcessLookupError, PermissionError:
        PID_FILE.unlink(missing_ok=True)
        return None


def resolve_sink(device: int | None) -> tuple[int, str]:
    """Resolve sink ID and monitor node name. Exits on failure."""
    sink_id = device or get_default_sink_id()
    if sink_id is None:
        console.print("[red]No audio sinks found. Is PipeWire running?[/red]")
        sys.exit(1)

    monitor_name = get_node_name(sink_id)
    if monitor_name is None:
        console.print(f"[red]Could not resolve node name for sink {sink_id}.[/red]")
        sys.exit(1)

    return sink_id, monitor_name


def resolve_source(device: int | None) -> tuple[int, str]:
    """Resolve source (mic) ID and node name. Exits on failure."""
    source_id = device or get_default_source_id()
    if source_id is None:
        console.print("[red]No audio sources (mics) found.[/red]")
        sys.exit(1)

    source_name = get_node_name(source_id)
    if source_name is None:
        console.print(f"[red]Could not resolve node name for source {source_id}.[/red]")
        sys.exit(1)

    return source_id, source_name


def make_output_path(
    output: str | None,
    audio_format: str,
    tag: str | None,
) -> Path:
    """Build the output file path from options or generate one."""
    if output:
        return Path(output)

    recordings_dir = _default_output_dir()
    recordings_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag_part = f"_{tag}" if tag else ""
    return recordings_dir / f"meeting{tag_part}_{timestamp}.{audio_format}"


def record_foreground(
    output_path: Path,
    monitor_name: str,
    sink_id: int,
    audio_format: str,
    mic_source: str | None = None,
    mic_id: int | None = None,
) -> None:
    """Run FFmpeg in the foreground, blocking until stopped."""
    cmd = build_ffmpeg_cmd(output_path, monitor_name, audio_format, mic_source=mic_source)

    meta = {
        "source": f"{monitor_name}.monitor",
        "sink_id": sink_id,
        "format": audio_format,
        "started_at": datetime.now().isoformat(),
        "output": str(output_path),
        "dual_channel": mic_source is not None,
    }
    if mic_source:
        meta["mic_source"] = mic_source
        meta["mic_id"] = mic_id

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))

    hooks.emit(
        "recording_started",
        pid=proc.pid,
        output_path=str(output_path),
        source=f"{monitor_name}.monitor",
    )

    def handle_stop(signum, frame):
        proc.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    _, stderr = proc.communicate()

    PID_FILE.unlink(missing_ok=True)
    meta["stopped_at"] = datetime.now().isoformat()

    if output_path.exists():
        meta["file_size_mb"] = round(output_path.stat().st_size / (1024 * 1024), 2)
        meta_path = output_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2))

        started = datetime.fromisoformat(meta["started_at"])
        stopped = datetime.fromisoformat(meta["stopped_at"])
        duration_secs = (stopped - started).total_seconds()

        hooks.emit(
            "recording_saved",
            output_path=str(output_path),
            meta_path=str(meta_path),
            duration_secs=duration_secs,
        )

        console.print(f"\n[bold green]Recording saved:[/bold green] {output_path}")
        console.print(f"[dim]Metadata: {meta_path}[/dim]")
    else:
        console.print("\n[red]Recording failed.[/red]")
        if stderr:
            console.print(f"[dim]{stderr.decode()[-500:]}[/dim]")


def record_background(
    output_path: Path,
    monitor_name: str,
    sink_id: int,
    audio_format: str,
    mic_source: str | None = None,
    mic_id: int | None = None,
) -> int:
    """Launch FFmpeg as a detached background process. Returns PID."""
    cmd = build_ffmpeg_cmd(output_path, monitor_name, audio_format, mic_source=mic_source)

    meta = {
        "source": f"{monitor_name}.monitor",
        "sink_id": sink_id,
        "format": audio_format,
        "started_at": datetime.now().isoformat(),
        "output": str(output_path),
        "dual_channel": mic_source is not None,
    }
    if mic_source:
        meta["mic_source"] = mic_source
        meta["mic_id"] = mic_id

    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))

    hooks.emit(
        "recording_started",
        pid=proc.pid,
        output_path=str(output_path),
        source=f"{monitor_name}.monitor",
    )

    return proc.pid
