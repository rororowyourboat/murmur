"""Murmur plugin: watch for meeting apps using the microphone."""

from __future__ import annotations

import json
import subprocess
import time

import click
from rich.console import Console

from murmur.config import get_section
from murmur.recorder import (
    is_recording,
    make_output_path,
    notify,
    record_background,
    resolve_sink,
)

console = Console()

# App names (from PipeWire application.name) that indicate a meeting
DEFAULT_MEETING_APPS = [
    "chrome",
    "chromium",
    "google-chrome",
    "firefox",
    "zoom",
    "teams",
    "teams-for-linux",
    "microsoft teams",
    "slack",
    "discord",
    "webex",
    "skype",
    "signal",
    "telegram",
]


def _get_mic_streams() -> list[dict]:
    """Get active PipeWire streams that are capturing mic input.

    Returns a list of dicts with 'app', 'name', and 'id' for each
    Stream/Input/Audio node (i.e., apps reading from the mic).
    """
    try:
        result = subprocess.run(
            ["pw-dump"],  # noqa: S603, S607
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
    except subprocess.TimeoutExpired, FileNotFoundError:
        return []

    try:
        nodes = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    streams = []
    for node in nodes:
        if node.get("type") != "PipeWire:Interface:Node":
            continue
        props = node.get("info", {}).get("props", {})
        media_class = props.get("media.class", "")
        if media_class != "Stream/Input/Audio":
            continue
        streams.append(
            {
                "id": node.get("id"),
                "app": (props.get("application.name") or "").lower(),
                "name": props.get("node.name", "unknown"),
            }
        )
    return streams


def _is_meeting_app(stream: dict, app_patterns: list[str]) -> tuple[bool, str]:
    """Check if a stream matches a known meeting app.

    Returns (is_match, display_name).
    """
    app = stream["app"]
    name = stream["name"].lower()

    for pattern in app_patterns:
        pattern = pattern.lower()
        if pattern in app or pattern in name:
            # Friendly display name
            display = app or stream["name"]
            return True, display.title()

    return False, ""


def register(cli: click.Group) -> None:
    """Register the watch command."""

    @cli.command()
    @click.option(
        "--interval",
        default=None,
        type=int,
        help="Poll interval in seconds (default: 5).",
    )
    @click.option(
        "--auto-record",
        is_flag=True,
        default=False,
        help="Automatically start recording when a meeting is detected.",
    )
    @click.option(
        "-f",
        "--format",
        "audio_format",
        type=click.Choice(["flac", "mp3", "wav", "ogg"]),
        default="flac",
        help="Audio format for auto-recordings (default: flac).",
    )
    @click.option(
        "--mic",
        is_flag=True,
        default=False,
        help="Include microphone in auto-recordings (dual-channel).",
    )
    def watch(interval: int | None, auto_record: bool, audio_format: str, mic: bool):
        """Watch for meeting apps using the microphone.

        Polls PipeWire for apps that grab the mic (Zoom, Chrome/Meet,
        Teams, etc.) and sends a desktop notification. With --auto-record,
        also starts recording automatically.
        """
        cfg = get_section("watch")
        poll_interval = interval or cfg.get("interval", 5)
        app_patterns = cfg.get("apps", DEFAULT_MEETING_APPS)

        if auto_record or cfg.get("auto_record"):
            auto_record = True

        console.print("[bold]Watching for meeting apps using the mic...[/bold]")
        console.print(f"  Poll interval: [cyan]{poll_interval}s[/cyan]")
        console.print(f"  Auto-record:   [cyan]{auto_record}[/cyan]")
        console.print(f"  Watching apps:  [dim]{', '.join(app_patterns[:6])}...[/dim]")
        console.print("\n[yellow]Press Ctrl+C to stop watching.[/yellow]\n")

        active_meeting: str | None = None  # display name of detected meeting app

        try:
            while True:
                streams = _get_mic_streams()
                meeting_streams = []
                for s in streams:
                    is_match, display = _is_meeting_app(s, app_patterns)
                    if is_match:
                        meeting_streams.append(display)

                if meeting_streams and active_meeting is None:
                    # Meeting just started
                    app_name = meeting_streams[0]
                    active_meeting = app_name
                    console.print(
                        f"[bold green]Meeting detected:[/bold green] {app_name} is using the mic"
                    )
                    notify(
                        "Meeting detected",
                        f"{app_name} is using your microphone.",
                    )

                    if auto_record and not is_recording():
                        sink_id, monitor_name = resolve_sink(None)
                        tag = app_name.lower().replace(" ", "-")
                        output_path = make_output_path(None, audio_format, tag)

                        mic_source = None
                        mic_id = None
                        if mic:
                            from murmur.recorder import resolve_source

                            mic_id, mic_source = resolve_source(None)

                        record_background(
                            output_path,
                            monitor_name,
                            sink_id,
                            audio_format,
                            mic_source=mic_source,
                            mic_id=mic_id,
                        )
                        notify(
                            "Auto-recording started",
                            f"Recording {app_name} meeting to {output_path.name}",
                        )
                        console.print(
                            f"[bold green]Auto-recording started:[/bold green] {output_path.name}"
                        )

                elif not meeting_streams and active_meeting is not None:
                    # Meeting ended
                    console.print(
                        f"[bold yellow]Meeting ended:[/bold yellow] {active_meeting} "
                        "released the mic"
                    )
                    notify(
                        "Meeting ended",
                        f"{active_meeting} released your microphone.",
                    )

                    if auto_record and is_recording():
                        import os
                        import signal

                        pid = is_recording()
                        if pid:
                            os.kill(pid, signal.SIGTERM)
                            notify("Auto-recording stopped", "Meeting recording saved.")
                            console.print("[bold yellow]Auto-recording stopped.[/bold yellow]")

                    active_meeting = None

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            console.print("\n[dim]Stopped watching.[/dim]")
