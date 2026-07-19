"""PipeWire discovery and FFmpeg recording logic."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from murmur import hooks
from murmur.config import get_section

console = Console()

PID_FILE = Path.home() / ".cache" / "murmur" / "murmur.pid"

MULTITRACK_FORMAT = "mka"
MULTITRACK_STREAMS = (
    {
        "index": 0,
        "title": "Mixed call",
        "codec": "opus",
        "source_role": "mixed",
        "default": True,
    },
    {
        "index": 1,
        "title": "Microphone",
        "codec": "opus",
        "source_role": "microphone",
        "default": False,
    },
    {
        "index": 2,
        "title": "Call output",
        "codec": "opus",
        "source_role": "call_output",
        "default": False,
    },
)


def _state_file() -> Path:
    return PID_FILE.with_name("recording.json")


def _lock_file() -> Path:
    return PID_FILE.with_name("control.lock")


@contextmanager
def _control_lock():
    """Serialize recording state transitions across CLI/plugin processes."""
    lock_path = _lock_file()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _read_active_state() -> dict[str, Any] | None:
    try:
        state = json.loads(_state_file().read_text())
    except FileNotFoundError, json.JSONDecodeError, OSError:
        return None
    return state if isinstance(state, dict) else None


def _clear_active_state() -> None:
    PID_FILE.unlink(missing_ok=True)
    _state_file().unlink(missing_ok=True)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError, PermissionError:
        return False

    # kill(pid, 0) succeeds for zombies, but a zombie has already finished
    # writing its output and should not block finalization.
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        if stat.rsplit(")", 1)[1].strip().startswith("Z"):
            return False
    except FileNotFoundError, IndexError, OSError:
        pass
    return True


def _process_matches(pid: int, output_path: str | None = None) -> bool:
    """Return whether pid is the FFmpeg process recorded in active state."""
    if not _pid_exists(pid):
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError, PermissionError, OSError:
        return False
    if not command or Path(os.fsdecode(command[0])).name != "ffmpeg":
        return False
    if output_path is None:
        return True
    encoded_output = os.fsencode(output_path)
    return encoded_output in command


def _recording_metadata(
    output_path: Path,
    monitor_name: str,
    sink_id: int,
    audio_format: str,
    mic_source: str | None,
    mic_id: int | None,
) -> dict[str, Any]:
    multitrack = mic_source is not None
    effective_format = MULTITRACK_FORMAT if multitrack else audio_format
    meta: dict[str, Any] = {
        "status": "recording",
        "source": f"{monitor_name}.monitor",
        "sink_id": sink_id,
        "format": effective_format,
        "started_at": datetime.now().isoformat(),
        "output": str(output_path),
        "dual_channel": multitrack,
        "capture_mode": "multitrack" if multitrack else "system_output",
    }
    if mic_source:
        meta["requested_format"] = audio_format
        meta["mic_source"] = mic_source
        meta["mic_id"] = mic_id
        source_names = {
            "mixed": [f"{monitor_name}.monitor", mic_source],
            "microphone": [mic_source],
            "call_output": [f"{monitor_name}.monitor"],
        }
        meta["stream_layout"] = [
            {**stream, "source_names": source_names[stream["source_role"]]}
            for stream in MULTITRACK_STREAMS
        ]
    return meta


def _write_active_state(pid: int, meta: dict[str, Any], log_path: Path) -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(f"{pid}\n")
    _write_json_atomic(
        _state_file(),
        {
            "pid": pid,
            "output": meta["output"],
            "meta_path": str(Path(meta["output"]).with_suffix(".json")),
            "log_path": str(log_path),
            "started_at": meta["started_at"],
        },
    )


def _metadata_from_state(state: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(state["output"])
    meta_path = Path(state.get("meta_path", output_path.with_suffix(".json")))
    try:
        meta = json.loads(meta_path.read_text())
    except FileNotFoundError, json.JSONDecodeError, OSError:
        meta = {
            "status": "recording",
            "started_at": state.get("started_at", datetime.now().isoformat()),
            "output": str(output_path),
        }
    if isinstance(meta, dict):
        return meta
    return {"status": "recording", "output": str(output_path)}


def _log_error(state: dict[str, Any]) -> str | None:
    log_path = state.get("log_path")
    if not log_path:
        return None
    try:
        content = Path(log_path).read_text(errors="replace").strip()
    except OSError:
        return None
    return content[-500:] or None


def _mark_start_failed(meta: dict[str, Any], error: OSError) -> None:
    failed = dict(meta)
    failed.update(
        {
            "status": "failed",
            "stopped_at": datetime.now().isoformat(),
            "error": f"Could not start FFmpeg: {error}",
        }
    )
    _write_json_atomic(Path(meta["output"]).with_suffix(".json"), failed)


def _probe_media(output_path: Path) -> dict[str, Any] | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    result = subprocess.run(  # noqa: S603
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=index,codec_name,codec_type,channels:"
            "stream_tags=title:stream_disposition=default",
            "-of",
            "json",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _finalize_recording(meta: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    output_path = Path(meta["output"])
    meta_path = output_path.with_suffix(".json")
    finalized = dict(meta)
    finalized["stopped_at"] = datetime.now().isoformat()

    probe = _probe_media(output_path) if output_path.is_file() else None
    if probe is not None and output_path.stat().st_size > 0:
        format_info = probe.get("format", {})
        streams = probe.get("streams", [])
        planned_streams = {stream["index"]: stream for stream in meta.get("stream_layout", [])}
        enriched_streams = []
        for stream in streams:
            planned = planned_streams.get(stream.get("index"), {})
            probed_title = stream.get("tags", {}).get("title")
            if not planned and not probed_title:
                enriched_streams.append(stream)
                continue
            enriched_streams.append(
                {
                    **stream,
                    "title": probed_title or planned.get("title"),
                    "source_role": planned.get("source_role"),
                    "source_names": planned.get("source_names", []),
                    "default": bool(
                        stream.get("disposition", {}).get("default", planned.get("default", False))
                    ),
                }
            )
        finalized.update(
            {
                "status": "recorded",
                "duration_secs": float(format_info.get("duration", 0.0)),
                "file_size_bytes": int(format_info.get("size", output_path.stat().st_size)),
                "file_size_mb": round(output_path.stat().st_size / (1024 * 1024), 2),
                "streams": enriched_streams,
            }
        )
        finalized.pop("error", None)
    else:
        finalized["status"] = "failed"
        finalized["error"] = error or "FFmpeg did not produce a valid media file."

    if finalized["status"] == "recorded":
        from murmur.artifacts import ArtifactStore

        store = ArtifactStore(output_path)
        finalized["recording_id"] = store.recording_id
        finalized["artifacts_dir"] = str(store.directory)
    _write_json_atomic(meta_path, finalized)
    if finalized["status"] == "recorded":
        store.ensure_manifest(finalized)
        hooks.emit(
            "recording_saved",
            output_path=str(output_path),
            meta_path=str(meta_path),
            duration_secs=finalized["duration_secs"],
        )
    return finalized


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
    result = subprocess.run(["wpctl", "status"], capture_output=True, text=True)  # noqa: S607
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
    result = subprocess.run(  # noqa: S603
        ["wpctl", "inspect", str(node_id)],  # noqa: S607
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

    If mic_source is provided, produce a Matroska file with a default listening
    mix plus independently selectable microphone and call-output streams.
    """
    system_source = f"{monitor_source}.monitor"

    cmd = ["ffmpeg", "-y", "-thread_queue_size", "1024"]

    # Input 0: system audio
    cmd += ["-f", "pulse", "-i", system_source]

    if mic_source:
        if output_path.suffix.lower() != ".mka":
            raise ValueError("Microphone capture requires an .mka output path.")
        cmd += ["-thread_queue_size", "1024", "-f", "pulse", "-i", mic_source]
        cmd += [
            "-filter_complex",
            "[0:a]aresample=async=1000:first_pts=0,asplit=2[call_mix][call_track];"
            "[1:a]aresample=async=1000:first_pts=0,asplit=2[mic_mix][mic_track];"
            "[call_mix][mic_mix]amix=inputs=2:duration=longest:dropout_transition=0:"
            "normalize=0,alimiter=limit=0.95:level=false[mix]",
            "-map",
            "[mix]",
            "-map",
            "[mic_track]",
            "-map",
            "[call_track]",
            "-c:a",
            "libopus",
            "-b:a:0",
            "128k",
            "-b:a:1",
            "96k",
            "-b:a:2",
            "96k",
            "-metadata:s:a:0",
            "title=Mixed call",
            "-metadata:s:a:1",
            "title=Microphone",
            "-metadata:s:a:2",
            "title=Call output",
            "-disposition:a:0",
            "default",
            "-disposition:a:1",
            "0",
            "-disposition:a:2",
            "0",
            "-f",
            "matroska",
        ]
    else:
        cmd += _codec_args(audio_format)
    cmd.append(str(output_path))
    return cmd


def notify(title: str, body: str, urgency: str = "normal"):
    """Send a desktop notification via notify-send."""
    subprocess.run(  # noqa: S603
        ["notify-send", f"--urgency={urgency}", "--app-name=Murmur", title, body],  # noqa: S607
        capture_output=True,
    )


def is_recording() -> int | None:
    """Check if a recording is in progress. Returns PID if running, else None."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except ValueError, OSError:
        _clear_active_state()
        return None

    state = _read_active_state()
    output_path = str(state.get("output")) if state and state.get("output") else None
    if _process_matches(pid, output_path):
        return pid

    if state and state.get("pid") == pid and state.get("output"):
        try:
            _finalize_recording(_metadata_from_state(state), error=_log_error(state))
        finally:
            _clear_active_state()
    else:
        _clear_active_state()
    return None


def stop_recording(timeout: float = 5.0) -> dict[str, Any]:
    """Gracefully stop and finalize the active background recording."""
    with _control_lock():
        state = _read_active_state()
        if state is None:
            raise RuntimeError("No recording is currently in progress.")

        try:
            pid = int(state["pid"])
        except (KeyError, TypeError, ValueError) as error:
            _clear_active_state()
            raise RuntimeError("Active recording state has no valid PID.") from error

        meta = _metadata_from_state(state)
        meta_path = Path(meta["output"]).with_suffix(".json")
        if not _process_matches(pid, str(state["output"])):
            try:
                return _finalize_recording(meta, error=_log_error(state))
            finally:
                _clear_active_state()

        os.kill(pid, signal.SIGINT)
        deadline = time.monotonic() + timeout
        while _pid_exists(pid) and time.monotonic() < deadline:
            time.sleep(0.1)

        if _pid_exists(pid):
            os.kill(pid, signal.SIGTERM)
            fallback_deadline = time.monotonic() + 1.0
            while _pid_exists(pid) and time.monotonic() < fallback_deadline:
                time.sleep(0.1)

        error = None
        if _pid_exists(pid):
            error = f"FFmpeg process {pid} did not exit after SIGINT and SIGTERM."
            failed = dict(meta)
            failed.update(
                {
                    "status": "failed",
                    "stopped_at": datetime.now().isoformat(),
                    "error": error,
                }
            )
            _write_json_atomic(meta_path, failed)
            raise RuntimeError(error)

        try:
            return _finalize_recording(meta)
        finally:
            _clear_active_state()


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
    multitrack: bool = False,
) -> Path:
    """Build the output file path from options or generate one."""
    if output:
        path = Path(output)
        return path.with_suffix(".mka") if multitrack else path

    recordings_dir = _default_output_dir()
    recordings_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag_part = f"_{tag}" if tag else ""
    suffix = MULTITRACK_FORMAT if multitrack else audio_format
    return recordings_dir / f"meeting{tag_part}_{timestamp}.{suffix}"


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
    meta = _recording_metadata(
        output_path, monitor_name, sink_id, audio_format, mic_source, mic_id
    )
    meta_path = output_path.with_suffix(".json")
    log_path = output_path.with_suffix(f"{output_path.suffix}.ffmpeg.log")

    with _control_lock():
        if is_recording():
            raise RuntimeError("A recording is already in progress.")
        _write_json_atomic(meta_path, meta)
        try:
            proc = subprocess.Popen(  # noqa: S603
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except OSError as error:
            _mark_start_failed(meta, error)
            raise RuntimeError(f"Could not start FFmpeg: {error}") from error
        _write_active_state(proc.pid, meta, log_path)

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
    if stderr:
        log_path.write_bytes(stderr)
    _clear_active_state()
    finalized = _finalize_recording(meta, error=stderr.decode(errors="replace")[-500:] or None)

    if finalized["status"] == "recorded":
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
    meta = _recording_metadata(
        output_path, monitor_name, sink_id, audio_format, mic_source, mic_id
    )
    meta_path = output_path.with_suffix(".json")
    log_path = output_path.with_suffix(f"{output_path.suffix}.ffmpeg.log")

    with _control_lock():
        if is_recording():
            raise RuntimeError("A recording is already in progress.")
        _write_json_atomic(meta_path, meta)
        try:
            with log_path.open("ab") as log:
                proc = subprocess.Popen(  # noqa: S603
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                    start_new_session=True,
                )
        except OSError as error:
            _mark_start_failed(meta, error)
            raise RuntimeError(f"Could not start FFmpeg: {error}") from error
        _write_active_state(proc.pid, meta, log_path)

    hooks.emit(
        "recording_started",
        pid=proc.pid,
        output_path=str(output_path),
        source=f"{monitor_name}.monitor",
    )

    return proc.pid
