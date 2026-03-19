"""Murmur plugin: Textual TUI dashboard."""

from __future__ import annotations

import json
import os
import signal
from datetime import datetime
from pathlib import Path

import click
from rich.text import Text

from murmur.config import get_section
from murmur.recorder import (
    _default_output_dir,
    is_recording,
    make_output_path,
    notify,
    record_background,
    resolve_sink,
)

AUDIO_SUFFIXES = {".flac", ".mp3", ".wav", ".ogg", ".m4a", ".opus"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_recordings() -> list[Path]:
    """Get sorted list of audio recording files."""
    recordings_dir = _default_output_dir()
    if not recordings_dir.exists():
        return []
    recs = sorted(recordings_dir.glob("meeting*.*"), reverse=True)
    return [r for r in recs if r.suffix in AUDIO_SUFFIXES][:30]


def _get_duration(rec: Path) -> str:
    meta_path = rec.with_suffix(".json")
    if not meta_path.exists():
        return "\u2014"
    try:
        meta = json.loads(meta_path.read_text())
        if "started_at" in meta and "stopped_at" in meta:
            start_dt = datetime.fromisoformat(meta["started_at"])
            stop_dt = datetime.fromisoformat(meta["stopped_at"])
            secs = int((stop_dt - start_dt).total_seconds())
            mins, secs = divmod(secs, 60)
            hours, mins = divmod(mins, 60)
            return f"{hours}h {mins}m {secs}s" if hours else f"{mins}m {secs}s"
    except json.JSONDecodeError, KeyError:
        pass
    return "\u2014"


def _artifact_exists(rec: Path, suffix: str) -> bool:
    if suffix == ".summary.md":
        return rec.with_suffix("").with_suffix(".summary.md").exists()
    return rec.with_suffix(suffix).exists()


def _read_artifact(rec: Path, suffix: str) -> str | None:
    if suffix == ".summary.md":
        path = rec.with_suffix("").with_suffix(".summary.md")
    else:
        path = rec.with_suffix(suffix)
    if not path.exists():
        return None
    return path.read_text()


# ---------------------------------------------------------------------------
# Textual App
# ---------------------------------------------------------------------------


def _build_app(audio_format: str):
    """Build and return the Textual App class (import textual lazily)."""
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal
    from textual.reactive import reactive
    from textual.widgets import (
        DataTable,
        Footer,
        Header,
        RichLog,
        Static,
        TabbedContent,
        TabPane,
    )

    class RecordingStatus(Static):
        """Top status bar showing recording state."""

        is_rec: reactive[bool] = reactive(False)
        elapsed: reactive[str] = reactive("")
        rec_file: reactive[str] = reactive("")

        def on_mount(self) -> None:
            self.set_interval(1.0, self._tick)

        def _tick(self) -> None:
            pid = is_recording()
            if pid:
                self.is_rec = True
                # Read elapsed from latest metadata
                recordings_dir = _default_output_dir()
                meta_files = sorted(recordings_dir.glob("meeting*.json"), reverse=True)
                if meta_files:
                    try:
                        meta = json.loads(meta_files[0].read_text())
                        started = datetime.fromisoformat(meta["started_at"])
                        delta = datetime.now() - started
                        mins, secs = divmod(int(delta.total_seconds()), 60)
                        hours, mins = divmod(mins, 60)
                        self.elapsed = (
                            f"{hours}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"
                        )
                        self.rec_file = Path(meta.get("output", "")).name
                    except json.JSONDecodeError, KeyError:
                        self.elapsed = "??:??"
            else:
                if self.is_rec:
                    # Recording just stopped — refresh the table
                    self.is_rec = False
                    self.elapsed = ""
                    self.rec_file = ""
                    self.app.query_one(MurmurApp).action_refresh_recordings()

        def render(self) -> Text:
            t = Text()
            if self.is_rec:
                t.append(" \u25cf REC ", style="bold white on red")
                t.append(f" {self.elapsed} ", style="bold white")
                t.append(f" {self.rec_file}", style="dim")
            else:
                t.append(" \u25cb IDLE ", style="dim")
                t.append(" Press ", style="dim")
                t.append("r", style="bold")
                t.append(" to start recording", style="dim")
            return t

    class MurmurApp(App):
        CSS = """
        Screen {
            layout: vertical;
        }
        #status-bar {
            height: 1;
            dock: top;
            background: $surface;
            margin-bottom: 1;
        }
        #main {
            height: 1fr;
        }
        #sidebar {
            width: 1fr;
            min-width: 40;
            max-width: 70;
        }
        #recordings-table {
            height: 1fr;
        }
        #detail {
            width: 2fr;
            min-width: 50;
        }
        #detail-content {
            height: 1fr;
        }
        TabbedContent {
            height: 1fr;
        }
        TabPane {
            height: 1fr;
            padding: 0 1;
        }
        RichLog {
            height: 1fr;
        }
        #info-panel {
            height: 1fr;
        }
        """

        TITLE = "Murmur"
        SUB_TITLE = "Meeting Recorder"

        BINDINGS = [
            Binding("r", "start_recording", "Record", priority=True),
            Binding("s", "stop_recording", "Stop", priority=True),
            Binding("t", "generate_transcript", "Transcribe", priority=True),
            Binding("m", "generate_summary", "Summarize", priority=True),
            Binding("d", "generate_diarization", "Diarize", priority=True),
            Binding("q", "quit", "Quit"),
            Binding("f5", "refresh_recordings", "Refresh"),
        ]

        selected_path: reactive[Path | None] = reactive(None)

        def compose(self) -> ComposeResult:
            yield Header()
            yield RecordingStatus(id="status-bar")
            with Horizontal(id="main"):
                yield DataTable(id="recordings-table", cursor_type="row")
                with TabbedContent(id="detail", initial="info"):
                    with TabPane("Info", id="info"):
                        yield RichLog(id="log-info", markup=True, wrap=True)
                    with TabPane("Transcript", id="transcript"):
                        yield RichLog(id="log-transcript", markup=True, wrap=True)
                    with TabPane("Summary", id="summary"):
                        yield RichLog(id="log-summary", markup=True, wrap=True)
                    with TabPane("Diarization", id="diarization"):
                        yield RichLog(id="log-diarization", markup=True, wrap=True)
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#recordings-table", DataTable)
            table.add_columns("Recording", "Duration", "T", "S", "M", "D")
            self._populate_table()
            self.set_interval(10.0, self._poll_recordings)

        def _populate_table(self) -> None:
            table = self.query_one("#recordings-table", DataTable)
            table.clear()
            recordings = _get_recordings()
            for rec in recordings:
                t_flag = "\u2713" if _artifact_exists(rec, ".txt") else "\u00b7"
                s_flag = "\u2713" if _artifact_exists(rec, ".srt") else "\u00b7"
                m_flag = "\u2713" if _artifact_exists(rec, ".summary.md") else "\u00b7"
                d_flag = "\u2713" if _artifact_exists(rec, ".diarized.txt") else "\u00b7"
                table.add_row(
                    rec.name,
                    _get_duration(rec),
                    t_flag,
                    s_flag,
                    m_flag,
                    d_flag,
                    key=str(rec),
                )
            if recordings:
                table.move_cursor(row=0)

        def _poll_recordings(self) -> None:
            """Refresh table if new recordings appear."""
            table = self.query_one("#recordings-table", DataTable)
            current_count = table.row_count
            new_count = len(_get_recordings())
            if new_count != current_count:
                self._populate_table()

        def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
            if event.row_key:
                self.selected_path = Path(str(event.row_key.value))

        def watch_selected_path(self, rec: Path | None) -> None:
            if rec is None:
                return
            self._load_detail(rec)

        def _load_detail(self, rec: Path) -> None:
            """Load all artifact tabs for a recording."""
            # Transcript
            log_t = self.query_one("#log-transcript", RichLog)
            log_t.clear()
            text = _read_artifact(rec, ".txt")
            if text:
                for line in text.splitlines():
                    log_t.write(line)
            else:
                log_t.write("[dim]No transcript. Run:[/dim]")
                log_t.write(f"[dim]  murmur transcribe {rec.name}[/dim]")

            # Summary
            log_s = self.query_one("#log-summary", RichLog)
            log_s.clear()
            text = _read_artifact(rec, ".summary.md")
            if text:
                for line in text.splitlines():
                    log_s.write(line)
            else:
                log_s.write("[dim]No summary. Run:[/dim]")
                log_s.write(f"[dim]  murmur summarize {rec.name}[/dim]")

            # Diarization
            log_d = self.query_one("#log-diarization", RichLog)
            log_d.clear()
            text = _read_artifact(rec, ".diarized.txt")
            if text:
                for line in text.splitlines():
                    log_d.write(line)
            else:
                log_d.write("[dim]No diarization. Run:[/dim]")
                log_d.write(f"[dim]  murmur diarize {rec.name}[/dim]")

            # Info (metadata JSON)
            log_i = self.query_one("#log-info", RichLog)
            log_i.clear()
            meta_path = rec.with_suffix(".json")
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    log_i.write(f"[bold]File:[/bold]     {rec.name}")
                    log_i.write(f"[bold]Path:[/bold]     {rec}")
                    log_i.write(
                        f"[bold]Size:[/bold]     {rec.stat().st_size / (1024 * 1024):.1f} MB"
                    )
                    log_i.write(f"[bold]Format:[/bold]   {meta.get('format', '?')}")
                    log_i.write(f"[bold]Source:[/bold]   {meta.get('source', '?')}")
                    log_i.write(f"[bold]Started:[/bold]  {meta.get('started_at', '?')}")
                    log_i.write(f"[bold]Stopped:[/bold]  {meta.get('stopped_at', '?')}")
                    if meta.get("dual_channel"):
                        log_i.write(f"[bold]Mic:[/bold]      {meta.get('mic_source', '?')}")
                    # Artifacts on disk
                    log_i.write("")
                    log_i.write("[bold]Artifacts:[/bold]")
                    artifacts = [
                        (".txt", "Transcript"),
                        (".srt", "Subtitles"),
                        (".summary.md", "Summary"),
                        (".diarized.txt", "Diarization"),
                        (".rttm", "RTTM"),
                    ]
                    for suffix, label in artifacts:
                        exists = _artifact_exists(rec, suffix)
                        icon = "[green]\u2713[/green]" if exists else "[dim]\u00b7[/dim]"
                        log_i.write(f"  {icon} {label}")
                except json.JSONDecodeError, OSError:
                    log_i.write("[dim]Could not read metadata.[/dim]")
            else:
                log_i.write("[dim]No metadata file found.[/dim]")

        def action_start_recording(self) -> None:
            if is_recording():
                self.notify("Already recording", severity="warning")
                return
            try:
                sink_id, monitor_name = resolve_sink(None)
            except SystemExit:
                self.notify("No audio sinks found", severity="error")
                return
            output_path = make_output_path(None, audio_format, None)
            record_background(output_path, monitor_name, sink_id, audio_format)
            notify("Recording started", f"Saving to {output_path.name}")
            self.notify(f"Recording: {output_path.name}")

        def action_stop_recording(self) -> None:
            pid = is_recording()
            if not pid:
                self.notify("Not recording", severity="warning")
                return
            try:
                os.kill(pid, signal.SIGTERM)
                notify("Recording stopped", "Meeting recording saved.")
                self.notify("Recording stopped")
            except ProcessLookupError:
                self.notify("Recording already stopped", severity="warning")
            # Table will refresh via poll or status bar detection
            self.set_timer(1.0, self._populate_table)

        def action_refresh_recordings(self) -> None:
            self._populate_table()
            self.notify("Refreshed")

        def _get_selected_rec(self) -> Path | None:
            """Return the currently selected recording path, or None."""
            rec = self.selected_path
            if rec is None or not rec.exists():
                self.notify("No recording selected", severity="warning")
                return None
            return rec

        def action_generate_transcript(self) -> None:
            rec = self._get_selected_rec()
            if rec is None:
                return
            if _artifact_exists(rec, ".txt"):
                self.notify("Transcript already exists", severity="information")
                return
            self.notify(f"Transcribing {rec.name}...")
            self.run_worker(self._run_transcribe(rec), name="transcribe", exclusive=True)

        def action_generate_summary(self) -> None:
            rec = self._get_selected_rec()
            if rec is None:
                return
            if _artifact_exists(rec, ".summary.md"):
                self.notify("Summary already exists", severity="information")
                return
            if not _artifact_exists(rec, ".txt"):
                self.notify("Transcribe first (press t)", severity="warning")
                return
            self.notify(f"Summarizing {rec.name}...")
            self.run_worker(self._run_summarize(rec), name="summarize", exclusive=True)

        def action_generate_diarization(self) -> None:
            rec = self._get_selected_rec()
            if rec is None:
                return
            if _artifact_exists(rec, ".diarized.txt"):
                self.notify("Diarization already exists", severity="information")
                return
            self.notify(f"Diarizing {rec.name}...")
            self.run_worker(self._run_diarize(rec), name="diarize", exclusive=True)

        async def _run_transcribe(self, rec: Path) -> None:
            """Run transcription in a background worker."""
            try:
                from murmur.plugins.transcribe import _check_dep, _transcribe_file

                if not _check_dep():
                    self.notify(
                        "faster-whisper not installed. Run: uv add murmur[transcribe]",
                        severity="error",
                        timeout=8,
                    )
                    return
                cfg = get_section("transcribe")
                model_size = cfg.get("model", "base")
                lang = cfg.get("language", "en")
                # Run the blocking transcription in a thread
                await self._run_in_thread(_transcribe_file, str(rec), model_size, lang)
                self.notify(f"Transcript ready: {rec.stem}.txt")
            except Exception as e:
                self.notify(f"Transcription failed: {e}", severity="error", timeout=8)
            self._populate_table()
            self._load_detail(rec)

        async def _run_summarize(self, rec: Path) -> None:
            """Run summarization in a background worker."""
            try:
                from murmur.plugins.summarize import (
                    DEFAULT_MODEL,
                    _find_transcript,
                    _llm_generate,
                )

                cfg = get_section("summarize")
                model_name = cfg.get("model", DEFAULT_MODEL)
                transcript_path = _find_transcript(rec)
                transcript = transcript_path.read_text()
                if not transcript.strip():
                    self.notify("Transcript is empty", severity="warning")
                    return
                summary = await self._run_in_thread(_llm_generate, model_name, transcript)
                summary_path = transcript_path.with_suffix(".summary.md")
                summary_path.write_text(summary)
                self.notify(f"Summary ready: {summary_path.name}")
            except SystemExit:
                self.notify(
                    "LLM API call failed. Check API keys in .env",
                    severity="error",
                    timeout=8,
                )
            except Exception as e:
                self.notify(f"Summarization failed: {e}", severity="error", timeout=8)
            self._populate_table()
            self._load_detail(rec)

        async def _run_diarize(self, rec: Path) -> None:
            """Run diarization in a background worker."""
            try:
                from murmur.plugins.diarize import _check_dep

                if not _check_dep():
                    self.notify(
                        "pyannote-audio not installed. Run: uv add murmur[diarize]",
                        severity="error",
                        timeout=8,
                    )
                    return

                cfg = get_section("diarize")
                token = cfg.get("hf_token") or os.environ.get("HF_TOKEN")
                if not token:
                    self.notify(
                        "HF_TOKEN required for diarization. Set in config or env.",
                        severity="error",
                        timeout=8,
                    )
                    return

                from pyannote.audio import Pipeline

                pipeline = await self._run_in_thread(
                    Pipeline.from_pretrained,
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=token,
                )
                diarization = await self._run_in_thread(pipeline, str(rec))

                rttm_path = rec.with_suffix(".rttm")
                diarized_path = rec.with_suffix(".diarized.txt")
                with rttm_path.open("w") as f:
                    diarization.write_rttm(f)
                with diarized_path.open("w") as f:
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        f.write(f"[{turn.start:.1f}s -> {turn.end:.1f}s] {speaker}\n")

                self.notify(f"Diarization ready: {diarized_path.name}")
            except Exception as e:
                self.notify(f"Diarization failed: {e}", severity="error", timeout=8)
            self._populate_table()
            self._load_detail(rec)

        @staticmethod
        async def _run_in_thread(fn, *args, **kwargs):
            """Run a blocking function in a thread."""
            import asyncio

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    return MurmurApp


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


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
        """Live dashboard showing recordings, transcripts, and summaries.

        Requires: uv pip install murmur[tui]
        """
        try:
            import textual  # noqa: F401
        except ImportError:
            from rich.console import Console

            Console().print(
                "[red]textual is not installed.[/red]\n"
                "Install with: [cyan]uv pip install murmur[tui][/cyan]"
            )
            raise SystemExit(1) from None
        app_cls = _build_app(audio_format)
        app_cls().run()
