"""Murmur plugin: transcription via faster-whisper."""

from __future__ import annotations

import click
from rich.console import Console

from murmur import hooks
from murmur.artifacts import ArtifactStore
from murmur.config import get_section

console = Console()


def _check_dep():
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        console.print(
            "[red]faster-whisper is not installed.[/red]\n"
            "Install with: [cyan]uv add murmur[transcribe][/cyan]"
        )
        return False


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _transcribe_file(file_path: str, model_size: str, lang: str) -> None:
    """Run local transcription into the canonical artifact directory."""
    from pathlib import Path

    from faster_whisper import WhisperModel

    console.print(f"[bold]Transcribing[/bold] {file_path} (model={model_size}, lang={lang})")

    audio_path = Path(file_path)
    store = ArtifactStore(audio_path)
    output_names = ["transcript_raw_local", "transcript_text", "transcript_srt"]
    transcript_path = store.path("transcript.txt")
    srt_path = store.path("transcript.srt")
    raw_path = store.path("raw-responses/transcribe-faster-whisper.json")
    _, should_run = store.begin_job(
        "transcribe",
        "faster-whisper",
        model=model_size,
        parameters={"language": lang, "compute_type": "int8"},
        output_paths=[raw_path, transcript_path, srt_path],
        output_artifacts=output_names,
    )
    if not should_run:
        console.print(f"[green]Transcript already complete:[/green] {transcript_path}")
        hooks.emit(
            "transcription_complete",
            audio_path=str(audio_path),
            transcript_path=str(transcript_path),
        )
        return

    try:
        whisper = WhisperModel(model_size, compute_type="int8")
        segments_iter, info = whisper.transcribe(file_path, language=lang)
        segments = list(segments_iter)

        text_lines = []
        srt_blocks = []
        raw_segments = []
        for i, segment in enumerate(segments, 1):
            line = f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}"
            text_lines.append(line)
            console.print(f"  [dim]{line}[/dim]")
            srt_blocks.append(
                f"{i}\n{_format_srt_time(segment.start)} --> "
                f"{_format_srt_time(segment.end)}\n{segment.text.strip()}\n"
            )
            raw_segments.append(
                {"id": i, "start": segment.start, "end": segment.end, "text": segment.text}
            )

        raw_path = store.write_json(
            "raw-responses/transcribe-faster-whisper.json",
            {
                "provider": "faster-whisper",
                "model": model_size,
                "language": getattr(info, "language", lang),
                "segments": raw_segments,
            },
        )
        transcript_path = store.write_text("transcript.txt", "\n".join(text_lines) + "\n")
        srt_path = store.write_text("transcript.srt", "\n".join(srt_blocks))
        provenance = {
            "provider": "faster-whisper",
            "model": model_size,
            "parameters": {"language": lang, "compute_type": "int8"},
        }
        store.register_artifact(
            "transcript_raw_local", raw_path, kind="raw_provider_response", provenance=provenance
        )
        store.register_artifact(
            "transcript_text", transcript_path, kind="transcript", provenance=provenance
        )
        store.register_artifact(
            "transcript_srt", srt_path, kind="subtitles", provenance=provenance
        )
        store.complete_job("transcribe", "faster-whisper")
    except Exception as error:
        store.fail_job("transcribe", "faster-whisper", error)
        raise

    hooks.emit(
        "transcription_complete",
        audio_path=str(audio_path),
        transcript_path=str(transcript_path),
    )

    console.print(f"\n[bold green]Transcript saved:[/bold green] {transcript_path}")
    console.print(f"[bold green]Subtitles saved:[/bold green] {srt_path}")


def register(cli: click.Group) -> None:
    """Register the transcribe command and hook."""

    @cli.command()
    @click.argument("file", type=click.Path(exists=True))
    @click.option("-m", "--model", default=None, help="Whisper model size (default: base).")
    @click.option("-l", "--language", default=None, help="Language code (default: en).")
    def transcribe(file: str, model: str | None, language: str | None):
        """Transcribe an audio file using faster-whisper."""
        if not _check_dep():
            raise SystemExit(1)

        cfg = get_section("transcribe")
        model_size = model or cfg.get("model", "base")
        lang = language or cfg.get("language", "en")
        _transcribe_file(file, model_size, lang)

    # Auto-transcribe on recording_saved if configured
    cfg = get_section("transcribe")
    if cfg.get("auto"):

        def _auto_transcribe(output_path: str, **kwargs):
            if not _check_dep():
                return
            model_size = cfg.get("model", "base")
            lang = cfg.get("language", "en")
            console.print(f"\n[bold]Auto-transcribing[/bold] {output_path}")
            _transcribe_file(output_path, model_size, lang)

        hooks.on("recording_saved", _auto_transcribe)
