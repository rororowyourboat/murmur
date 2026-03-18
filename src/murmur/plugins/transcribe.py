"""Murmur plugin: transcription via faster-whisper."""

from __future__ import annotations

import click
from rich.console import Console

from murmur import hooks
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
    """Run transcription on a file, producing .txt and .srt outputs."""
    from pathlib import Path

    from faster_whisper import WhisperModel

    console.print(f"[bold]Transcribing[/bold] {file_path} (model={model_size}, lang={lang})")

    whisper = WhisperModel(model_size, compute_type="int8")
    segments_iter, _info = whisper.transcribe(file_path, language=lang)

    audio_path = Path(file_path)
    transcript_path = audio_path.with_suffix(".txt")
    srt_path = audio_path.with_suffix(".srt")

    segments = list(segments_iter)

    with transcript_path.open("w") as txt_f, srt_path.open("w") as srt_f:
        for i, segment in enumerate(segments, 1):
            line = f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}"
            txt_f.write(line + "\n")
            console.print(f"  [dim]{line}[/dim]")

            # SRT format
            srt_f.write(f"{i}\n")
            srt_f.write(f"{_format_srt_time(segment.start)} --> {_format_srt_time(segment.end)}\n")
            srt_f.write(f"{segment.text.strip()}\n\n")

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
