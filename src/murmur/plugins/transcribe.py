"""Murmur transcription plugin with local and OpenAI providers."""

from __future__ import annotations

import click
from rich.console import Console

from murmur import hooks
from murmur.artifacts import ArtifactStore
from murmur.cloud_transcribe import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_OVERLAP_SECONDS,
    transcribe_openai,
)
from murmur.cloud_transcribe import (
    DEFAULT_MODEL as DEFAULT_OPENAI_MODEL,
)
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
    @click.option(
        "--provider",
        type=click.Choice(["local", "openai"]),
        default=None,
        help="Transcription provider (default: local).",
    )
    @click.option("-m", "--model", default=None, help="Provider model name.")
    @click.option("-l", "--language", default=None, help="Language code (default: en).")
    @click.option(
        "--resume/--restart",
        default=True,
        help="Resume valid completed chunks or submit every chunk again.",
    )
    @click.option(
        "--chunk-seconds",
        type=click.FloatRange(min=1.0),
        default=DEFAULT_CHUNK_SECONDS,
        show_default=True,
    )
    @click.option(
        "--overlap-seconds",
        type=click.FloatRange(min=0.0),
        default=DEFAULT_OVERLAP_SECONDS,
        show_default=True,
    )
    @click.option("--prompt", default=None, help="Optional vocabulary/context prompt.")
    def transcribe(
        file: str,
        provider: str | None,
        model: str | None,
        language: str | None,
        resume: bool,
        chunk_seconds: float,
        overlap_seconds: float,
        prompt: str | None,
    ):
        """Transcribe audio locally or with resumable OpenAI cloud processing."""
        cfg = get_section("transcribe")
        provider_name = provider or cfg.get("provider", "local")
        lang = language or cfg.get("language", "en")
        if provider_name == "local":
            if not _check_dep():
                raise SystemExit(1)
            model_size = model or cfg.get("model", "base")
            _transcribe_file(file, model_size, lang)
            return

        model_name = model or cfg.get("openai_model", DEFAULT_OPENAI_MODEL)
        try:
            result = transcribe_openai(
                file,
                model=model_name,
                language=lang,
                prompt=prompt,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                resume=resume,
            )
        except (RuntimeError, ValueError) as error:
            raise click.ClickException(str(error)) from error
        transcript_path = ArtifactStore(file).path("transcript.md")
        hooks.emit(
            "transcription_complete",
            audio_path=str(file),
            transcript_path=str(transcript_path),
        )
        console.print(
            f"[bold green]Transcript saved:[/bold green] {transcript_path} "
            f"({len(result['segments'])} segments)"
        )

    # Auto-transcribe on recording_saved if configured
    cfg = get_section("transcribe")
    if cfg.get("auto"):

        def _auto_transcribe(output_path: str, **kwargs):
            lang = cfg.get("language", "en")
            console.print(f"\n[bold]Auto-transcribing[/bold] {output_path}")
            if cfg.get("provider", "local") == "openai":
                transcribe_openai(
                    output_path,
                    model=cfg.get("openai_model", DEFAULT_OPENAI_MODEL),
                    language=lang,
                    chunk_seconds=cfg.get("chunk_seconds", DEFAULT_CHUNK_SECONDS),
                    overlap_seconds=cfg.get("overlap_seconds", DEFAULT_OVERLAP_SECONDS),
                )
                transcript_path = ArtifactStore(output_path).path("transcript.md")
                hooks.emit(
                    "transcription_complete",
                    audio_path=output_path,
                    transcript_path=str(transcript_path),
                )
                return
            if not _check_dep():
                return
            _transcribe_file(output_path, cfg.get("model", "base"), lang)

        hooks.on("recording_saved", _auto_transcribe)
