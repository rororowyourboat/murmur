"""Murmur plugin: summarization via Ollama HTTP API."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import click
from rich.console import Console

from murmur import hooks
from murmur.config import get_section

console = Console()

OLLAMA_URL = "http://localhost:11434/api/generate"

AUDIO_EXTENSIONS = {".flac", ".mp3", ".wav", ".ogg", ".m4a", ".opus"}


def _ollama_generate(model: str, prompt: str) -> str:
    """Call the Ollama generate API and return the full response text."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "")
    except URLError as e:
        console.print(
            f"[red]Could not connect to Ollama at {OLLAMA_URL}.[/red]\n"
            f"Make sure Ollama is running: [cyan]ollama serve[/cyan]\n"
            f"Error: {e}"
        )
        raise SystemExit(1) from e


def _find_transcript(file_path: Path) -> Path:
    """Given an audio or transcript file, find the transcript .txt file.

    If the input is an audio file, looks for a sibling .txt file.
    If the input is already a .txt file, returns it directly.
    """
    if file_path.suffix == ".txt":
        return file_path

    # Audio file — look for transcript alongside it
    transcript_path = file_path.with_suffix(".txt")
    if transcript_path.exists():
        return transcript_path

    console.print(
        f"[red]No transcript found for {file_path.name}.[/red]\n"
        f"Expected: [cyan]{transcript_path}[/cyan]\n"
        f"Run [cyan]murmur transcribe {file_path}[/cyan] first."
    )
    raise SystemExit(1)


def register(cli: click.Group) -> None:
    """Register the summarize command and hook."""

    @cli.command()
    @click.argument("file", type=click.Path(exists=True))
    @click.option("-m", "--model", default=None, help="Ollama model name (default: llama3).")
    def summarize(file: str, model: str | None):
        """Summarize a transcript using Ollama.

        FILE can be a transcript (.txt) or an audio file — if given an audio
        file, the command looks for a sibling .txt transcript automatically.
        """
        cfg = get_section("summarize")
        model_name = model or cfg.get("model", "llama3")

        transcript_path = _find_transcript(Path(file))
        transcript = transcript_path.read_text()
        if not transcript.strip():
            console.print("[yellow]Transcript is empty, nothing to summarize.[/yellow]")
            return

        console.print(f"[bold]Summarizing[/bold] {transcript_path} (model={model_name})")

        prompt = (
            "Summarize the following meeting transcript. "
            "Include key decisions, action items, and important discussion points.\n\n"
            f"{transcript}"
        )

        summary = _ollama_generate(model_name, prompt)

        summary_path = transcript_path.with_suffix(".summary.md")
        summary_path.write_text(summary)

        console.print(f"\n[bold green]Summary saved:[/bold green] {summary_path}")
        console.print(f"\n{summary}")

    # Auto-summarize on transcription_complete if configured
    cfg = get_section("summarize")
    if cfg.get("auto"):

        def _auto_summarize(transcript_path: str, **kwargs):
            model_name = cfg.get("model", "llama3")
            transcript = Path(transcript_path).read_text()
            if not transcript.strip():
                return

            console.print(f"\n[bold]Auto-summarizing[/bold] {transcript_path}")
            prompt = (
                "Summarize the following meeting transcript. "
                "Include key decisions, action items, and important discussion points.\n\n"
                f"{transcript}"
            )
            summary = _ollama_generate(model_name, prompt)

            summary_path = Path(transcript_path).with_suffix(".summary.md")
            summary_path.write_text(summary)
            console.print(f"[bold green]Auto-summary saved:[/bold green] {summary_path}")

        hooks.on("transcription_complete", _auto_summarize)
