"""Murmur plugin: speaker diarization via pyannote-audio."""

from __future__ import annotations

import click
from rich.console import Console

from murmur.config import get_section

console = Console()


def _check_dep():
    try:
        import pyannote.audio  # noqa: F401

        return True
    except ImportError:
        console.print(
            "[red]pyannote-audio is not installed.[/red]\n"
            "Install with: [cyan]uv add murmur[diarize][/cyan]"
        )
        return False


def register(cli: click.Group) -> None:
    """Register the diarize command."""

    @cli.command()
    @click.argument("file", type=click.Path(exists=True))
    @click.option(
        "--hf-token",
        envvar="HF_TOKEN",
        default=None,
        help="Hugging Face token for pyannote model access.",
    )
    @click.option(
        "--speakers",
        default=None,
        help="Comma-separated speaker names to assign (e.g., 'Alice,Bob').",
    )
    def diarize(file: str, hf_token: str | None, speakers: str | None):
        """Identify speakers in an audio file using pyannote-audio."""
        if not _check_dep():
            raise SystemExit(1)

        from pathlib import Path

        from pyannote.audio import Pipeline

        cfg = get_section("diarize")
        token = hf_token or cfg.get("hf_token")
        if not token:
            console.print(
                "[red]Hugging Face token required for pyannote.[/red]\n"
                "Set HF_TOKEN env var or add hf_token to [diarize] config."
            )
            raise SystemExit(1)

        # Parse speaker name mapping
        speaker_names: dict[str, str] = {}
        if speakers:
            for i, name in enumerate(speakers.split(",")):
                speaker_names[f"SPEAKER_{i:02d}"] = name.strip()

        console.print(f"[bold]Diarizing[/bold] {file}")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
        diarization = pipeline(file)

        audio_path = Path(file)
        rttm_path = audio_path.with_suffix(".rttm")
        diarized_txt_path = audio_path.with_suffix(".diarized.txt")

        # Write RTTM
        with rttm_path.open("w") as f:
            diarization.write_rttm(f)

        # Write human-readable diarized transcript
        seen_speakers = set()
        with diarized_txt_path.open("w") as f:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                seen_speakers.add(speaker)
                label = speaker_names.get(speaker, speaker)
                line = f"[{turn.start:.1f}s -> {turn.end:.1f}s] {label}"
                f.write(line + "\n")
                console.print(f"  [dim]{line}[/dim]")

        console.print(
            f"\n[bold green]Diarization saved:[/bold green] {rttm_path}"
            f"\n[bold green]Speaker timeline:[/bold green] {diarized_txt_path}"
            f"\n  Speakers found: {len(seen_speakers)}"
        )
        if speaker_names:
            mapped = ", ".join(
                f"{k} -> {v}" for k, v in speaker_names.items() if k in seen_speakers
            )
            if mapped:
                console.print(f"  Mapped: {mapped}")
