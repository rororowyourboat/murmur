"""Murmur plugin: speaker diarization via pyannote-audio."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from murmur.artifacts import ArtifactStore
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


def _diarize_file(
    file: str, token: str, speaker_names: dict[str, str] | None = None
) -> tuple[Path, Path, set[str]]:
    """Run pyannote and persist canonical diarization artifacts."""
    from io import StringIO

    from pyannote.audio import Pipeline

    speaker_names = speaker_names or {}
    audio_path = Path(file)
    store = ArtifactStore(audio_path)
    rttm_path = store.path("speakers/diarization.rttm")
    timeline_path = store.path("speakers/diarization.txt")
    _, should_run = store.begin_job(
        "diarize",
        "pyannote",
        model="pyannote/speaker-diarization-3.1",
        parameters={"speaker_names": list(speaker_names.values())},
        output_paths=[rttm_path, timeline_path],
        output_artifacts=["diarization_rttm", "diarization_timeline"],
    )
    if not should_run:
        return rttm_path, timeline_path, set()

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
        diarization = pipeline(file)
        rttm_buffer = StringIO()
        diarization.write_rttm(rttm_buffer)
        rttm_path = store.write_text("speakers/diarization.rttm", rttm_buffer.getvalue())

        seen_speakers = set()
        timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            seen_speakers.add(speaker)
            label = speaker_names.get(speaker, speaker)
            line = f"[{turn.start:.1f}s -> {turn.end:.1f}s] {label}"
            timeline.append(line)
            console.print(f"  [dim]{line}[/dim]")
        timeline_path = store.write_text("speakers/diarization.txt", "\n".join(timeline) + "\n")
        provenance = {
            "provider": "pyannote",
            "model": "pyannote/speaker-diarization-3.1",
            "parameters": {"speaker_names": list(speaker_names.values())},
        }
        store.register_artifact(
            "diarization_rttm", rttm_path, kind="speaker_timeline", provenance=provenance
        )
        store.register_artifact(
            "diarization_timeline",
            timeline_path,
            kind="speaker_timeline_text",
            provenance=provenance,
        )
        store.complete_job("diarize", "pyannote")
        return rttm_path, timeline_path, seen_speakers
    except Exception as error:
        store.fail_job("diarize", "pyannote", error)
        raise


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
        rttm_path, diarized_txt_path, seen_speakers = _diarize_file(file, token, speaker_names)

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
