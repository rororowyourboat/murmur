"""Murmur plugin: speaker diarization via pyannote-audio."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from murmur.artifacts import ArtifactStore
from murmur.config import get_section
from murmur.speaker_profiles import (
    SIDES,
    add_speaker,
    delete_profile,
    export_profile,
    export_unknown_candidates,
    list_profiles,
)

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


def _diarize_file(file: str, token: str) -> tuple[Path, Path, set[str]]:
    """Run pyannote and persist canonical diarization artifacts."""
    from io import StringIO

    from pyannote.audio import Pipeline

    audio_path = Path(file)
    store = ArtifactStore(audio_path)
    rttm_path = store.path("speakers/diarization.rttm")
    timeline_path = store.path("speakers/diarization.txt")
    _, should_run = store.begin_job(
        "diarize",
        "pyannote-clusters",
        model="pyannote/speaker-diarization-3.1",
        parameters={"identity_mapping": "unresolved"},
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
            line = f"[{turn.start:.1f}s -> {turn.end:.1f}s] {speaker}"
            timeline.append(line)
            console.print(f"  [dim]{line}[/dim]")
        timeline_path = store.write_text("speakers/diarization.txt", "\n".join(timeline) + "\n")
        provenance = {
            "provider": "pyannote",
            "model": "pyannote/speaker-diarization-3.1",
            "parameters": {"identity_mapping": "unresolved"},
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
        store.complete_job("diarize", "pyannote-clusters")
        return rttm_path, timeline_path, seen_speakers
    except Exception as error:
        store.fail_job("diarize", "pyannote-clusters", error)
        raise


def register(cli: click.Group) -> None:
    """Register the diarize command."""

    @cli.group()
    def speakers():
        """Manage private confirmed speaker profiles."""

    @speakers.command("add")
    @click.argument("display_name")
    @click.option("--side", type=click.Choice(SIDES), required=True)
    @click.option("--clip", type=click.Path(exists=True), required=True)
    @click.option("--profile", default="default", show_default=True)
    @click.option("--source-recording", type=click.Path(exists=True), default=None)
    @click.option("--source-start", type=click.FloatRange(min=0), default=None)
    @click.option("--source-end", type=click.FloatRange(min=0), default=None)
    def speakers_add(
        display_name: str,
        side: str,
        clip: str,
        profile: str,
        source_recording: str | None,
        source_start: float | None,
        source_end: float | None,
    ):
        """Add a confirmed 2-10 second reference clip."""
        try:
            speaker = add_speaker(
                display_name,
                side=side,
                clip=clip,
                profile=profile,
                source_recording=source_recording,
                source_start=source_start,
                source_end=source_end,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            raise click.ClickException(str(error)) from error
        console.print(
            f"[green]Saved {speaker['display_name']} ({speaker['side']}) in profile "
            f"{profile}.[/green]"
        )

    @speakers.command("list")
    def speakers_list():
        """List profiles and locally stored voice references."""
        profiles = list_profiles()
        if not profiles:
            console.print("[yellow]No speaker profiles found.[/yellow]")
            return
        for profile in profiles:
            console.print(f"[bold]{profile['name']}[/bold]")
            for speaker in profile["speakers"]:
                console.print(
                    f"  {speaker['display_name']} ({speaker['side']}): "
                    f"{len(speaker.get('references', []))} reference(s)"
                )

    @speakers.command("export")
    @click.argument("profile", default="default")
    @click.option("--output", type=click.Path(), required=True)
    def speakers_export(profile: str, output: str):
        """Export a profile and clips as a ZIP archive."""
        try:
            path = export_profile(profile, output)
        except (FileNotFoundError, OSError, ValueError) as error:
            raise click.ClickException(str(error)) from error
        console.print(f"[green]Speaker profile exported:[/green] {path}")

    @speakers.command("delete")
    @click.argument("profile", default="default")
    @click.option("--speaker", "display_name", default=None)
    @click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
    def speakers_delete(profile: str, display_name: str | None, yes: bool):
        """Delete one speaker or an entire private profile."""
        target = (
            f"speaker {display_name!r} from {profile}" if display_name else f"profile {profile!r}"
        )
        if not yes and not click.confirm(f"Delete {target} and its stored voice clips?"):
            return
        try:
            delete_profile(profile, display_name)
        except (FileNotFoundError, OSError, ValueError) as error:
            raise click.ClickException(str(error)) from error
        console.print(f"[green]Deleted {target}.[/green]")

    @speakers.command("identify")
    @click.argument("file", type=click.Path(exists=True))
    def speakers_identify(file: str):
        """Export unresolved candidate clips for human confirmation."""
        try:
            index = export_unknown_candidates(file)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            raise click.ClickException(str(error)) from error
        console.print(
            f"[green]Candidate clips exported:[/green] {index.parent}\n"
            "Confirm one with `murmur speakers add NAME --side SIDE --clip CLIP`."
        )

    @cli.command()
    @click.argument("file", type=click.Path(exists=True))
    @click.option(
        "--hf-token",
        envvar="HF_TOKEN",
        default=None,
        help="Hugging Face token for pyannote model access.",
    )
    def diarize(file: str, hf_token: str | None):
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

        console.print(f"[bold]Diarizing[/bold] {file}")
        rttm_path, diarized_txt_path, seen_speakers = _diarize_file(file, token)

        console.print(
            f"\n[bold green]Diarization saved:[/bold green] {rttm_path}"
            f"\n[bold green]Speaker timeline:[/bold green] {diarized_txt_path}"
            f"\n  Speakers found: {len(seen_speakers)}"
        )
