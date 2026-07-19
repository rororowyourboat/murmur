"""Private, reusable speaker profiles backed by confirmed reference clips."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from murmur.artifacts import ArtifactStore, fingerprint_file

PROFILE_SCHEMA_VERSION = 1
MIN_REFERENCE_SECONDS = 2.0
MAX_REFERENCE_SECONDS = 10.0
SIDES = ("local", "remote", "unknown")


def profiles_root() -> Path:
    configured = os.environ.get("MURMUR_SPEAKER_PROFILES_DIR")
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".config" / "murmur" / "speaker-profiles"
    ).resolve()


def _safe_name(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError(f"{label} must use letters, numbers, dots, dashes, or underscores.")
    return value


def _profile_dir(profile: str) -> Path:
    return profiles_root() / _safe_name(profile, "Profile name")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _private_mkdir(path: Path) -> None:
    root = profiles_root()
    if not path.resolve().is_relative_to(root):
        raise ValueError("Speaker profile paths must remain inside the private profile root.")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    current = root
    for part in path.resolve().relative_to(root).parts:
        current /= part
        current.mkdir(exist_ok=True, mode=0o700)
        current.chmod(0o700)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _private_mkdir(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_profile(profile: str = "default") -> dict[str, Any]:
    profile_path = _profile_dir(profile) / "profile.json"
    try:
        payload = json.loads(profile_path.read_text())
    except FileNotFoundError:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "name": profile,
            "speakers": [],
            "created_at": _now(),
            "updated_at": _now(),
        }
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Could not read speaker profile {profile}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("speakers"), list):
        raise ValueError(f"Speaker profile {profile} has an invalid schema.")
    return payload


def list_profiles() -> list[dict[str, Any]]:
    root = profiles_root()
    if not root.exists():
        return []
    profiles = []
    for path in sorted(root.glob("*/profile.json")):
        profiles.append(load_profile(path.parent.name))
    return profiles


def _probe_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate speaker reference clips.")
    result = subprocess.run(  # noqa: S603
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Reference clip is not readable audio: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
        if not payload.get("streams"):
            raise KeyError("streams")
        return float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Reference clip does not contain a valid audio stream.") from error


def _normalize_clip(source: Path, destination: Path) -> float:
    if not source.is_file():
        raise FileNotFoundError(f"Reference clip not found: {source}")
    duration = _probe_duration(source)
    if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
        raise ValueError(
            f"Reference clips must be {MIN_REFERENCE_SECONDS:g}-{MAX_REFERENCE_SECONDS:g} "
            f"seconds; received {duration:.2f} seconds."
        )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to normalize speaker reference clips.")
    _private_mkdir(destination.parent)
    result = subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not normalize reference clip: {result.stderr.strip()}")
    destination.chmod(0o600)
    return duration


def add_speaker(
    display_name: str,
    *,
    side: str,
    clip: str | Path,
    profile: str = "default",
    source_recording: str | Path | None = None,
    source_start: float | None = None,
    source_end: float | None = None,
) -> dict[str, Any]:
    """Add a confirmed clip to a named speaker, creating the profile if needed."""
    name = display_name.strip()
    if not name:
        raise ValueError("Display name cannot be empty.")
    if side not in SIDES:
        raise ValueError(f"Side must be one of: {', '.join(SIDES)}.")
    payload = load_profile(profile)
    source = Path(clip).expanduser().resolve()
    speaker = next((item for item in payload["speakers"] if item["display_name"] == name), None)
    if speaker is not None and speaker["side"] != side:
        raise ValueError(f"{name} already belongs to side {speaker['side']} in profile {profile}.")
    if speaker is None:
        speaker = {
            "id": f"speaker-{uuid4().hex[:12]}",
            "display_name": name,
            "side": side,
            "references": [],
            "created_at": _now(),
        }
        payload["speakers"].append(speaker)
    reference_id = f"reference-{uuid4().hex[:12]}"
    relative = Path("clips") / f"{speaker['id']}-{reference_id}.wav"
    destination = _profile_dir(profile) / relative
    duration = _normalize_clip(source, destination)
    provenance: dict[str, Any] = {
        "input_clip": str(source),
        "input_fingerprint": fingerprint_file(source),
        "confirmed_at": _now(),
    }
    if source_recording is not None:
        provenance["source_recording"] = str(Path(source_recording).expanduser().resolve())
    if source_start is not None:
        provenance["source_start"] = source_start
    if source_end is not None:
        provenance["source_end"] = source_end
    reference = {
        "id": reference_id,
        "path": str(relative),
        "duration_secs": round(duration, 3),
        "fingerprint": fingerprint_file(destination),
        "provenance": provenance,
    }
    speaker["references"].append(reference)
    speaker["updated_at"] = _now()
    payload["updated_at"] = _now()
    _write_json(_profile_dir(profile) / "profile.json", payload)
    return speaker


def reference_payload(profile: str, side: str, *, limit: int = 4) -> tuple[list[str], list[str]]:
    """Return aligned known-speaker names and in-memory audio data URLs."""
    payload = load_profile(profile)
    allowed = set(SIDES) if side == "unknown" else {side}
    names: list[str] = []
    references: list[str] = []
    for speaker in payload["speakers"]:
        if speaker.get("side") not in allowed or not speaker.get("references"):
            continue
        reference = speaker["references"][0]
        clip_path = _profile_dir(profile) / reference["path"]
        if fingerprint_file(clip_path)["sha256"] != reference["fingerprint"]["sha256"]:
            raise ValueError(f"Reference clip checksum failed for {speaker['display_name']}.")
        encoded = base64.b64encode(clip_path.read_bytes()).decode("ascii")
        names.append(speaker["display_name"])
        references.append(f"data:audio/wav;base64,{encoded}")
        if len(names) == limit:
            break
    return names, references


def export_profile(profile: str, output: str | Path) -> Path:
    """Export profile metadata and clips as a portable ZIP archive."""
    directory = _profile_dir(profile)
    if not (directory / "profile.json").is_file():
        raise FileNotFoundError(f"Speaker profile not found: {profile}")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory))
    return output_path


def delete_profile(profile: str, display_name: str | None = None) -> None:
    """Delete one speaker and their clips, or delete the complete profile."""
    directory = _profile_dir(profile)
    if display_name is None:
        if not directory.exists():
            raise FileNotFoundError(f"Speaker profile not found: {profile}")
        shutil.rmtree(directory)
        return
    payload = load_profile(profile)
    speaker = next(
        (item for item in payload["speakers"] if item["display_name"] == display_name), None
    )
    if speaker is None:
        raise ValueError(f"Speaker not found in profile {profile}: {display_name}")
    for reference in speaker.get("references", []):
        (_profile_dir(profile) / reference["path"]).unlink(missing_ok=True)
    payload["speakers"].remove(speaker)
    payload["updated_at"] = _now()
    _write_json(directory / "profile.json", payload)


def export_unknown_candidates(recording: str | Path) -> Path:
    """Export one candidate clip per unresolved diarization label for confirmation."""
    recording_path = Path(recording).expanduser().resolve()
    store = ArtifactStore(recording_path)
    transcript_path = store.path("transcript.json")
    payload = json.loads(transcript_path.read_text())
    segments = payload.get("segments", [])
    candidates: dict[str, dict[str, Any]] = {}
    for segment in segments:
        speaker = str(segment.get("speaker", ""))
        if speaker.startswith("unknown:") and speaker not in candidates:
            candidates[speaker] = segment
    output_dir = store.path("speakers/candidates")
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export speaker candidates.")
    index = []
    for number, (speaker, segment) in enumerate(candidates.items(), 1):
        output = output_dir / f"candidate-{number:03d}.wav"
        result = subprocess.run(  # noqa: S603
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(segment["start"]),
                "-i",
                str(recording_path),
                "-t",
                str(
                    min(
                        MAX_REFERENCE_SECONDS,
                        max(MIN_REFERENCE_SECONDS, segment["end"] - segment["start"]),
                    )
                ),
                "-map",
                f"0:{segment.get('stream_index', 0)}",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Could not export {speaker}: {result.stderr.strip()}")
        index.append(
            {
                "speaker": speaker,
                "side": segment.get("side", "unknown"),
                "start": segment["start"],
                "end": segment["end"],
                "clip": output.name,
            }
        )
    index_path = store.write_json("speakers/candidates/index.json", {"candidates": index})
    store.register_artifact(
        "speaker_candidates",
        index_path,
        kind="speaker_identity_candidates",
        provenance={"source": str(recording_path)},
    )
    return index_path
