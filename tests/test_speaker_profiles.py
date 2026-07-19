"""Tests for private speaker profile lifecycle and reference validation."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from murmur.artifacts import ArtifactStore
from murmur.speaker_profiles import (
    add_speaker,
    delete_profile,
    export_profile,
    export_unknown_candidates,
    list_profiles,
    load_profile,
    reference_payload,
)


@pytest.fixture(autouse=True)
def isolated_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("MURMUR_SPEAKER_PROFILES_DIR", str(tmp_path / "profiles"))


def _fake_normalize(source: Path, destination: Path) -> float:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"RIFF confirmed voice")
    return 4.0


def test_profile_add_reference_export_and_delete(tmp_path):
    source = tmp_path / "rohan.wav"
    source.write_bytes(b"source audio")
    with patch("murmur.speaker_profiles._normalize_clip", side_effect=_fake_normalize):
        speaker = add_speaker(
            "Rohan",
            side="local",
            clip=source,
            source_recording=tmp_path / "meeting.mka",
            source_start=12.0,
            source_end=16.0,
        )

    assert speaker["display_name"] == "Rohan"
    assert speaker["side"] == "local"
    assert speaker["references"][0]["duration_secs"] == 4.0
    assert list_profiles()[0]["name"] == "default"
    names, references = reference_payload("default", "local")
    assert names == ["Rohan"]
    assert references[0].startswith("data:audio/wav;base64,")
    assert "data:audio" not in json.dumps(load_profile("default"))
    assert (tmp_path / "profiles").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "profiles/default/profile.json").stat().st_mode & 0o777 == 0o600

    archive = export_profile("default", tmp_path / "profile.zip")
    assert archive.is_file()
    delete_profile("default", "Rohan")
    assert load_profile("default")["speakers"] == []
    delete_profile("default")
    assert not (tmp_path / "profiles/default").exists()


def test_profile_rejects_side_change_for_existing_identity(tmp_path):
    source = tmp_path / "voice.wav"
    source.write_bytes(b"source audio")
    with patch("murmur.speaker_profiles._normalize_clip", side_effect=_fake_normalize):
        add_speaker("Abby", side="remote", clip=source)
        with pytest.raises(ValueError, match="already belongs"):
            add_speaker("Abby", side="local", clip=source)


def test_identify_exports_one_candidate_per_unresolved_label(tmp_path):
    recording = tmp_path / "meeting.mka"
    recording.write_bytes(b"recording")
    store = ArtifactStore(recording)
    store.ensure_manifest()
    store.write_json(
        "transcript.json",
        {
            "segments": [
                {
                    "speaker": "unknown:local:chunk-0000:A",
                    "side": "local",
                    "stream_index": 1,
                    "start": 1,
                    "end": 5,
                },
                {
                    "speaker": "unknown:local:chunk-0000:A",
                    "side": "local",
                    "stream_index": 1,
                    "start": 7,
                    "end": 9,
                },
                {
                    "speaker": "unknown:remote:chunk-0000:B",
                    "side": "remote",
                    "stream_index": 2,
                    "start": 3,
                    "end": 8,
                },
                {"speaker": "Rohan", "start": 0, "end": 1},
            ]
        },
    )

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"RIFF candidate")
        return SimpleNamespace(returncode=0, stderr="")

    with (
        patch("murmur.speaker_profiles.shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("murmur.speaker_profiles.subprocess.run", side_effect=fake_run) as run,
    ):
        index = export_unknown_candidates(recording)

    candidates = json.loads(index.read_text())["candidates"]
    assert len(candidates) == 2
    assert [candidate["side"] for candidate in candidates] == ["local", "remote"]
    assert len(run.call_args_list) == 2


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg integration tools are unavailable",
)
def test_reference_duration_is_validated_with_real_audio(tmp_path):
    source = tmp_path / "short.wav"
    subprocess.run(  # noqa: S603
        [
            str(shutil.which("ffmpeg")),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            str(source),
        ],
        check=True,
    )

    with pytest.raises(ValueError, match="2-10 seconds"):
        add_speaker("Sujata", side="remote", clip=source)
