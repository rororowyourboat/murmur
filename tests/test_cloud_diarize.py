"""Tests for channel-aware, identity-anchored OpenAI diarization."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from murmur.artifacts import ArtifactStore, fingerprint_file
from murmur.cloud_diarize import (
    Track,
    _merge_track_segments,
    _normalize_segments,
    _select_tracks,
    _submit_diarized_chunk,
    transcribe_openai_diarized,
)


class FakeTranscriptions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        response = next(self.responses)
        return SimpleNamespace(model_dump=lambda mode="json": response)


def _client(responses):
    transcriptions = FakeTranscriptions(responses)
    return SimpleNamespace(
        audio=SimpleNamespace(transcriptions=transcriptions), calls=transcriptions
    )


def _recording(tmp_path, *, multitrack=True, duration=30):
    recording = tmp_path / "meeting.mka"
    recording.write_bytes(b"original recording")
    streams = (
        [
            {"index": 0, "codec_type": "audio", "source_role": "mixed"},
            {"index": 1, "codec_type": "audio", "source_role": "microphone"},
            {"index": 2, "codec_type": "audio", "source_role": "call_output"},
        ]
        if multitrack
        else [{"index": 0, "codec_type": "audio", "source_role": "mixed"}]
    )
    recording.with_suffix(".json").write_text(
        json.dumps({"duration_secs": duration, "streams": streams})
    )
    return recording


def _profile(root: Path):
    directory = root / "default"
    clips = directory / "clips"
    clips.mkdir(parents=True)
    local = clips / "rohan.wav"
    remote = clips / "abby.wav"
    local.write_bytes(b"RIFF local")
    remote.write_bytes(b"RIFF remote")
    payload = {
        "schema_version": 1,
        "name": "default",
        "speakers": [
            {
                "id": "speaker-rohan",
                "display_name": "Rohan",
                "side": "local",
                "references": [
                    {
                        "path": "clips/rohan.wav",
                        "duration_secs": 4,
                        "fingerprint": fingerprint_file(local),
                    }
                ],
            },
            {
                "id": "speaker-abby",
                "display_name": "Abby",
                "side": "remote",
                "references": [
                    {
                        "path": "clips/abby.wav",
                        "duration_secs": 4,
                        "fingerprint": fingerprint_file(remote),
                    }
                ],
            },
        ],
    }
    (directory / "profile.json").write_text(json.dumps(payload))


def _fake_extract(recording, output, chunk, stream_index):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(f"wav:{chunk.index}:{stream_index}".encode())


def test_track_selection_prefers_independent_microphone_and_call_output():
    manifest = {
        "media": {
            "streams": [
                {"index": 0, "source_role": "mixed"},
                {"index": 1, "source_role": "microphone"},
                {"index": 2, "source_role": "call_output"},
            ]
        }
    }
    assert _select_tracks(manifest) == [Track("local", 1), Track("remote", 2)]


def test_unknown_labels_are_chunk_scoped_and_known_names_are_anchored():
    first = _normalize_segments(
        {"segments": [{"speaker": "Rohan", "start": 0, "end": 2, "text": "hello"}]},
        side="local",
        stream_index=1,
        chunk_index=0,
        chunk_start=0,
        chunk_duration=10,
        known_names={"Rohan"},
    )
    second = _normalize_segments(
        {"segments": [{"speaker": "A", "start": 0, "end": 2, "text": "unknown"}]},
        side="local",
        stream_index=1,
        chunk_index=1,
        chunk_start=8,
        chunk_duration=10,
        known_names={"Rohan"},
    )
    assert first[0]["speaker"] == "Rohan"
    assert second[0]["speaker"] == "unknown:local:chunk-0001:A"


def test_track_merge_deduplicates_only_within_one_timeline():
    chunks = [
        [{"start": 0, "end": 10, "text": "we chose the launch plan"}],
        [{"start": 8, "end": 14, "text": "the launch plan next"}],
    ]
    merged = _merge_track_segments(chunks)
    assert [segment["text"] for segment in merged] == ["we chose the launch plan", "next"]
    assert merged[1]["start"] == 10


def test_channel_aware_pipeline_anchors_profiles_and_preserves_overlap(tmp_path, monkeypatch):
    profile_root = tmp_path / "profiles"
    monkeypatch.setenv("MURMUR_SPEAKER_PROFILES_DIR", str(profile_root))
    _profile(profile_root)
    recording = _recording(tmp_path)
    client = _client(
        [
            {"segments": [{"speaker": "Rohan", "start": 0, "end": 10, "text": "local speech"}]},
            {
                "segments": [
                    {"speaker": "Abby", "start": 5, "end": 12, "text": "remote overlap"},
                    {"speaker": "B", "start": 14, "end": 18, "text": "unknown remote"},
                ]
            },
        ]
    )
    with patch("murmur.cloud_diarize._extract_chunk", side_effect=_fake_extract):
        first = transcribe_openai_diarized(recording, client=client)
        second = transcribe_openai_diarized(recording, client=client)

    assert first == second
    assert first["channel_aware"] is True
    assert [segment["speaker"] for segment in first["segments"]] == [
        "Rohan",
        "Abby",
        "unknown:remote:chunk-0000:B",
    ]
    assert first["segments"][0]["end"] == 10
    assert first["segments"][1]["start"] == 5
    assert len(client.calls.requests) == 2
    assert client.calls.requests[0]["known_speaker_names"] == ["Rohan"]
    assert client.calls.requests[1]["known_speaker_names"] == ["Abby"]
    assert "data:audio" not in (ArtifactStore(recording).jobs_path.read_text())
    assert recording.read_bytes() == b"original recording"


def test_diarized_sdk_request_uses_auto_chunking_and_known_references(tmp_path):
    import httpx
    from openai import OpenAI

    chunk = tmp_path / "chunk.wav"
    chunk.write_bytes(b"RIFF fake")

    def handler(request):
        body = request.read()
        assert b"diarized_json" in body
        assert b'name="chunking_strategy"' in body
        assert b"auto" in body
        assert b'name="known_speaker_names[]"' in body
        assert b"Rohan" in body
        assert b'name="known_speaker_references[]"' in body
        return httpx.Response(200, json={"segments": []})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAI(
        api_key="test-api-key", base_url="https://api.openai.test/v1", http_client=http_client
    )
    result = _submit_diarized_chunk(
        client,
        chunk,
        model="gpt-4o-transcribe-diarize",
        language="en",
        known_names=["Rohan"],
        known_references=["data:audio/wav;base64,UklGRg=="],
    )
    assert result["segments"] == []
