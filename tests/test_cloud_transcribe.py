"""Tests for resumable OpenAI cloud transcription."""

import json
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from murmur.artifacts import ArtifactStore
from murmur.cloud_transcribe import (
    Chunk,
    TranscriptionProviderError,
    _dedupe_boundary,
    _extract_chunk,
    _merge_responses,
    _select_audio_stream,
    _submit_chunk,
    plan_chunks,
    transcribe_openai,
)


class FakeTranscriptions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(model_dump=lambda mode="json": response)


def _client(responses):
    transcriptions = FakeTranscriptions(responses)
    return SimpleNamespace(
        audio=SimpleNamespace(transcriptions=transcriptions), calls=transcriptions
    )


def _recording(tmp_path, content=b"original recording"):
    recording = tmp_path / "meeting.mka"
    recording.write_bytes(content)
    return recording


def _fake_extract(recording, output, chunk, stream_index):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(f"wav:{chunk.index}:{stream_index}".encode())


def test_plan_chunks_covers_long_audio_with_overlap():
    chunks = plan_chunks(3601, chunk_seconds=600, overlap_seconds=2)

    assert chunks[0] == Chunk(index=0, start=0.0, duration=600)
    assert chunks[1] == Chunk(index=1, start=598.0, duration=600)
    assert chunks[-1].end == 3601
    assert all(chunk.duration <= 600 for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_seconds", "overlap_seconds"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_plan_chunks_rejects_invalid_boundaries(chunk_seconds, overlap_seconds):
    with pytest.raises(ValueError, match="Chunk duration"):
        plan_chunks(60, chunk_seconds, overlap_seconds)


def test_select_audio_stream_prefers_mixed_then_default():
    manifest = {
        "media": {
            "streams": [
                {"index": 1, "codec_type": "audio", "default": True},
                {"index": 2, "codec_type": "audio", "source_role": "mixed"},
            ]
        }
    }
    assert _select_audio_stream(manifest) == 2
    manifest["media"]["streams"][1].pop("source_role")
    assert _select_audio_stream(manifest) == 1


def test_boundary_dedup_and_artifact_filtering():
    assert (
        _dedupe_boundary("we discussed the launch plan", "the launch plan and owners")
        == "and owners"
    )
    segments = _merge_responses(
        [
            (Chunk(0, 0, 10), {"text": "we discussed the launch plan"}),
            (Chunk(1, 8, 10), {"text": "the launch plan and owners"}),
            (Chunk(2, 16, 2), {"text": "[Music]"}),
        ]
    )
    assert [segment["text"] for segment in segments] == [
        "we discussed the launch plan",
        "and owners",
    ]
    assert segments[1]["start"] == 10
    assert [segment["id"] for segment in segments] == [
        "segment-000001",
        "segment-000002",
    ]


def test_openai_pipeline_writes_outputs_and_resumes_without_rebilling(tmp_path):
    recording = _recording(tmp_path)
    client = _client(
        [
            {"text": "alpha boundary words repeat"},
            {"text": "boundary words repeat next section"},
            {"text": "[Silence]"},
        ]
    )

    with (
        patch("murmur.cloud_transcribe._media_duration", return_value=1201),
        patch("murmur.cloud_transcribe._extract_chunk", side_effect=_fake_extract),
    ):
        first = transcribe_openai(recording, client=client)
        second = transcribe_openai(recording, client=client)

    store = ArtifactStore(recording)
    assert first == second
    assert client.calls.calls == 3
    assert [segment["text"] for segment in first["segments"]] == [
        "alpha boundary words repeat",
        "next section",
    ]
    assert store.path("transcript.raw.json").exists()
    assert json.loads(store.path("transcript.json").read_text()) == first
    assert store.path("transcript.md").read_text().startswith("# Transcript\n")
    assert "00:00:00,000 --> 00:10:00,000" in store.path("transcript.srt").read_text()
    assert len(list(store.path("raw-responses").glob("chunk-*.json"))) == 3
    assert store.jobs()["jobs"]["transcribe:openai"]["status"] == "complete"
    assert recording.read_bytes() == b"original recording"


def test_interrupted_pipeline_resumes_only_incomplete_chunks(tmp_path):
    recording = _recording(tmp_path)
    failing = RuntimeError("temporary provider failure")
    first_client = _client([{"text": "first chunk"}, failing])

    with (
        patch("murmur.cloud_transcribe._media_duration", return_value=900),
        patch("murmur.cloud_transcribe._extract_chunk", side_effect=_fake_extract),
        pytest.raises(TranscriptionProviderError),
    ):
        transcribe_openai(recording, client=first_client)

    store = ArtifactStore(recording)
    job = store.jobs()["jobs"]["transcribe:openai"]
    assert job["status"] == "failed"
    assert job["units"]["chunk-0000"]["status"] == "complete"
    assert job["units"]["chunk-0001"]["status"] == "failed"
    error_path = store.path("raw-responses/chunk-0001.error.json")
    assert error_path.exists()
    assert json.loads(error_path.read_text())["retryable"] is True

    resumed_client = _client([{"text": "second chunk"}])
    with (
        patch("murmur.cloud_transcribe._media_duration", return_value=900),
        patch("murmur.cloud_transcribe._extract_chunk", side_effect=_fake_extract),
    ):
        result = transcribe_openai(recording, client=resumed_client)

    assert resumed_client.calls.calls == 1
    assert result["text"] == "first chunk second chunk"
    assert store.jobs()["jobs"]["transcribe:openai"]["attempt_count"] == 2


def test_corrupt_completed_raw_response_is_resubmitted(tmp_path):
    recording = _recording(tmp_path)
    initial_client = _client([{"text": "one"}, {"text": "two"}])
    with (
        patch("murmur.cloud_transcribe._media_duration", return_value=900),
        patch("murmur.cloud_transcribe._extract_chunk", side_effect=_fake_extract),
    ):
        transcribe_openai(recording, client=initial_client)

    store = ArtifactStore(recording)
    store.path("transcript.json").write_text("corrupt final")
    store.path("raw-responses/chunk-0001.json").write_text("corrupt raw")
    repair_client = _client([{"text": "two repaired"}])
    with (
        patch("murmur.cloud_transcribe._media_duration", return_value=900),
        patch("murmur.cloud_transcribe._extract_chunk", side_effect=_fake_extract),
    ):
        repaired = transcribe_openai(recording, client=repair_client)

    assert repair_client.calls.calls == 1
    assert repaired["text"] == "one two repaired"


def test_changed_model_invalidates_completed_resume(tmp_path):
    recording = _recording(tmp_path)
    first_client = _client([{"text": "old model"}])
    with (
        patch("murmur.cloud_transcribe._media_duration", return_value=30),
        patch("murmur.cloud_transcribe._extract_chunk", side_effect=_fake_extract),
    ):
        transcribe_openai(recording, model="first-model", client=first_client)

    second_client = _client([{"text": "new model"}])
    with (
        patch("murmur.cloud_transcribe._media_duration", return_value=30),
        patch("murmur.cloud_transcribe._extract_chunk", side_effect=_fake_extract),
    ):
        result = transcribe_openai(recording, model="second-model", client=second_client)

    assert second_client.calls.calls == 1
    assert result["model"] == "second-model"
    assert result["text"] == "new model"


def test_provider_error_is_sanitized_and_marked_non_retryable(tmp_path):
    recording = _recording(tmp_path)

    class BadRequest(Exception):
        status_code = 400
        request_id = "request-123"

    client = _client([BadRequest("bad key sk-secretvalue123")])
    with (
        patch("murmur.cloud_transcribe._media_duration", return_value=30),
        patch("murmur.cloud_transcribe._extract_chunk", side_effect=_fake_extract),
        pytest.raises(TranscriptionProviderError),
    ):
        transcribe_openai(recording, client=client)

    error_path = ArtifactStore(recording).path("raw-responses/chunk-0000.error.json")
    persisted = error_path.read_text()
    assert "secretvalue" not in persisted
    assert json.loads(persisted)["retryable"] is False
    job = ArtifactStore(recording).jobs()["jobs"]["transcribe:openai"]
    assert job["retryable"] is False


def test_submit_chunk_uses_openai_multipart_contract(tmp_path):
    import httpx
    from openai import OpenAI

    chunk = tmp_path / "chunk.wav"
    chunk.write_bytes(b"RIFF fake wav")

    def handler(request):
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer test-api-key"
        body = request.read()
        assert b'name="model"' in body
        assert b"gpt-4o-transcribe" in body
        assert b'name="response_format"' in body
        assert b"json" in body
        assert b'name="language"' in body
        assert b"en" in body
        assert b'name="prompt"' in body
        assert b"Sujata Abby Rohan Jatan" in body
        assert b'name="file"; filename="chunk.wav"' in body
        return httpx.Response(200, json={"text": "hello"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAI(
        api_key="test-api-key", base_url="https://api.openai.test/v1", http_client=http_client
    )

    result = _submit_chunk(
        client,
        chunk,
        model="gpt-4o-transcribe",
        language="en",
        prompt="Sujata Abby Rohan Jatan",
    )

    assert result["text"] == "hello"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg integration tools are unavailable",
)
def test_extract_chunk_produces_16khz_mono_lossless_wav(tmp_path):
    recording = tmp_path / "source.wav"
    subprocess.run(  # noqa: S603
        [
            str(shutil.which("ffmpeg")),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            str(recording),
        ],
        check=True,
    )
    output = tmp_path / "chunk.wav"

    _extract_chunk(recording, output, Chunk(0, 0.5, 1.0), stream_index=0)

    probe = subprocess.run(  # noqa: S603
        [
            str(shutil.which("ffprobe")),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream == {"codec_name": "pcm_s16le", "sample_rate": "16000", "channels": 1}
