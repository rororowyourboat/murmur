"""Tests for canonical artifacts and durable processing jobs."""

import json
from unittest.mock import patch

import pytest

from murmur import artifacts
from murmur.artifacts import ArtifactStore, CorruptStateError


def _recording(tmp_path, name="meeting.mka", content=b"recording"):
    path = tmp_path / "recordings" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_ensure_manifest_creates_canonical_layout(tmp_path):
    recording = _recording(tmp_path)
    store = ArtifactStore(recording, root=tmp_path / "artifacts")

    manifest = store.ensure_manifest(
        {
            "format": "mka",
            "duration_secs": 12.5,
            "streams": [{"index": 0, "title": "Mixed call"}],
        }
    )

    assert store.directory == tmp_path / "artifacts" / "meeting"
    assert store.manifest_path.exists()
    assert store.jobs_path.exists()
    assert manifest["recording"] == str(recording.resolve())
    assert manifest["source_fingerprint"]["algorithm"] == "sha256"
    assert manifest["media"]["duration_secs"] == 12.5
    assert manifest["media"]["streams"][0]["title"] == "Mixed call"
    assert json.loads(store.jobs_path.read_text())["jobs"] == {}


def test_manifest_reuses_fingerprint_for_unchanged_source(tmp_path):
    recording = _recording(tmp_path)
    store = ArtifactStore(recording, root=tmp_path / "artifacts")
    first = store.ensure_manifest()

    with patch.object(artifacts, "fingerprint_file", wraps=artifacts.fingerprint_file) as digest:
        second = store.ensure_manifest()

    digest.assert_not_called()
    assert second["source_fingerprint"] == first["source_fingerprint"]


def test_register_artifact_tracks_checksum_and_sanitizes_provenance(tmp_path):
    recording = _recording(tmp_path)
    store = ArtifactStore(recording, root=tmp_path / "artifacts")
    store.ensure_manifest()
    transcript = store.write_text("transcript.md", "hello")

    entry = store.register_artifact(
        "transcript_markdown",
        transcript,
        kind="transcript",
        provenance={
            "provider": "openai",
            "api_key": "sk-secretvalue123",
            "reference": "data:audio/wav;base64,secret",
            "message": "Bearer secret-token-value",
        },
    )

    persisted = store.manifest_path.read_text()
    assert entry["path"] == "transcript.md"
    assert store.artifact_valid("transcript_markdown")
    assert "sk-secretvalue123" not in persisted
    assert "base64,secret" not in persisted
    assert "secret-token-value" not in persisted


def test_completed_job_skips_valid_output_and_resumes_corrupt_output(tmp_path):
    recording = _recording(tmp_path)
    store = ArtifactStore(recording, root=tmp_path / "artifacts")
    output = store.write_text("transcript.md", "hello")
    store.register_artifact("transcript_markdown", output, kind="transcript")
    store.begin_job("transcribe", "openai", output_artifacts=["transcript_markdown"])
    store.complete_job("transcribe", "openai")

    completed, should_run = store.begin_job(
        "transcribe", "openai", output_artifacts=["transcript_markdown"]
    )
    assert should_run is False
    assert completed["attempt_count"] == 1

    output.write_text("changed")
    resumed, should_run = store.begin_job(
        "transcribe", "openai", output_artifacts=["transcript_markdown"]
    )
    assert should_run is True
    assert resumed["attempt_count"] == 2


def test_source_change_invalidates_completed_job(tmp_path):
    recording = _recording(tmp_path, content=b"first")
    store = ArtifactStore(recording, root=tmp_path / "artifacts")
    output = store.write_text("summary.md", "summary")
    store.register_artifact("summary", output, kind="summary")
    store.begin_job(
        "summarize",
        "test",
        output_paths=[output],
        output_artifacts=["summary"],
    )
    store.complete_job("summarize", "test")

    recording.write_bytes(b"different source")
    job, should_run = store.begin_job(
        "summarize", "test", output_paths=[output], output_artifacts=["summary"]
    )

    assert should_run is True
    assert job["attempt_count"] == 2
    assert job["output_paths"] == [str(output)]


def test_failed_job_survives_restart_and_can_be_retried(tmp_path):
    recording = _recording(tmp_path)
    root = tmp_path / "artifacts"
    store = ArtifactStore(recording, root=root)
    store.begin_job("transcribe", "openai", parameters={"token": "never-store-this"})
    store.fail_job(
        "transcribe",
        "openai",
        "request failed with sk-secretvalue123",
        retryable=True,
    )

    restarted = ArtifactStore(recording, root=root)
    failed = restarted.jobs()["jobs"]["transcribe:openai"]
    assert failed["status"] == "failed"
    assert failed["attempt_count"] == 1
    assert "secretvalue" not in restarted.jobs_path.read_text()
    assert restarted.retry_failed() == ["transcribe:openai"]
    assert restarted.jobs()["jobs"]["transcribe:openai"]["status"] == "pending"


def test_completed_units_survive_restart_and_are_not_resubmitted(tmp_path):
    recording = _recording(tmp_path)
    root = tmp_path / "artifacts"
    store = ArtifactStore(recording, root=root)
    store.begin_job("transcribe", "openai")
    raw = store.write_json("raw-responses/chunk-0000.json", {"text": "done"})
    store.register_artifact("chunk_0000_raw", raw, kind="raw_provider_response")
    store.begin_unit(
        "transcribe",
        "openai",
        "chunk-0000",
        output_artifacts=["chunk_0000_raw"],
    )
    store.complete_unit("transcribe", "openai", "chunk-0000")

    restarted = ArtifactStore(recording, root=root)
    unit, should_submit = restarted.begin_unit(
        "transcribe",
        "openai",
        "chunk-0000",
        output_artifacts=["chunk_0000_raw"],
    )

    assert should_submit is False
    assert unit["status"] == "complete"
    assert unit["attempt_count"] == 1


def test_running_job_state_survives_process_restart(tmp_path):
    recording = _recording(tmp_path)
    root = tmp_path / "artifacts"
    ArtifactStore(recording, root=root).begin_job("transcribe", "openai")

    restarted = ArtifactStore(recording, root=root)

    assert restarted.jobs()["jobs"]["transcribe:openai"]["status"] == "running"


def test_atomic_write_preserves_previous_state_on_replace_failure(tmp_path):
    target = tmp_path / "state.json"
    artifacts._atomic_write_json(target, {"state": "old"})

    with (
        patch.object(artifacts.os, "replace", side_effect=OSError("interrupted")),
        pytest.raises(OSError, match="interrupted"),
    ):
        artifacts._atomic_write_json(target, {"state": "new"})

    assert json.loads(target.read_text()) == {"state": "old"}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_corrupted_jobs_file_is_reported(tmp_path):
    recording = _recording(tmp_path)
    store = ArtifactStore(recording, root=tmp_path / "artifacts")
    store.ensure_manifest()
    store.jobs_path.write_text("not-json")

    with pytest.raises(CorruptStateError, match="Could not read"):
        store.jobs()


def test_artifact_paths_cannot_escape_store(tmp_path):
    recording = _recording(tmp_path)
    store = ArtifactStore(recording, root=tmp_path / "artifacts")

    with pytest.raises(ValueError, match="inside"):
        store.path("../escape.json")


def test_same_stem_recordings_get_distinct_stores(tmp_path):
    first = _recording(tmp_path / "one")
    second = _recording(tmp_path / "two")
    root = tmp_path / "artifacts"
    first_store = ArtifactStore(first, root=root)
    first_store.ensure_manifest()
    second_store = ArtifactStore(second, root=root)
    second_store.ensure_manifest()

    assert first_store.directory != second_store.directory
    assert second_store.recording_id.startswith("meeting-")
