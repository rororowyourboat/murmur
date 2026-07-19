"""Tests for plugin registration and discovery."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import click
from click.testing import CliRunner

from murmur.artifacts import ArtifactStore
from murmur.plugins import diarize, summarize, tasks, transcribe, tui
from murmur.plugins.transcribe import _format_srt_time

runner = CliRunner()


def _make_group() -> click.Group:
    @click.group()
    def grp():
        pass

    return grp


def test_transcribe_registers_command():
    grp = _make_group()
    transcribe.register(grp)
    assert "transcribe" in [c for c in grp.commands]
    param_names = [parameter.name for parameter in grp.commands["transcribe"].params]
    assert {"provider", "resume", "chunk_seconds", "overlap_seconds"} <= set(param_names)


def test_summarize_registers_command():
    grp = _make_group()
    summarize.register(grp)
    assert "summarize" in [c for c in grp.commands]


def test_diarize_registers_command():
    grp = _make_group()
    diarize.register(grp)
    assert "diarize" in [c for c in grp.commands]
    assert "speakers" in [c for c in grp.commands]


def test_tui_registers_command():
    grp = _make_group()
    tui.register(grp)
    assert "tui" in [c for c in grp.commands]


def test_task_ingest_is_preview_first_with_explicit_approval():
    grp = _make_group()
    tasks.register(grp)
    ingest = grp.commands["tasks"].commands["ingest"]
    param_names = {parameter.name for parameter in ingest.params}
    assert "approve" in param_names
    assert "dry_run" not in param_names


def test_transcribe_missing_dep():
    grp = _make_group()
    transcribe.register(grp)

    with (
        patch.dict("sys.modules", {"faster_whisper": None}),
        patch.object(transcribe, "_check_dep", return_value=False),
    ):
        result = runner.invoke(grp, ["transcribe", __file__])

    assert result.exit_code != 0


def test_diarize_missing_dep():
    grp = _make_group()
    diarize.register(grp)

    with patch.object(diarize, "_check_dep", return_value=False):
        result = runner.invoke(grp, ["diarize", __file__])

    assert result.exit_code != 0


def test_diarize_does_not_offer_unsafe_cluster_order_name_mapping():
    grp = _make_group()
    diarize.register(grp)
    cmd = grp.commands["diarize"]
    param_names = [p.name for p in cmd.params]
    assert "speakers" not in param_names


def test_format_srt_time():
    assert _format_srt_time(0.0) == "00:00:00,000"
    assert _format_srt_time(1.5) == "00:00:01,500"
    assert _format_srt_time(65.123) == "00:01:05,123"
    assert _format_srt_time(3661.0) == "01:01:01,000"


def test_summarize_finds_transcript_from_audio(tmp_path):
    """summarize should auto-find .txt when given an audio file."""
    audio = tmp_path / "meeting.flac"
    audio.write_bytes(b"fake")
    transcript = tmp_path / "meeting.txt"
    transcript.write_text("Some transcript content")

    result = summarize._find_transcript(audio)
    assert result == transcript


def test_summarize_uses_txt_directly(tmp_path):
    txt = tmp_path / "meeting.txt"
    txt.write_text("content")
    assert summarize._find_transcript(txt) == txt


def test_summarize_missing_transcript(tmp_path):
    audio = tmp_path / "meeting.flac"
    audio.write_bytes(b"fake")

    import pytest

    with pytest.raises(SystemExit):
        summarize._find_transcript(audio)


def test_local_transcription_persists_artifacts_and_resumes(tmp_path):
    audio = tmp_path / "meeting.flac"
    audio.write_bytes(b"audio")
    segments = [
        SimpleNamespace(start=0.0, end=1.5, text=" Hello"),
        SimpleNamespace(start=1.5, end=3.0, text=" world"),
    ]

    class FakeWhisperModel:
        calls = 0

        def __init__(self, model, compute_type):
            self.model = model
            self.compute_type = compute_type

        def transcribe(self, file, language):
            type(self).calls += 1
            return iter(segments), SimpleNamespace(language=language)

    fake_module = ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    with (
        patch.dict(sys.modules, {"faster_whisper": fake_module}),
        patch.object(transcribe.hooks, "emit"),
    ):
        transcribe._transcribe_file(str(audio), "base", "en")
        transcribe._transcribe_file(str(audio), "base", "en")

    artifact_dir = tmp_path / "artifacts" / "meeting"
    assert (artifact_dir / "transcript.txt").read_text().endswith("world\n")
    assert (artifact_dir / "transcript.srt").read_text().startswith("1\n00:00:00,000")
    assert (artifact_dir / "raw-responses/transcribe-faster-whisper.json").exists()
    jobs = json.loads((artifact_dir / "jobs.json").read_text())["jobs"]
    assert jobs["transcribe:faster-whisper"]["status"] == "complete"
    assert FakeWhisperModel.calls == 1


def test_transcribe_openai_provider_dispatches_cloud_pipeline(tmp_path):
    audio = tmp_path / "meeting.flac"
    audio.write_bytes(b"audio")
    grp = _make_group()
    transcribe.register(grp)

    with (
        patch.object(transcribe, "transcribe_openai", return_value={"segments": []}) as cloud,
        patch.object(transcribe.hooks, "emit"),
    ):
        result = runner.invoke(
            grp,
            [
                "transcribe",
                str(audio),
                "--provider",
                "openai",
                "--model",
                "gpt-4o-transcribe",
                "--chunk-seconds",
                "300",
                "--overlap-seconds",
                "1",
                "--prompt",
                "project glossary",
            ],
        )

    assert result.exit_code == 0
    cloud.assert_called_once_with(
        str(audio),
        model="gpt-4o-transcribe",
        language="en",
        prompt="project glossary",
        chunk_seconds=300.0,
        overlap_seconds=1.0,
        resume=True,
    )


def test_transcribe_diarize_dispatches_channel_aware_pipeline(tmp_path):
    audio = tmp_path / "meeting.mka"
    audio.write_bytes(b"audio")
    grp = _make_group()
    transcribe.register(grp)

    with (
        patch.object(
            transcribe, "transcribe_openai_diarized", return_value={"segments": []}
        ) as cloud,
        patch.object(transcribe.hooks, "emit"),
    ):
        result = runner.invoke(
            grp,
            [
                "transcribe",
                str(audio),
                "--diarize",
                "--speaker-profile",
                "team",
            ],
        )

    assert result.exit_code == 0
    cloud.assert_called_once_with(
        str(audio),
        profile_name="team",
        model="gpt-4o-transcribe-diarize",
        language="en",
        chunk_seconds=600.0,
        overlap_seconds=2.0,
        resume=True,
    )


def test_diarization_persists_canonical_outputs(tmp_path):
    audio = tmp_path / "meeting.flac"
    audio.write_bytes(b"audio")

    class FakeDiarization:
        def write_rttm(self, handle):
            handle.write("SPEAKER meeting 1 0.000 1.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n")

        def itertracks(self, yield_label=False):
            yield SimpleNamespace(start=0.0, end=1.0), None, "SPEAKER_00"

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model, use_auth_token):
            return cls()

        def __call__(self, file):
            return FakeDiarization()

    pyannote = ModuleType("pyannote")
    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = FakePipeline
    pyannote.audio = pyannote_audio

    with patch.dict(sys.modules, {"pyannote": pyannote, "pyannote.audio": pyannote_audio}):
        rttm, timeline, speakers = diarize._diarize_file(str(audio), "hf_never_persist")

    assert rttm.parent.name == "speakers"
    assert "SPEAKER_00" in rttm.read_text()
    assert "SPEAKER_00" in timeline.read_text()
    assert speakers == {"SPEAKER_00"}
    assert "hf_never_persist" not in (tmp_path / "artifacts/meeting/jobs.json").read_text()


def test_summary_persists_and_skips_valid_completed_output(tmp_path):
    audio = tmp_path / "meeting.flac"
    audio.write_bytes(b"audio")
    store = ArtifactStore(audio)
    store.ensure_manifest()
    transcript_path = store.write_text("transcript.txt", "A useful meeting transcript")
    store.register_artifact("transcript_text", transcript_path, kind="transcript")

    candidate = {
        "title": "Useful meeting",
        "attendees": [],
        "executive_summary": [
            {"text": "The meeting was useful.", "segment_ids": ["legacy-000001"]}
        ],
        "topics": [],
        "decisions": [],
        "open_questions": [],
        "action_items": [],
    }
    with patch.object(summarize, "_llm_generate", return_value=candidate) as generate:
        first = summarize._summarize_file(transcript_path, "test/model")
        second = summarize._summarize_file(transcript_path, "test/model")

    assert first == second == store.path("summary.md")
    assert first.read_text().startswith("# Useful meeting")
    generate.assert_called_once()
    assert store.jobs()["jobs"]["summarize:litellm"]["status"] == "complete"
    assert store.path("summary.json").is_file()
    assert store.path("transcript.cleaned.json").is_file()
    assert transcript_path.read_text() == "A useful meeting transcript"
    metadata = json.loads(store.path("summary.json").read_text())["metadata"]
    assert metadata["model"] == "test/model"
    assert metadata["prompt_version"] == 2
    assert metadata["source_sha256"]
    assert metadata["generated_at"]
