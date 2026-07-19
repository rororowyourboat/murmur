"""Tests for CLI commands using Click's test runner."""

import json
from unittest.mock import patch

from click.testing import CliRunner

from murmur.artifacts import ArtifactStore
from murmur.cli import cli

runner = CliRunner()


def test_help():
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Murmur" in result.output
    assert "start" in result.output
    assert "devices" in result.output


def test_version():
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_devices_no_sinks_no_sources():
    with (
        patch("murmur.cli.get_pipewire_sinks", return_value=[]),
        patch("murmur.cli.get_pipewire_sources", return_value=[]),
    ):
        result = runner.invoke(cli, ["devices"])

    assert result.exit_code == 0
    assert "No audio sinks" in result.output
    assert "No audio sources" in result.output


def test_devices_with_sinks_and_sources():
    sinks = [
        {"id": 55, "name": "HDMI Output", "default": False},
        {"id": 91, "name": "Speakers", "default": True},
    ]
    sources = [
        {"id": 58, "name": "Built-in Mic", "default": True},
    ]
    with (
        patch("murmur.cli.get_pipewire_sinks", return_value=sinks),
        patch("murmur.cli.get_pipewire_sources", return_value=sources),
    ):
        result = runner.invoke(cli, ["devices"])

    assert result.exit_code == 0
    assert "HDMI Output" in result.output
    assert "Speakers" in result.output
    assert "Built-in Mic" in result.output


def test_status_not_recording():
    with patch("murmur.cli.is_recording", return_value=None):
        result = runner.invoke(cli, ["status"])

    assert result.exit_code == 0
    assert "Not recording" in result.output


def test_toggle_stop_uses_shared_finalizer(tmp_path):
    output = tmp_path / "meeting.flac"
    finalized = {"status": "recorded", "output": str(output), "duration_secs": 12.5}

    with (
        patch("murmur.cli.is_recording", return_value=4321),
        patch("murmur.cli.stop_recording", return_value=finalized) as stop,
        patch("murmur.cli.notify"),
    ):
        result = runner.invoke(cli, ["toggle"])

    assert result.exit_code == 0
    assert "Stopped recording" in result.output
    assert str(output) in result.output
    stop.assert_called_once_with()


def test_list_no_directory(tmp_path):
    with patch(
        "murmur.recorder._default_output_dir",
        return_value=tmp_path / "nonexistent",
    ):
        result = runner.invoke(cli, ["list"])

    assert result.exit_code == 0
    assert "No recordings directory" in result.output


def test_list_empty_directory(tmp_path):
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    with patch("murmur.recorder._default_output_dir", return_value=recordings_dir):
        result = runner.invoke(cli, ["list"])

    assert result.exit_code == 0
    assert "No recordings found" in result.output


def test_import_copies_file(tmp_path):
    source = tmp_path / "external.flac"
    source.write_bytes(b"fake audio data")

    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()

    with patch("murmur.recorder._default_output_dir", return_value=recordings_dir):
        result = runner.invoke(cli, ["import", str(source)])

    assert result.exit_code == 0
    assert "Imported" in result.output
    assert (recordings_dir / "external.flac").exists()
    assert (recordings_dir / "external.json").exists()


def test_import_with_tag(tmp_path):
    source = tmp_path / "interview.mp3"
    source.write_bytes(b"fake audio")

    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()

    with patch("murmur.recorder._default_output_dir", return_value=recordings_dir):
        result = runner.invoke(cli, ["import", "--tag", "retro", str(source)])

    assert result.exit_code == 0
    assert "Imported" in result.output
    assert (recordings_dir / "meeting_retro_interview.mp3").exists()


def test_import_existing_file_warns(tmp_path):
    source = tmp_path / "existing.flac"
    source.write_bytes(b"fake audio")

    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    (recordings_dir / "existing.flac").write_bytes(b"already here")

    with patch("murmur.recorder._default_output_dir", return_value=recordings_dir):
        result = runner.invoke(cli, ["import", str(source)])

    assert result.exit_code == 0
    assert "already exists" in result.output


def test_start_help_shows_mic_option():
    result = runner.invoke(cli, ["start", "--help"])
    assert result.exit_code == 0
    assert "--mic" in result.output
    assert "--mic-device" in result.output


def test_start_with_mic_uses_multitrack_mka(tmp_path):
    requested = tmp_path / "call.flac"

    with (
        patch("murmur.cli.resolve_sink", return_value=(91, "alsa_output.test")),
        patch("murmur.cli.resolve_source", return_value=(58, "alsa_input.mic")),
        patch("murmur.cli.record_foreground") as record,
    ):
        result = runner.invoke(cli, ["start", "--mic", "--output", str(requested)])

    assert result.exit_code == 0
    assert "call.mka" in result.output
    assert "3 Opus streams" in result.output
    record.assert_called_once_with(
        tmp_path / "call.mka",
        "alsa_output.test",
        91,
        "flac",
        mic_source="alsa_input.mic",
        mic_id=58,
    )


def test_jobs_status_creates_and_prints_canonical_state(tmp_path):
    recording = tmp_path / "meeting.flac"
    recording.write_bytes(b"audio")

    result = runner.invoke(cli, ["jobs", "status", str(recording), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["manifest"]["recording"] == str(recording.resolve())
    assert payload["jobs"] == {}
    assert (tmp_path / "artifacts" / "meeting" / "manifest.json").exists()


def test_jobs_retry_resets_failed_job(tmp_path):
    recording = tmp_path / "meeting.flac"
    recording.write_bytes(b"audio")
    store = ArtifactStore(recording)
    store.begin_job("transcribe", "openai")
    store.fail_job("transcribe", "openai", "temporary failure")

    result = runner.invoke(cli, ["jobs", "retry", str(recording), "--job", "transcribe"])

    assert result.exit_code == 0
    assert "transcribe:openai" in result.output
    assert store.jobs()["jobs"]["transcribe:openai"]["status"] == "pending"
