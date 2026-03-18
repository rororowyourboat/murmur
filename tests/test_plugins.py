"""Tests for plugin registration and discovery."""

from unittest.mock import patch

import click
from click.testing import CliRunner

from murmur.plugins import diarize, summarize, transcribe, tui
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


def test_summarize_registers_command():
    grp = _make_group()
    summarize.register(grp)
    assert "summarize" in [c for c in grp.commands]


def test_diarize_registers_command():
    grp = _make_group()
    diarize.register(grp)
    assert "diarize" in [c for c in grp.commands]


def test_tui_registers_command():
    grp = _make_group()
    tui.register(grp)
    assert "tui" in [c for c in grp.commands]


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


def test_diarize_has_speakers_option():
    grp = _make_group()
    diarize.register(grp)
    cmd = grp.commands["diarize"]
    param_names = [p.name for p in cmd.params]
    assert "speakers" in param_names


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
