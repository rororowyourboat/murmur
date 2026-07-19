"""Tests for recorder module — PipeWire parsing, FFmpeg cmd building, helpers."""

import json
import signal
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from murmur import recorder

WPCTL_STATUS_OUTPUT = """\
PipeWire 'pipewire-0' [1.2.7, rohan@zenbook, cookie:123]
 \u2514\u2500 Clients:
        31. pipewire                            [1.2.7, rohan@zenbook, pid:1234]

Audio
 \u251c\u2500 Devices:
 \u2502      44. Meteor Lake-P HD Audio Controller   [alsa]
 \u2502
 \u251c\u2500 Sinks:
 \u2502      55. Meteor Lake-P HDMI 3                [vol: 0.40]
 \u2502  *   91. BRAVIA Theatre U                    [vol: 0.97]
 \u2502
 \u251c\u2500 Sources:
 \u2502  *   58. Meteor Lake-P Mic                   [vol: 1.00]
 \u2502      59. External USB Mic                    [vol: 0.80]
 \u2502
 \u2514\u2500 Streams:

Video
 \u2514\u2500 Streams:
"""


WPCTL_INSPECT_OUTPUT = """\
  id 91, type PipeWire:Interface:Node
    media.class = "Audio/Sink"
    node.name = "alsa_output.usb-Sony_BRAVIA"
    node.nick = "BRAVIA Theatre U"
"""

WPCTL_INSPECT_SOURCE = """\
  id 58, type PipeWire:Interface:Node
    media.class = "Audio/Source"
    node.name = "alsa_input.pci-0000_00_1f.3-mic"
    node.nick = "Meteor Lake-P Mic"
"""


def test_get_pipewire_sinks():
    mock_result = MagicMock(returncode=0, stdout=WPCTL_STATUS_OUTPUT)
    with patch("murmur.recorder.subprocess.run", return_value=mock_result):
        sinks = recorder.get_pipewire_sinks()

    assert len(sinks) == 2
    assert sinks[0] == {"id": 55, "name": "Meteor Lake-P HDMI 3", "default": False}
    assert sinks[1] == {"id": 91, "name": "BRAVIA Theatre U", "default": True}


def test_get_pipewire_sources():
    mock_result = MagicMock(returncode=0, stdout=WPCTL_STATUS_OUTPUT)
    with patch("murmur.recorder.subprocess.run", return_value=mock_result):
        sources = recorder.get_pipewire_sources()

    assert len(sources) == 2
    assert sources[0] == {"id": 58, "name": "Meteor Lake-P Mic", "default": True}
    assert sources[1] == {"id": 59, "name": "External USB Mic", "default": False}


def test_get_pipewire_sinks_empty():
    output = "PipeWire 'pipewire-0'\n \u2514\u2500 Clients:\n\nAudio\n \u251c\u2500 Sinks:\n\n"
    mock_result = MagicMock(returncode=0, stdout=output)
    with patch("murmur.recorder.subprocess.run", return_value=mock_result):
        sinks = recorder.get_pipewire_sinks()

    assert sinks == []


def test_get_default_sink_id():
    mock_result = MagicMock(returncode=0, stdout=WPCTL_STATUS_OUTPUT)
    with patch("murmur.recorder.subprocess.run", return_value=mock_result):
        sink_id = recorder.get_default_sink_id()

    assert sink_id == 91


def test_get_default_source_id():
    mock_result = MagicMock(returncode=0, stdout=WPCTL_STATUS_OUTPUT)
    with patch("murmur.recorder.subprocess.run", return_value=mock_result):
        source_id = recorder.get_default_source_id()

    assert source_id == 58


def test_get_default_sink_id_no_default():
    output = (
        "Audio\n \u251c\u2500 Sinks:\n"
        " \u2502      55. Some Sink [vol: 1.0]\n \u2502\n \u2514\u2500 Sources:\n"
    )
    mock_result = MagicMock(returncode=0, stdout=output)
    with patch("murmur.recorder.subprocess.run", return_value=mock_result):
        sink_id = recorder.get_default_sink_id()

    assert sink_id == 55


def test_get_node_name():
    mock_result = MagicMock(returncode=0, stdout=WPCTL_INSPECT_OUTPUT)
    with patch("murmur.recorder.subprocess.run", return_value=mock_result):
        name = recorder.get_node_name(91)

    assert name == "alsa_output.usb-Sony_BRAVIA"


def test_get_node_name_source():
    mock_result = MagicMock(returncode=0, stdout=WPCTL_INSPECT_SOURCE)
    with patch("murmur.recorder.subprocess.run", return_value=mock_result):
        name = recorder.get_node_name(58)

    assert name == "alsa_input.pci-0000_00_1f.3-mic"


def test_get_node_name_failure():
    mock_result = MagicMock(returncode=1)
    with patch("murmur.recorder.subprocess.run", return_value=mock_result):
        name = recorder.get_node_name(999)

    assert name is None


def test_build_ffmpeg_cmd_flac():
    cmd = recorder.build_ffmpeg_cmd(Path("/tmp/out.flac"), "alsa_output.test", "flac")
    assert cmd == [
        "ffmpeg",
        "-y",
        "-thread_queue_size",
        "1024",
        "-f",
        "pulse",
        "-i",
        "alsa_output.test.monitor",
        "-c:a",
        "flac",
        "/tmp/out.flac",
    ]


def test_build_ffmpeg_cmd_mp3():
    cmd = recorder.build_ffmpeg_cmd(Path("/tmp/out.mp3"), "alsa_output.test", "mp3")
    assert "-c:a" in cmd
    assert "libmp3lame" in cmd
    assert "-q:a" in cmd


def test_build_ffmpeg_cmd_wav():
    cmd = recorder.build_ffmpeg_cmd(Path("/tmp/out.wav"), "alsa_output.test", "wav")
    assert "pcm_s16le" in cmd


def test_build_ffmpeg_cmd_ogg():
    cmd = recorder.build_ffmpeg_cmd(Path("/tmp/out.ogg"), "alsa_output.test", "ogg")
    assert "libvorbis" in cmd


def test_build_ffmpeg_cmd_multitrack_mka():
    cmd = recorder.build_ffmpeg_cmd(
        Path("/tmp/out.mka"),
        "alsa_output.test",
        "flac",
        mic_source="alsa_input.mic",
    )
    assert cmd == [
        "ffmpeg",
        "-y",
        "-thread_queue_size",
        "1024",
        "-f",
        "pulse",
        "-i",
        "alsa_output.test.monitor",
        "-thread_queue_size",
        "1024",
        "-f",
        "pulse",
        "-i",
        "alsa_input.mic",
        "-filter_complex",
        "[0:a]aresample=async=1000:first_pts=0,asplit=2[call_mix][call_track];"
        "[1:a]aresample=async=1000:first_pts=0,asplit=2[mic_mix][mic_track];"
        "[call_mix][mic_mix]amix=inputs=2:duration=longest:dropout_transition=0:"
        "normalize=0,alimiter=limit=0.95:level=false[mix]",
        "-map",
        "[mix]",
        "-map",
        "[mic_track]",
        "-map",
        "[call_track]",
        "-c:a",
        "libopus",
        "-b:a:0",
        "128k",
        "-b:a:1",
        "96k",
        "-b:a:2",
        "96k",
        "-metadata:s:a:0",
        "title=Mixed call",
        "-metadata:s:a:1",
        "title=Microphone",
        "-metadata:s:a:2",
        "title=Call output",
        "-disposition:a:0",
        "default",
        "-disposition:a:1",
        "0",
        "-disposition:a:2",
        "0",
        "-f",
        "matroska",
        "/tmp/out.mka",
    ]


def test_build_ffmpeg_cmd_multitrack_requires_mka():
    with pytest.raises(ValueError, match=r"requires an \.mka"):
        recorder.build_ffmpeg_cmd(
            Path("/tmp/out.flac"),
            "alsa_output.test",
            "flac",
            mic_source="alsa_input.mic",
        )


def test_make_output_path_explicit():
    path = recorder.make_output_path("/tmp/my_recording.flac", "flac", None)
    assert path == Path("/tmp/my_recording.flac")


def test_make_output_path_generated(tmp_path):
    with patch.object(recorder, "_default_output_dir", return_value=tmp_path):
        path = recorder.make_output_path(None, "flac", None)

    assert path.parent == tmp_path
    assert path.suffix == ".flac"
    assert path.name.startswith("meeting_")


def test_make_output_path_with_tag(tmp_path):
    with patch.object(recorder, "_default_output_dir", return_value=tmp_path):
        path = recorder.make_output_path(None, "mp3", "standup")

    assert "_standup_" in path.name
    assert path.suffix == ".mp3"


def test_make_output_path_multitrack_forces_mka(tmp_path):
    with patch.object(recorder, "_default_output_dir", return_value=tmp_path):
        generated = recorder.make_output_path(None, "flac", "standup", multitrack=True)
        explicit = recorder.make_output_path("/tmp/custom.flac", "flac", None, multitrack=True)

    assert generated.suffix == ".mka"
    assert explicit == Path("/tmp/custom.mka")


def test_multitrack_metadata_records_stream_contract():
    meta = recorder._recording_metadata(
        Path("/tmp/meeting.mka"),
        "alsa_output.test",
        91,
        "flac",
        "alsa_input.mic",
        58,
    )

    assert meta["format"] == "mka"
    assert meta["requested_format"] == "flac"
    assert meta["capture_mode"] == "multitrack"
    assert meta["stream_layout"] == [
        {
            "index": 0,
            "title": "Mixed call",
            "codec": "opus",
            "source_role": "mixed",
            "default": True,
            "source_names": ["alsa_output.test.monitor", "alsa_input.mic"],
        },
        {
            "index": 1,
            "title": "Microphone",
            "codec": "opus",
            "source_role": "microphone",
            "default": False,
            "source_names": ["alsa_input.mic"],
        },
        {
            "index": 2,
            "title": "Call output",
            "codec": "opus",
            "source_role": "call_output",
            "default": False,
            "source_names": ["alsa_output.test.monitor"],
        },
    ]


def test_is_recording_no_pid_file(tmp_path):
    with patch.object(recorder, "PID_FILE", tmp_path / "nope.pid"):
        assert recorder.is_recording() is None


def test_is_recording_stale_pid(tmp_path):
    pid_file = tmp_path / "murmur.pid"
    pid_file.write_text("999999999")
    with patch.object(recorder, "PID_FILE", pid_file):
        assert recorder.is_recording() is None
    assert not pid_file.exists()


def test_is_recording_rejects_reused_pid(tmp_path):
    pid_file = tmp_path / "murmur.pid"
    pid_file.write_text("1234")
    pid_file.with_name("recording.json").write_text(
        json.dumps({"pid": 1234, "output": "/tmp/meeting.flac"})
    )

    with (
        patch.object(recorder, "PID_FILE", pid_file),
        patch.object(recorder, "_process_matches", return_value=False),
        patch.object(recorder, "_finalize_recording") as finalize,
    ):
        assert recorder.is_recording() is None

    finalize.assert_called_once()
    assert not pid_file.exists()
    assert not pid_file.with_name("recording.json").exists()


def test_finalize_recording_uses_ffprobe_metadata(tmp_path):
    output = tmp_path / "meeting.flac"
    output.write_bytes(b"audio")
    meta = {
        "status": "recording",
        "started_at": "2026-07-20T10:00:00",
        "output": str(output),
    }
    probe = {
        "format": {"duration": "12.5", "size": "5"},
        "streams": [{"index": 0, "codec_name": "flac", "codec_type": "audio"}],
    }

    with (
        patch.object(recorder, "_probe_media", return_value=probe),
        patch.object(recorder.hooks, "emit") as emit,
    ):
        finalized = recorder._finalize_recording(meta)

    assert finalized["status"] == "recorded"
    assert finalized["duration_secs"] == 12.5
    assert finalized["file_size_bytes"] == 5
    assert finalized["streams"] == probe["streams"]
    assert finalized["recording_id"] == "meeting"
    assert (tmp_path / "artifacts/meeting/manifest.json").exists()
    assert (tmp_path / "artifacts/meeting/jobs.json").exists()
    assert json.loads(output.with_suffix(".json").read_text())["status"] == "recorded"
    emit.assert_called_once_with(
        "recording_saved",
        output_path=str(output),
        meta_path=str(output.with_suffix(".json")),
        duration_secs=12.5,
    )


def test_finalize_recording_enriches_multitrack_stream_metadata(tmp_path):
    output = tmp_path / "meeting.mka"
    output.write_bytes(b"audio")
    meta = recorder._recording_metadata(
        output,
        "alsa_output.test",
        91,
        "flac",
        "alsa_input.mic",
        58,
    )
    probe = {
        "format": {"duration": "12.5", "size": "5"},
        "streams": [
            {
                "index": index,
                "codec_name": "opus",
                "codec_type": "audio",
                "tags": {"title": title},
                "disposition": {"default": int(index == 0)},
            }
            for index, title in enumerate(("Mixed call", "Microphone", "Call output"))
        ],
    }

    with (
        patch.object(recorder, "_probe_media", return_value=probe),
        patch.object(recorder.hooks, "emit"),
    ):
        finalized = recorder._finalize_recording(meta)

    assert [stream["title"] for stream in finalized["streams"]] == [
        "Mixed call",
        "Microphone",
        "Call output",
    ]
    assert [stream["source_role"] for stream in finalized["streams"]] == [
        "mixed",
        "microphone",
        "call_output",
    ]
    assert finalized["streams"][0]["default"] is True
    assert finalized["streams"][1]["source_names"] == ["alsa_input.mic"]


def test_record_background_persists_active_state(tmp_path):
    pid_file = tmp_path / "cache" / "murmur.pid"
    output = tmp_path / "meeting.flac"
    process = MagicMock(pid=4321)

    with (
        patch.object(recorder, "PID_FILE", pid_file),
        patch.object(recorder, "is_recording", return_value=None),
        patch.object(recorder.subprocess, "Popen", return_value=process),
        patch.object(recorder.hooks, "emit") as emit,
    ):
        pid = recorder.record_background(output, "sink", 91, "flac")

    assert pid == 4321
    assert pid_file.read_text().strip() == "4321"
    state = json.loads(pid_file.with_name("recording.json").read_text())
    assert state["pid"] == 4321
    assert state["output"] == str(output)
    assert json.loads(output.with_suffix(".json").read_text())["status"] == "recording"
    emit.assert_called_once_with(
        "recording_started",
        pid=4321,
        output_path=str(output),
        source="sink.monitor",
    )


def test_stop_recording_gracefully_finalizes_and_clears_state(tmp_path):
    pid_file = tmp_path / "cache" / "murmur.pid"
    pid_file.parent.mkdir()
    pid_file.write_text("4321")
    output = tmp_path / "meeting.flac"
    meta_path = output.with_suffix(".json")
    meta = {"status": "recording", "started_at": "2026-07-20T10:00:00", "output": str(output)}
    meta_path.write_text(json.dumps(meta))
    pid_file.with_name("recording.json").write_text(
        json.dumps({"pid": 4321, "output": str(output), "meta_path": str(meta_path)})
    )
    finalized = {**meta, "status": "recorded", "duration_secs": 12.5}

    with (
        patch.object(recorder, "PID_FILE", pid_file),
        patch.object(recorder, "_process_matches", return_value=True),
        patch.object(recorder, "_pid_exists", return_value=False),
        patch.object(recorder, "_finalize_recording", return_value=finalized) as finalize,
        patch.object(recorder.os, "kill") as kill,
    ):
        result = recorder.stop_recording()

    assert result == finalized
    kill.assert_called_once_with(4321, signal.SIGINT)
    finalize.assert_called_once_with(meta)
    assert not pid_file.exists()
    assert not pid_file.with_name("recording.json").exists()


def test_stop_recording_uses_sigterm_after_timeout(tmp_path):
    pid_file = tmp_path / "cache" / "murmur.pid"
    pid_file.parent.mkdir()
    pid_file.write_text("4321")
    output = tmp_path / "meeting.flac"
    meta_path = output.with_suffix(".json")
    meta = {"status": "recording", "started_at": "2026-07-20T10:00:00", "output": str(output)}
    meta_path.write_text(json.dumps(meta))
    pid_file.with_name("recording.json").write_text(
        json.dumps({"pid": 4321, "output": str(output), "meta_path": str(meta_path)})
    )

    with (
        patch.object(recorder, "PID_FILE", pid_file),
        patch.object(recorder, "_process_matches", return_value=True),
        patch.object(recorder, "_pid_exists", side_effect=[True, True, False, False]),
        patch.object(recorder, "_finalize_recording", return_value={**meta, "status": "recorded"}),
        patch.object(recorder.os, "kill") as kill,
    ):
        recorder.stop_recording(timeout=0)

    assert kill.call_args_list == [call(4321, signal.SIGINT), call(4321, signal.SIGTERM)]


def test_record_background_marks_spawn_failure(tmp_path):
    pid_file = tmp_path / "cache" / "murmur.pid"
    output = tmp_path / "meeting.flac"

    with (
        patch.object(recorder, "PID_FILE", pid_file),
        patch.object(recorder, "is_recording", return_value=None),
        patch.object(recorder.subprocess, "Popen", side_effect=OSError("ffmpeg missing")),
        pytest.raises(RuntimeError, match="Could not start FFmpeg"),
    ):
        recorder.record_background(output, "sink", 91, "flac")

    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["status"] == "failed"
    assert "ffmpeg missing" in metadata["error"]
