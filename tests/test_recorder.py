"""Tests for recorder module — PipeWire parsing, FFmpeg cmd building, helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_build_ffmpeg_cmd_dual_channel():
    cmd = recorder.build_ffmpeg_cmd(
        Path("/tmp/out.flac"),
        "alsa_output.test",
        "flac",
        mic_source="alsa_input.mic",
    )
    # Should have two -i inputs
    i_indices = [i for i, v in enumerate(cmd) if v == "-i"]
    assert len(i_indices) == 2
    assert cmd[i_indices[0] + 1] == "alsa_output.test.monitor"
    assert cmd[i_indices[1] + 1] == "alsa_input.mic"
    # Should use amerge filter to mix into stereo
    assert "-filter_complex" in cmd
    assert any("amerge" in v for v in cmd)
    assert "-map" in cmd
    assert "[out]" in cmd


def test_build_ffmpeg_cmd_dual_channel_codec():
    cmd = recorder.build_ffmpeg_cmd(
        Path("/tmp/out.flac"),
        "alsa_output.test",
        "flac",
        mic_source="alsa_input.mic",
    )
    assert "-c:a" in cmd
    assert "flac" in cmd


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


def test_is_recording_no_pid_file(tmp_path):
    with patch.object(recorder, "PID_FILE", tmp_path / "nope.pid"):
        assert recorder.is_recording() is None


def test_is_recording_stale_pid(tmp_path):
    pid_file = tmp_path / "murmur.pid"
    pid_file.write_text("999999999")
    with patch.object(recorder, "PID_FILE", pid_file):
        assert recorder.is_recording() is None
    assert not pid_file.exists()
