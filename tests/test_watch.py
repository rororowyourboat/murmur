"""Tests for the watch plugin (meeting detection)."""

import json
from unittest.mock import patch

from murmur.plugins.watch import _get_mic_streams, _is_meeting_app


class TestIsMeetingApp:
    """Tests for _is_meeting_app matching logic."""

    def test_matches_app_name(self):
        stream = {"app": "zoom", "name": "zoom-node", "id": 1}
        match, display = _is_meeting_app(stream, ["zoom"])
        assert match is True
        assert display == "Zoom"

    def test_matches_case_insensitive(self):
        stream = {"app": "Google Chrome", "name": "chrome-node", "id": 2}
        match, display = _is_meeting_app(stream, ["chrome"])
        assert match is True

    def test_matches_node_name(self):
        stream = {"app": "", "name": "firefox", "id": 3}
        match, display = _is_meeting_app(stream, ["firefox"])
        assert match is True

    def test_no_match(self):
        stream = {"app": "spotify", "name": "spotify-player", "id": 4}
        match, display = _is_meeting_app(stream, ["zoom", "chrome", "teams"])
        assert match is False
        assert display == ""

    def test_partial_match(self):
        stream = {"app": "teams-for-linux", "name": "teams-node", "id": 5}
        match, display = _is_meeting_app(stream, ["teams"])
        assert match is True

    def test_display_uses_app_name_when_available(self):
        stream = {"app": "discord", "name": "some-node", "id": 6}
        match, display = _is_meeting_app(stream, ["discord"])
        assert match is True
        assert display == "Discord"

    def test_display_falls_back_to_node_name(self):
        stream = {"app": "", "name": "slack", "id": 7}
        match, display = _is_meeting_app(stream, ["slack"])
        assert match is True
        assert display == "Slack"


class TestGetMicStreams:
    """Tests for _get_mic_streams PipeWire parsing."""

    def _pw_dump_output(self, nodes):
        return json.dumps(nodes)

    def test_returns_audio_input_streams(self):
        nodes = [
            {
                "type": "PipeWire:Interface:Node",
                "id": 42,
                "info": {
                    "props": {
                        "media.class": "Stream/Input/Audio",
                        "application.name": "Chrome",
                        "node.name": "chrome-input",
                    }
                },
            }
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = self._pw_dump_output(nodes)
            streams = _get_mic_streams()

        assert len(streams) == 1
        assert streams[0]["app"] == "chrome"
        assert streams[0]["id"] == 42

    def test_ignores_non_input_streams(self):
        nodes = [
            {
                "type": "PipeWire:Interface:Node",
                "id": 10,
                "info": {
                    "props": {
                        "media.class": "Stream/Output/Audio",
                        "application.name": "Spotify",
                        "node.name": "spotify-out",
                    }
                },
            }
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = self._pw_dump_output(nodes)
            streams = _get_mic_streams()

        assert len(streams) == 0

    def test_ignores_non_node_types(self):
        nodes = [{"type": "PipeWire:Interface:Link", "id": 1}]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = self._pw_dump_output(nodes)
            streams = _get_mic_streams()

        assert len(streams) == 0

    def test_returns_empty_on_command_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            streams = _get_mic_streams()

        assert streams == []

    def test_returns_empty_on_invalid_json(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "not json"
            streams = _get_mic_streams()

        assert streams == []

    def test_handles_missing_app_name(self):
        nodes = [
            {
                "type": "PipeWire:Interface:Node",
                "id": 99,
                "info": {
                    "props": {
                        "media.class": "Stream/Input/Audio",
                        "node.name": "unknown-node",
                    }
                },
            }
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = self._pw_dump_output(nodes)
            streams = _get_mic_streams()

        assert len(streams) == 1
        assert streams[0]["app"] == ""
        assert streams[0]["name"] == "unknown-node"
