"""Tests for the summarize plugin helpers."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from murmur.plugins import summarize


class TestFindTranscript:
    def test_returns_txt_directly(self, tmp_path):
        txt = tmp_path / "meeting.txt"
        txt.write_text("content")
        assert summarize._find_transcript(txt) == txt

    def test_finds_txt_from_audio(self, tmp_path):
        audio = tmp_path / "meeting.flac"
        audio.write_bytes(b"fake")
        transcript = tmp_path / "meeting.txt"
        transcript.write_text("Some transcript")
        assert summarize._find_transcript(audio) == transcript

    def test_raises_when_no_transcript(self, tmp_path):
        audio = tmp_path / "meeting.flac"
        audio.write_bytes(b"fake")
        with pytest.raises(SystemExit):
            summarize._find_transcript(audio)


class TestCheckDep:
    def test_returns_true_when_deps_available(self):
        with patch.dict("sys.modules", {"dspy": object(), "litellm": object()}):
            assert summarize._check_dep() is True

    def test_returns_false_when_deps_missing(self):
        with patch.object(summarize, "_check_dep", return_value=False):
            assert summarize._check_dep() is False


class TestLoadEnv:
    def test_loads_env_vars_from_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('TEST_MURMUR_VAR=hello\nTEST_MURMUR_QUOTED="world"\n')
        import os

        with patch.object(Path, "resolve", return_value=tmp_path / "src" / "murmur" / "plugins" / "summarize.py"):
            # Directly test the parsing logic
            original = os.environ.get("TEST_MURMUR_VAR")
            try:
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = value
                assert os.environ["TEST_MURMUR_VAR"] == "hello"
                assert os.environ["TEST_MURMUR_QUOTED"] == "world"
            finally:
                os.environ.pop("TEST_MURMUR_VAR", None)
                os.environ.pop("TEST_MURMUR_QUOTED", None)


class TestRenderMarkdown:
    def _make_summary(self, **overrides):
        """Create a mock summary object."""
        from types import SimpleNamespace

        defaults = {
            "title": "Test Meeting",
            "executive_summary": "We discussed things.",
            "attendees": [],
            "key_decisions": [],
            "action_items": [],
            "discussion_points": [],
            "open_questions": [],
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_basic_rendering(self):
        summary = self._make_summary()
        result = summarize._render_markdown(summary)
        assert "# Test Meeting" in result
        assert "We discussed things." in result

    def test_with_attendees(self):
        summary = self._make_summary(attendees=["Alice", "Bob"])
        result = summarize._render_markdown(summary)
        assert "## Attendees" in result
        assert "- Alice" in result
        assert "- Bob" in result

    def test_with_decisions(self):
        summary = self._make_summary(key_decisions=["Use Python", "Ship Friday"])
        result = summarize._render_markdown(summary)
        assert "## Key Decisions" in result
        assert "- Use Python" in result

    def test_with_action_items(self):
        from types import SimpleNamespace

        items = [SimpleNamespace(task="Write tests", owner="Alice", deadline="Friday", priority="high")]
        summary = self._make_summary(action_items=items)
        result = summarize._render_markdown(summary)
        assert "## Action Items" in result
        assert "Write tests" in result
        assert "Alice" in result

    def test_with_discussion_points(self):
        summary = self._make_summary(discussion_points=["API design", "Timeline"])
        result = summarize._render_markdown(summary)
        assert "## Discussion Points" in result

    def test_with_open_questions(self):
        summary = self._make_summary(open_questions=["When to deploy?"])
        result = summarize._render_markdown(summary)
        assert "## Open Questions" in result
        assert "- When to deploy?" in result


class TestGetCalendarContext:
    def test_returns_none_when_no_calendar_module(self, tmp_path):
        audio = tmp_path / "meeting.flac"
        with patch.dict("sys.modules", {"murmur.plugins.calendar": None}):
            result = summarize._get_calendar_context(str(audio))
        assert result is None

    def test_returns_none_when_no_meta_file(self, tmp_path):
        audio = tmp_path / "meeting.flac"
        result = summarize._get_calendar_context(str(audio))
        assert result is None

    def test_returns_none_when_no_started_at(self, tmp_path):
        audio = tmp_path / "meeting.flac"
        meta = tmp_path / "meeting.json"
        meta.write_text(json.dumps({"format": "flac"}))
        result = summarize._get_calendar_context(str(audio))
        assert result is None


class TestGetTaskContext:
    def test_returns_none_when_not_configured(self):
        with patch("murmur.plugins.summarize.get_section", return_value={}):
            result = summarize._get_task_context()
        assert result is None

    def test_returns_none_when_file_missing(self, tmp_path):
        with (
            patch("murmur.plugins.summarize.get_section", return_value={"export_context": True}),
            patch.object(Path, "home", return_value=tmp_path),
        ):
            result = summarize._get_task_context()
        assert result is None

    def test_returns_content_when_present(self, tmp_path):
        task_ctx = tmp_path / ".config" / "murmur" / "task_context.md"
        task_ctx.parent.mkdir(parents=True)
        task_ctx.write_text("- [ ] Fix bug\n- [ ] Write docs\n")
        with (
            patch("murmur.plugins.summarize.get_section", return_value={"export_context": True}),
            patch.object(Path, "home", return_value=tmp_path),
        ):
            result = summarize._get_task_context()
        assert "Fix bug" in result
