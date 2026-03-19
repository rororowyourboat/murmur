"""Tests for the tasks_extract plugin helpers."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from murmur.plugins import tasks_extract


class TestCheckDep:
    def test_returns_true_when_available(self):
        with patch.dict("sys.modules", {"dspy": object(), "litellm": object()}):
            assert tasks_extract._check_dep() is True

    def test_returns_false_when_missing(self):
        with patch.object(tasks_extract, "_check_dep", return_value=False):
            assert tasks_extract._check_dep() is False


class TestFindInputFile:
    def test_returns_txt_directly(self, tmp_path):
        txt = tmp_path / "meeting.txt"
        txt.write_text("transcript content")
        assert tasks_extract._find_input_file(txt) == txt

    def test_returns_md_directly(self, tmp_path):
        md = tmp_path / "meeting.summary.md"
        md.write_text("# Summary")
        assert tasks_extract._find_input_file(md) == md

    def test_prefers_summary_over_transcript(self, tmp_path):
        audio = tmp_path / "meeting.flac"
        audio.write_bytes(b"fake")
        summary = tmp_path / "meeting.summary.md"
        summary.write_text("# Summary")
        transcript = tmp_path / "meeting.txt"
        transcript.write_text("transcript")
        result = tasks_extract._find_input_file(audio)
        assert result == summary

    def test_falls_back_to_transcript(self, tmp_path):
        audio = tmp_path / "meeting.flac"
        audio.write_bytes(b"fake")
        transcript = tmp_path / "meeting.txt"
        transcript.write_text("transcript content")
        result = tasks_extract._find_input_file(audio)
        assert result == transcript

    def test_raises_when_nothing_found(self, tmp_path):
        audio = tmp_path / "meeting.flac"
        audio.write_bytes(b"fake")
        with pytest.raises(click.ClickException):
            tasks_extract._find_input_file(audio)

    def test_raises_for_missing_txt(self, tmp_path):
        txt = tmp_path / "nonexistent.txt"
        with pytest.raises(click.ClickException):
            tasks_extract._find_input_file(txt)


class TestFormatExistingTasks:
    def test_empty_tasks(self):
        assert tasks_extract._format_existing_tasks([]) == "No existing open tasks."

    def test_formats_basic_task(self):
        task = SimpleNamespace(
            title="Fix bug", priority="high", owner="", project="", deadline=""
        )
        result = tasks_extract._format_existing_tasks([task])
        assert "[high] Fix bug" in result

    def test_formats_task_with_owner(self):
        task = SimpleNamespace(
            title="Write docs", priority="normal", owner="Alice", project="", deadline=""
        )
        result = tasks_extract._format_existing_tasks([task])
        assert "(@Alice)" in result

    def test_formats_task_with_project(self):
        task = SimpleNamespace(
            title="Deploy", priority="normal", owner="", project="murmur", deadline=""
        )
        result = tasks_extract._format_existing_tasks([task])
        assert "(+murmur)" in result

    def test_formats_task_with_deadline(self):
        task = SimpleNamespace(
            title="Ship it", priority="high", owner="", project="", deadline="2026-03-25"
        )
        result = tasks_extract._format_existing_tasks([task])
        assert "(due:2026-03-25)" in result

    def test_formats_full_task(self):
        task = SimpleNamespace(
            title="Review PR",
            priority="high",
            owner="Bob",
            project="backend",
            deadline="Friday",
        )
        result = tasks_extract._format_existing_tasks([task])
        assert "[high] Review PR" in result
        assert "(@Bob)" in result
        assert "(+backend)" in result
        assert "(due:Friday)" in result


class TestGetCalendarContext:
    def test_returns_none_when_no_calendar(self, tmp_path):
        with patch.dict("sys.modules", {"murmur.plugins.calendar": None}):
            result = tasks_extract._get_calendar_context(str(tmp_path / "meeting.flac"))
        assert result is None

    def test_returns_none_when_no_meta(self, tmp_path):
        result = tasks_extract._get_calendar_context(str(tmp_path / "meeting.flac"))
        assert result is None


class TestWriteTasksJson:
    def test_writes_sidecar_file(self, tmp_path):
        analysis = SimpleNamespace(
            new_tasks=[
                SimpleNamespace(
                    title="Fix bug",
                    owner="Alice",
                    deadline="Friday",
                    priority="high",
                    project="backend",
                    confidence=0.9,
                )
            ],
            blockers_raised=["CI is broken"],
            blockers_resolved=["Deploy access granted"],
        )
        audio = tmp_path / "meeting.flac"
        sidecar = tasks_extract._write_tasks_json(str(audio), analysis)
        assert sidecar.exists()
        assert sidecar.suffix == ".json"

        import json

        data = json.loads(sidecar.read_text())
        assert len(data["new_tasks"]) == 1
        assert data["new_tasks"][0]["title"] == "Fix bug"
        assert data["blockers_raised"] == ["CI is broken"]
        assert data["blockers_resolved"] == ["Deploy access granted"]

    def test_includes_updates_when_provided(self, tmp_path):
        analysis = SimpleNamespace(
            new_tasks=[], blockers_raised=[], blockers_resolved=[]
        )
        updates = [
            (
                SimpleNamespace(id="abc123", title="Old task"),
                SimpleNamespace(
                    new_status="done",
                    new_deadline="",
                    discussion_context="Completed in meeting",
                ),
            )
        ]
        audio = tmp_path / "meeting.flac"
        sidecar = tasks_extract._write_tasks_json(str(audio), analysis, updates)

        import json

        data = json.loads(sidecar.read_text())
        assert len(data["task_updates"]) == 1
        assert data["task_updates"][0]["task_id"] == "abc123"
        assert data["task_updates"][0]["new_status"] == "done"
