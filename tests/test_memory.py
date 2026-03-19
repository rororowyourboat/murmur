"""Tests for the memory plugin."""

from unittest.mock import patch

import click
from click.testing import CliRunner

from murmur.plugins import memory

runner = CliRunner()


def _make_group():
    @click.group()
    def grp():
        pass

    return grp


class TestLoadMemory:
    def test_returns_none_when_no_file(self, tmp_path):
        fake_path = tmp_path / "nonexistent" / "memory.md"
        with patch.object(memory, "MEMORY_PATH", fake_path):
            assert memory.load_memory() is None

    def test_returns_content(self, tmp_path):
        mem_file = tmp_path / "memory.md"
        mem_file.write_text("# About me\nI am a developer.")
        with patch.object(memory, "MEMORY_PATH", mem_file):
            result = memory.load_memory()
        assert "I am a developer" in result

    def test_returns_none_for_empty_file(self, tmp_path):
        mem_file = tmp_path / "memory.md"
        mem_file.write_text("   \n  ")
        with patch.object(memory, "MEMORY_PATH", mem_file):
            assert memory.load_memory() is None


class TestMemoryCommands:
    def test_register_adds_memory_group(self):
        grp = _make_group()
        memory.register(grp)
        assert "memory" in grp.commands

    def test_memory_show_with_content(self, tmp_path):
        mem_file = tmp_path / "memory.md"
        mem_file.write_text("# About me\nTest content")
        grp = _make_group()
        memory.register(grp)
        with patch.object(memory, "MEMORY_PATH", mem_file):
            result = runner.invoke(grp, ["memory", "show"])
        assert result.exit_code == 0

    def test_memory_show_no_content(self, tmp_path):
        fake_path = tmp_path / "nonexistent.md"
        grp = _make_group()
        memory.register(grp)
        with patch.object(memory, "MEMORY_PATH", fake_path):
            result = runner.invoke(grp, ["memory", "show"])
        assert result.exit_code == 0
        assert "No memory" in result.output

    def test_memory_path(self, tmp_path):
        grp = _make_group()
        memory.register(grp)
        fake_path = tmp_path / "memory.md"
        with patch.object(memory, "MEMORY_PATH", fake_path):
            result = runner.invoke(grp, ["memory", "path"])
        assert str(fake_path) in result.output

    def test_memory_reset(self, tmp_path):
        mem_file = tmp_path / "memory.md"
        mem_file.write_text("custom content")
        grp = _make_group()
        memory.register(grp)
        with patch.object(memory, "MEMORY_PATH", mem_file):
            result = runner.invoke(grp, ["memory", "reset"])
        assert result.exit_code == 0
        assert "Reset" in result.output
        assert "About me" in mem_file.read_text()

    def test_memory_default_shows_content(self, tmp_path):
        mem_file = tmp_path / "memory.md"
        mem_file.write_text("# My memory content")
        grp = _make_group()
        memory.register(grp)
        with patch.object(memory, "MEMORY_PATH", mem_file):
            result = runner.invoke(grp, ["memory"])
        assert result.exit_code == 0

    def test_memory_default_no_content(self, tmp_path):
        fake_path = tmp_path / "nonexistent.md"
        grp = _make_group()
        memory.register(grp)
        with patch.object(memory, "MEMORY_PATH", fake_path):
            result = runner.invoke(grp, ["memory"])
        assert "No memory" in result.output
