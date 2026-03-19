"""Tests for the diarize plugin."""

from unittest.mock import patch

import click
from click.testing import CliRunner

from murmur.plugins import diarize

runner = CliRunner()


def _make_group():
    @click.group()
    def grp():
        pass

    return grp


class TestCheckDep:
    def test_returns_false_when_missing(self):
        with patch.object(diarize, "_check_dep", return_value=False):
            assert diarize._check_dep() is False


class TestRegister:
    def test_registers_command(self):
        grp = _make_group()
        diarize.register(grp)
        assert "diarize" in grp.commands

    def test_has_hf_token_option(self):
        grp = _make_group()
        diarize.register(grp)
        cmd = grp.commands["diarize"]
        param_names = [p.name for p in cmd.params]
        assert "hf_token" in param_names

    def test_has_speakers_option(self):
        grp = _make_group()
        diarize.register(grp)
        cmd = grp.commands["diarize"]
        param_names = [p.name for p in cmd.params]
        assert "speakers" in param_names

    def test_exits_when_dep_missing(self):
        grp = _make_group()
        diarize.register(grp)
        with patch.object(diarize, "_check_dep", return_value=False):
            result = runner.invoke(grp, ["diarize", __file__])
        assert result.exit_code != 0
