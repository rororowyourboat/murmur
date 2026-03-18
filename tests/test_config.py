"""Tests for TOML config loading."""

from unittest.mock import patch

import murmur.config as config_mod


def setup_function():
    # Reset cached config between tests
    config_mod._config = None


def test_load_missing_file(tmp_path):
    fake_path = tmp_path / "nonexistent" / "config.toml"
    with patch.object(config_mod, "CONFIG_PATH", fake_path):
        result = config_mod.load()

    assert result == {}


def test_load_valid_toml(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[recording]\nformat = "mp3"\noutput_dir = "/tmp/recordings"\n')
    with patch.object(config_mod, "CONFIG_PATH", cfg_file):
        result = config_mod.load()

    assert result["recording"]["format"] == "mp3"
    assert result["recording"]["output_dir"] == "/tmp/recordings"


def test_get_section_exists(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[transcribe]\nmodel = "large"\nlanguage = "fr"\n')
    with patch.object(config_mod, "CONFIG_PATH", cfg_file):
        section = config_mod.get_section("transcribe")

    assert section["model"] == "large"
    assert section["language"] == "fr"


def test_get_section_missing(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[recording]\nformat = "flac"\n')
    with patch.object(config_mod, "CONFIG_PATH", cfg_file):
        section = config_mod.get_section("nonexistent")

    assert section == {}


def test_load_caches_result(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[recording]\nformat = "wav"\n')
    with patch.object(config_mod, "CONFIG_PATH", cfg_file):
        first = config_mod.load()
        # Modify file — should still return cached
        cfg_file.write_text('[recording]\nformat = "ogg"\n')
        second = config_mod.load()

    assert first is second
    assert first["recording"]["format"] == "wav"
