"""TOML configuration for Murmur (~/.config/murmur/config.toml)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

CONFIG_PATH = Path.home() / ".config" / "murmur" / "config.toml"

_config: dict[str, Any] | None = None


def load() -> dict[str, Any]:
    """Load config from disk, returning empty dict if file doesn't exist."""
    global _config
    if _config is not None:
        return _config
    _config = tomllib.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    return _config


def get_section(name: str) -> dict[str, Any]:
    """Get a config section by name, returning empty dict if missing."""
    return load().get(name, {})
