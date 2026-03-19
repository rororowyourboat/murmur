# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Murmur

Murmur is a CLI tool for recording system audio from meetings on Linux. It captures audio via PipeWire + FFmpeg, with optional microphone input for dual-channel recording. Recordings are saved to `~/Recordings/meetings/` with JSON metadata sidecars.

## Commands

```bash
# Install dependencies (dev)
uv sync --extra dev

# Install with all optional features
uv sync --extra all

# Run the CLI
uv run murmur --help

# Run tests
uv run pytest
uv run pytest tests/test_recorder.py        # single file
uv run pytest tests/test_hooks.py::test_name # single test

# Lint and format (also runs as pre-commit hooks)
uv run ruff check --fix .
uv run ruff format .
```

## Architecture

**Plugin system**: Murmur uses a `click.Group` subclass (`MurmurCLI`) that auto-discovers plugins via `entry_points(group="murmur.plugins")`. Each plugin exports a `register(cli)` function that adds commands and/or hooks.

**Event hooks** (`hooks.py`): Simple pub/sub — plugins call `hooks.on(event, callback)` to subscribe and the recorder calls `hooks.emit(event, **payload)` when things happen. Key events:
- `recording_started` — emitted when FFmpeg starts
- `recording_saved` — emitted when a recording file is written (triggers auto-transcribe)
- `transcription_complete` — emitted after Whisper finishes (triggers auto-summarize)

**Configuration** (`config.py`): TOML config at `~/.config/murmur/config.toml`. Plugins read their section via `get_section("plugin_name")`. The `auto` key in a plugin's config section enables automatic processing via hooks.

**Plugins**:
- `transcribe` — faster-whisper, outputs `.txt` + `.srt`. Optional dep: `murmur[transcribe]`
- `summarize` — Ollama HTTP API (no Python deps), outputs `.summary.md`
- `diarize` — pyannote-audio, outputs `.rttm` + `.diarized.txt`. Optional dep: `murmur[diarize]`
- `watch` — polls PipeWire (`pw-dump`) for mic-using meeting apps, sends notifications, optional auto-record
- `tui` — Rich Live dashboard with keyboard controls (r/s/q)

**Recording** (`recorder.py`): PipeWire sink discovery via `wpctl`, FFmpeg recording via PulseAudio compat layer. Background recordings use a PID file at `~/.cache/murmur/murmur.pid`. Dual-channel mode uses `amerge` filter to mix system audio (left) and mic (right) into stereo.

## Conventions

- Python 3.14+, managed with `uv`
- Ruff for linting (`E, F, I, UP, B, SIM` rules) and formatting, line length 99
- Pre-commit hooks: ruff check + ruff format
- CLI built with Click, output styled with Rich
- No external HTTP dependencies except stdlib `urllib` (summarize plugin)
- Optional heavy deps (torch, faster-whisper, pyannote) are extras, not core deps
