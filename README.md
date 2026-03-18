# recorder

Record system audio from meetings (Zoom, Google Meet, Teams, etc.) on Linux using PipeWire + FFmpeg.

## Why

No-fuss local meeting recording — hit a keyboard shortcut when a call starts, hit it again when it ends. Audio is saved locally, no cloud services or meeting bots needed.

## Goals

- **Simple toggle workflow** — one shortcut key to start/stop recording, with desktop notifications
- **Universal** — captures any audio playing through your speakers/headphones, works with any meeting app
- **Local-first** — recordings stay on disk at `~/Recordings/meetings/`, no uploads
- **Low overhead** — just FFmpeg recording a PipeWire monitor source
- **Future: transcription** — pipe recordings through Whisper or a transcription API for searchable meeting notes

## How it works

1. Discovers PipeWire audio sinks via `wpctl`
2. Attaches to the **monitor source** of your default output device (captures everything you hear)
3. Records via FFmpeg's PulseAudio input (`pipewire-pulse` compatibility layer)
4. Saves audio file + JSON metadata (timestamps, source, duration)

## Requirements

- Linux with PipeWire + `pipewire-pulse`
- FFmpeg
- `wpctl` (WirePlumber)
- `notify-send` (for desktop notifications from toggle)
- Python 3.14+ / uv

## Install

```bash
# Install globally as a CLI tool
uv tool install /path/to/recorder

# Or run from the project directory
uv run recorder --help
```

## Usage

```bash
# List audio output devices
recorder devices

# Start recording (interactive, Ctrl+C to stop)
recorder start
recorder start --tag standup --format mp3

# Toggle recording on/off (for keyboard shortcuts)
recorder toggle

# Check recording status
recorder status

# List saved recordings
recorder list
```

## Keyboard shortcut

**Super+Shift+R** toggles recording on/off with a desktop notification.

Set up via GNOME custom shortcuts — runs `recorder toggle` in the background.

## Recordings

Saved to `~/Recordings/meetings/` with naming convention:

```
meeting_standup_2026-03-18_14-30-00.flac
meeting_standup_2026-03-18_14-30-00.json   # metadata
```

## Roadmap

- [ ] Automatic transcription (Whisper local / Deepgram API)
- [ ] Speaker diarization
- [ ] Auto-detect meeting apps and start recording
- [ ] Web UI for browsing/searching recordings
