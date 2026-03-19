# murmur

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
uv tool install /path/to/murmur

# Or run from the project directory
uv run murmur --help
```

## Usage

```bash
# List audio output devices
murmur devices

# Start recording (interactive, Ctrl+C to stop)
murmur start
murmur start --tag standup --format mp3

# Toggle recording on/off (for keyboard shortcuts)
murmur toggle

# Check recording status
murmur status

# List saved recordings
murmur list
```

## Meeting detection

Murmur can watch for meeting apps (Zoom, Google Meet, Teams, etc.) using your microphone and notify you — or automatically start recording.

```bash
# Get notified when a meeting app grabs your mic
murmur watch

# Auto-record when a meeting is detected
murmur watch --auto-record

# Auto-record with mic input (dual-channel: system left, mic right)
murmur watch --auto-record --mic

# Faster polling (default: 5s)
murmur watch --interval 3
```

Works by polling PipeWire for `Stream/Input/Audio` nodes — detects Chrome, Firefox, Zoom, Teams, Slack, Discord, WebEx, Skype, and more. When the app releases the mic, you get a second notification and auto-recording stops.

## Keyboard shortcut

**Super+Shift+R** toggles recording on/off with a desktop notification.

Set up via GNOME custom shortcuts — runs `murmur toggle` in the background.

## Recordings

Saved to `~/Recordings/meetings/` with naming convention:

```
meeting_standup_2026-03-18_14-30-00.flac
meeting_standup_2026-03-18_14-30-00.json   # metadata
```

## Configuration

Optional TOML config at `~/.config/murmur/config.toml`:

```toml
[recording]
output_dir = "~/Recordings/meetings"
format = "flac"

[watch]
interval = 3
auto_record = true
apps = ["chrome", "firefox", "zoom", "teams", "slack", "discord"]

[transcribe]
auto = true          # auto-transcribe after recording
model = "base"       # whisper model size
language = "en"

[summarize]
auto = true          # auto-summarize after transcription
model = "llama3"     # ollama model
ollama_url = "http://localhost:11434"

[diarize]
hf_token = "hf_..."  # hugging face token for pyannote
```

## Plugins

| Plugin | Command | What it does | Install |
|---|---|---|---|
| **watch** | `murmur watch` | Detect meeting apps using the mic, notify + auto-record | built-in |
| **tui** | `murmur tui` | Live dashboard with keyboard controls (r/s/q) | built-in |
| **transcribe** | `murmur transcribe <file>` | Whisper transcription → `.txt` + `.srt` | `uv pip install murmur[transcribe]` |
| **summarize** | `murmur summarize <file>` | Ollama summarization → `.summary.md` | built-in (needs Ollama running) |
| **diarize** | `murmur diarize <file>` | Speaker diarization → `.rttm` + `.diarized.txt` | `uv pip install murmur[diarize]` |

## Roadmap

- [ ] Automatic transcription (Whisper local / Deepgram API)
- [ ] Speaker diarization
- [x] Auto-detect meeting apps and start recording
- [ ] Web UI for browsing/searching recordings
