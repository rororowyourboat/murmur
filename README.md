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
# Core only (recording, toggle, watch, devices)
uv pip install murmur

# With TUI dashboard
uv pip install murmur[tui]

# With AI summarization (DSPy + LiteLLM)
uv pip install murmur[ai]

# With transcription (faster-whisper)
uv pip install murmur[transcribe]

# With OpenAI cloud transcription
uv pip install murmur[cloud]

# Everything
uv pip install murmur[all]

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

# Capture a default playback mix plus separate mic and call-output tracks
murmur start --mic

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

# Auto-record with a listening mix plus separate mic and call-output tracks
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

Recordings made with `--mic` use Matroska (`.mka`) and contain three named Opus
streams: **Mixed call** (the default playback stream), **Microphone**, and
**Call output**. The source streams remain independently selectable in players
such as VLC and in FFmpeg.

## Configuration

Optional TOML config at `~/.config/murmur/config.toml`:

```toml
[recording]
output_dir = "~/Recordings/meetings"
artifacts_dir = "~/Recordings/artifacts"
format = "flac"

[watch]
interval = 3
auto_record = true
apps = ["chrome", "firefox", "zoom", "teams", "slack", "discord"]

[transcribe]
auto = true          # auto-transcribe after recording
provider = "local"   # local or openai
model = "base"       # whisper model size
openai_model = "gpt-4o-transcribe"
language = "en"
chunk_seconds = 600
overlap_seconds = 2

[summarize]
auto = true          # auto-summarize after transcription
model = "llama3"     # ollama model
ollama_url = "http://localhost:11434"

[diarize]
hf_token = "hf_..."  # hugging face token for pyannote
```

## Artifacts and processing jobs

Each recording gets a content-fingerprinted manifest and durable job state. By
default, generated files live under `~/Recordings/artifacts/<recording-id>/`:

```text
manifest.json
jobs.json
raw-responses/
speakers/
transcript.txt
transcript.srt
summary.md
```

Writes are atomic. Completed outputs are checksum-validated before they are
reused, so interrupted commands can be run again without repeating valid work.

```bash
murmur jobs status <recording>
murmur jobs status <recording> --json
murmur jobs retry <recording> [--job transcribe]
```

Job metadata records provider and model parameters but removes credentials and
embedded audio data. Raw provider responses are kept separately from derived
transcripts and summaries.

### OpenAI cloud transcription

OpenAI uploads are limited to 25 MB, so Murmur extracts the selected mixed
stream into lossless 16 kHz mono WAV chunks, submits them sequentially, and
writes every response before continuing. The default ten-minute chunks remain
below the limit while a short overlap protects boundary words.

```bash
# Standard environment variable
OPENAI_API_KEY=... murmur transcribe meeting.mka --provider openai

# 1Password reference without placing the secret in config or shell history
OPENAI_API_KEY='op://Personal/OPENAI_API_KEY/credential' \
  op run -- murmur transcribe meeting.mka --provider openai --resume
```

Resume is enabled by default; use `--restart` only when you intentionally want
to submit every chunk again. The source recording is opened read-only and is
never rewritten.

## Plugins

| Plugin | Command | What it does | Install |
|---|---|---|---|
| **watch** | `murmur watch` | Detect meeting apps using the mic, notify + auto-record | built-in |
| **memory** | `murmur memory` | Personal context for LLM summaries | built-in |
| **tui** | `murmur tui` | Live dashboard with artifact viewer + generation | `murmur[tui]` |
| **summarize** | `murmur summarize <file>` | DSPy structured summarization → `.summary.md` | `murmur[ai]` |
| **transcribe** | `murmur transcribe <file>` | Local or resumable OpenAI transcription | `murmur[transcribe]` / `murmur[cloud]` |
| **diarize** | `murmur diarize <file>` | Speaker diarization → `.rttm` + `.diarized.txt` | `murmur[diarize]` |

## Development

```bash
# Set up dev environment
make install

# Run all checks (lint + format + test)
make check

# Individual targets
make lint          # ruff check
make format        # ruff format (auto-fix)
make format-check  # ruff format --check (CI mode)
make test          # pytest
make fix           # ruff check --fix
```

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the small
issue → branch → draft PR → CI workflow. Please report vulnerabilities using
the private process in [SECURITY.md](SECURITY.md).

## Roadmap

- [x] Automatic transcription (local Whisper / OpenAI)
- [ ] Speaker diarization
- [x] Auto-detect meeting apps and start recording
- [ ] Web UI for browsing/searching recordings
