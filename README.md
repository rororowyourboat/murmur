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
diarize = false
diarize_model = "gpt-4o-transcribe-diarize"
speaker_profile = "default"
language = "en"
chunk_seconds = 600
overlap_seconds = 2

[summarize]
auto = true          # auto-summarize after transcription
model = "gemini/gemini-3-flash-preview"
max_chunk_characters = 24000

[summarize.glossary]
k8s = "Kubernetes"
Soo-jata = "Sujata"

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
transcript.json
transcript.cleaned.json
transcript.cleaned.md
transcript.srt
summary.json
summary.md
tasks.preview.json
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

### Speaker profiles and channel-aware diarization

Speaker names are never assigned from anonymous cluster order. Add confirmed
2-10 second clips to a private profile, classifying people by the side of the
call where their voice is captured:

```bash
murmur speakers add Rohan --side local --clip rohan-reference.wav
murmur speakers add Jatan --side local --clip jatan-reference.wav
murmur speakers add Sujata --side remote --clip sujata-reference.wav
murmur speakers add Abby --side remote --clip abby-reference.wav

OPENAI_API_KEY='op://Personal/OPENAI_API_KEY/credential' \
  op run -- murmur transcribe meeting.mka --diarize --speaker-profile default
```

On multitrack recordings, Murmur submits the microphone and call-output streams
independently with only the references relevant to that side, then merges their
absolute timelines while preserving simultaneous speech. Unknown identities
remain explicit, such as `unknown:remote:chunk-0002:B`; they are not silently
guessed across chunk label resets.

Reference clips are normalized to private 16 kHz mono WAV files under
`~/.config/murmur/speaker-profiles` and are converted to data URLs only in
memory for an API request. Review and manage them with:

```bash
murmur speakers list
murmur speakers identify meeting.mka
murmur speakers export default --output default-speakers.zip
murmur speakers delete default --speaker Rohan
murmur speakers delete default
```

`speakers identify` exports candidate clips for unresolved labels. Adding one
of those candidates to a profile is the explicit human confirmation step.

### Grounded summaries and approved tasks

Summaries are generated from the canonical segment-labelled transcript. Murmur
keeps provider output untouched, writes deterministic whitespace cleanup and
uncertainty annotations to separate `transcript.cleaned.*` artifacts, and
validates generated claims against real segment IDs. Unsupported claims are
dropped into the audit trail in `summary.json` rather than presented as facts.

```bash
murmur summarize meeting.mka \
  --glossary Soo-jata=Sujata \
  --glossary k8s=Kubernetes
```

The Markdown output includes attendees, executive summary, topics, decisions,
open questions, action items, and transcription uncertainties. Decisions and
actions show whether they were explicit or inferred, confidence, source
excerpts, and links to timestamps in `transcript.cleaned.md`. Long calls use
map-reduce when they exceed the configured character budget. `summary.json`
records the model, prompt version, source hash, glossary, generation time, and
map/reduce call plan so regeneration is auditable.

Task backends are never changed by extraction alone. The first command writes
an exact, reviewable `tasks.preview.json`; a second explicit command applies
that same preview only if its source has not changed:

```bash
murmur tasks ingest meeting.mka
# review tasks.preview.json
murmur tasks ingest meeting.mka --approve
```

Automatic task extraction also stops at the preview stage.

## Plugins

| Plugin | Command | What it does | Install |
|---|---|---|---|
| **watch** | `murmur watch` | Detect meeting apps using the mic, notify + auto-record | built-in |
| **memory** | `murmur memory` | Personal context for LLM summaries | built-in |
| **tui** | `murmur tui` | Live dashboard with artifact viewer + generation | `murmur[tui]` |
| **summarize** | `murmur summarize <file>` | Grounded map-reduce summary with citations | `murmur[ai]` |
| **transcribe** | `murmur transcribe <file>` | Local or resumable OpenAI transcription | `murmur[transcribe]` / `murmur[cloud]` |
| **diarize** | `murmur diarize <file>` | Speaker diarization → `.rttm` + `.diarized.txt` | `murmur[diarize]` |
| **tasks** | `murmur tasks ingest <file>` | Preview and explicitly approve task changes | `murmur[tasks]` |

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
issue → branch → draft PR → CI workflow, and follow the
[Code of Conduct](CODE_OF_CONDUCT.md). Please report vulnerabilities using the
private process in [SECURITY.md](SECURITY.md).

No license has been selected yet. The explicit decision is deferred in
[#33](https://github.com/rororowyourboat/murmur/issues/33); no reuse license
should be inferred until the maintainer chooses one.

## Roadmap

- [x] Automatic transcription (local Whisper / OpenAI)
- [x] Speaker diarization with confirmed identity profiles
- [x] Grounded summaries and approval-gated action items
- [x] Auto-detect meeting apps and start recording
- [ ] Web UI for browsing/searching recordings
