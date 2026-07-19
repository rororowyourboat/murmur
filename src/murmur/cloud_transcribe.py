"""Resumable OpenAI transcription for long recording files."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from murmur.artifacts import ArtifactStore

DEFAULT_MODEL = "gpt-4o-transcribe"
DEFAULT_CHUNK_SECONDS = 600.0
DEFAULT_OVERLAP_SECONDS = 2.0
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SAFE_UPLOAD_BYTES = 24 * 1024 * 1024
ARTIFACT_TEXT = re.compile(
    r"^\s*[\[(](?:music|silence|inaudible|applause|noise)[\])]\s*$", re.IGNORECASE
)


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def unit_id(self) -> str:
        return f"chunk-{self.index:04d}"


class TranscriptionProviderError(RuntimeError):
    """An inspectable provider error with retry guidance."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True)  # noqa: S603


def _media_duration(recording: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required for cloud transcription.")
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(recording),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not inspect recording: {result.stderr.strip()}")
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("ffprobe did not return a valid recording duration.") from error


def _select_audio_stream(manifest: dict[str, Any]) -> int:
    """Prefer the named mixed/default stream, then the first audio stream."""
    streams = manifest.get("media", {}).get("streams", [])
    audio_streams = [stream for stream in streams if stream.get("codec_type", "audio") == "audio"]
    for stream in audio_streams:
        if stream.get("source_role") == "mixed" or stream.get("title") == "Mixed call":
            return int(stream.get("index", 0))
    for stream in audio_streams:
        if stream.get("default") or stream.get("disposition", {}).get("default") == 1:
            return int(stream.get("index", 0))
    return int(audio_streams[0].get("index", 0)) if audio_streams else 0


def plan_chunks(
    duration: float,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> list[Chunk]:
    if duration <= 0:
        raise ValueError("Recording duration must be positive.")
    if chunk_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("Chunk duration must be positive and greater than the overlap.")
    chunks = []
    start = 0.0
    step = chunk_seconds - overlap_seconds
    while start < duration:
        chunks.append(
            Chunk(
                index=len(chunks),
                start=round(start, 6),
                duration=round(min(chunk_seconds, duration - start), 6),
            )
        )
        start += step
    return chunks


def _extract_chunk(recording: Path, output: Path, chunk: Chunk, stream_index: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for cloud transcription.")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(chunk.start),
            "-i",
            str(recording),
            "-t",
            str(chunk.duration),
            "-map",
            f"0:{stream_index}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not create {chunk.unit_id}: {result.stderr.strip()}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg created an empty chunk: {output}")
    if output.stat().st_size >= MAX_UPLOAD_BYTES:
        raise RuntimeError(
            f"{output.name} is {output.stat().st_size} bytes; reduce --chunk-seconds "
            "to remain below OpenAI's 25 MB upload limit."
        )


def _response_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
        if isinstance(payload, dict):
            return payload
    if hasattr(response, "json"):
        payload = json.loads(response.json())
        if isinstance(payload, dict):
            return payload
    text = getattr(response, "text", None)
    if text is not None:
        return {"text": text}
    raise TypeError("OpenAI returned an unsupported transcription response.")


def _retryable_error(error: Exception) -> bool:
    retryable = getattr(error, "retryable", None)
    if isinstance(retryable, bool):
        return retryable
    status = getattr(error, "status_code", None)
    if status is None:
        return True
    try:
        numeric_status = int(status)
    except TypeError, ValueError:
        return True
    return numeric_status in (408, 409, 429) or numeric_status >= 500


def _error_payload(error: Exception) -> dict[str, Any]:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "status_code": getattr(error, "status_code", None),
        "request_id": getattr(error, "request_id", None),
        "retryable": _retryable_error(error),
    }


def _submit_chunk(
    client: Any,
    chunk_path: Path,
    *,
    model: str,
    language: str | None,
    prompt: str | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {"model": model, "response_format": "json"}
    if language:
        request["language"] = language
    if prompt:
        request["prompt"] = prompt
    with chunk_path.open("rb") as audio:
        response = client.audio.transcriptions.create(file=audio, **request)
    return _response_payload(response)


def _artifact_segment(text: str) -> bool:
    return not text.strip() or bool(ARTIFACT_TEXT.fullmatch(text))


def _dedupe_boundary(previous: str, current: str, max_words: int = 80) -> str:
    """Remove the longest repeated word suffix/prefix at a chunk boundary."""
    previous_words = previous.split()
    current_words = current.split()
    maximum = min(max_words, len(previous_words), len(current_words))
    for count in range(maximum, 2, -1):
        left = [re.sub(r"\W+", "", word).casefold() for word in previous_words[-count:]]
        right = [re.sub(r"\W+", "", word).casefold() for word in current_words[:count]]
        if left == right:
            return " ".join(current_words[count:])
    return current.strip()


def _segments_from_response(payload: dict[str, Any], chunk: Chunk) -> list[dict[str, Any]]:
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raw_segments = [
            {
                "id": f"{chunk.unit_id}:0",
                "start": 0.0,
                "end": chunk.duration,
                "text": payload.get("text", ""),
            }
        ]
    normalized = []
    for index, segment in enumerate(raw_segments):
        text = str(segment.get("text", "")).strip()
        if _artifact_segment(text):
            continue
        local_start = max(0.0, float(segment.get("start", 0.0)))
        local_end = min(chunk.duration, float(segment.get("end", chunk.duration)))
        if local_end <= local_start:
            continue
        normalized.append(
            {
                "id": str(segment.get("id", f"{chunk.unit_id}:{index}")),
                "chunk_index": chunk.index,
                "start": round(chunk.start + local_start, 3),
                "end": round(chunk.start + local_end, 3),
                "text": text,
            }
        )
    return normalized


def _merge_responses(responses: list[tuple[Chunk, dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for chunk, response in responses:
        incoming = _segments_from_response(response, chunk)
        if merged and incoming:
            incoming[0]["text"] = _dedupe_boundary(merged[-1]["text"], incoming[0]["text"])
            incoming[0]["start"] = max(incoming[0]["start"], merged[-1]["end"])
            if (
                _artifact_segment(incoming[0]["text"])
                or incoming[0]["end"] <= incoming[0]["start"]
            ):
                incoming.pop(0)
        merged.extend(incoming)
    for index, segment in enumerate(merged, 1):
        segment["id"] = f"segment-{index:06d}"
    return merged


def _clock(seconds: float, *, srt: bool = False) -> str:
    millis = round(seconds * 1000)
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def _render_markdown(segments: list[dict[str, Any]]) -> str:
    lines = ["# Transcript", ""]
    lines.extend(f"[{_clock(segment['start'])}] {segment['text']}" for segment in segments)
    return "\n".join(lines) + "\n"


def _render_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for index, segment in enumerate(segments, 1):
        blocks.append(
            f"{index}\n{_clock(segment['start'], srt=True)} --> "
            f"{_clock(segment['end'], srt=True)}\n{segment['text']}\n"
        )
    return "\n".join(blocks)


def _load_raw(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def transcribe_openai(
    recording: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    prompt: str | None = None,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    resume: bool = True,
    client: Any | None = None,
) -> dict[str, Any]:
    """Transcribe a recording sequentially with durable chunk-level resume."""
    recording_path = Path(recording).expanduser().resolve()
    store = ArtifactStore(recording_path)
    manifest = store.ensure_manifest()
    duration = float(
        manifest.get("media", {}).get("duration_secs") or _media_duration(recording_path)
    )
    chunks = plan_chunks(duration, chunk_seconds, overlap_seconds)
    stream_index = _select_audio_stream(manifest)
    final_names = [
        "transcript_raw",
        "transcript_json",
        "transcript_markdown",
        "transcript_srt",
    ]
    final_paths = [
        store.path("transcript.raw.json"),
        store.path("transcript.json"),
        store.path("transcript.md"),
        store.path("transcript.srt"),
    ]
    job_parameters = {
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "stream_index": stream_index,
        "sequential": True,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }
    if language is not None:
        job_parameters["language"] = language
    if prompt is not None:
        job_parameters["prompt"] = prompt
    previous_job = store.jobs().get("jobs", {}).get("transcribe:openai", {})
    compatible_resume = bool(
        resume
        and previous_job.get("model") == model
        and previous_job.get("parameters") == job_parameters
    )
    _, should_run = store.begin_job(
        "transcribe",
        "openai",
        model=model,
        parameters=job_parameters,
        output_paths=final_paths,
        output_artifacts=final_names,
        resume=compatible_resume,
    )
    if not should_run:
        return _load_raw(store.path("transcript.json"))

    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            store.fail_job("transcribe", "openai", "OPENAI_API_KEY is not set.", retryable=False)
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI transcription.")
        if api_key.startswith("op://"):
            store.fail_job(
                "transcribe",
                "openai",
                "OPENAI_API_KEY contains an unresolved op:// reference.",
                retryable=False,
            )
            raise RuntimeError("Resolve the op:// key with `op run` before starting Murmur.")
        try:
            from openai import OpenAI
        except ImportError as error:
            store.fail_job(
                "transcribe",
                "openai",
                "The optional openai package is not installed.",
                retryable=False,
            )
            raise RuntimeError("Install Murmur with the `cloud` extra.") from error
        client = OpenAI(api_key=api_key)

    responses = []
    raw_index = []
    provenance = {
        "provider": "openai",
        "model": model,
        "parameters": {
            "language": language,
            "prompt": prompt,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "stream_index": stream_index,
        },
    }
    try:
        for chunk in chunks:
            chunk_path = store.path(f"chunks/{chunk.unit_id}.wav")
            chunk_artifact = f"{chunk.unit_id}_audio"
            if not (compatible_resume and store.artifact_valid(chunk_artifact)):
                _extract_chunk(recording_path, chunk_path, chunk, stream_index)
                store.register_artifact(
                    chunk_artifact,
                    chunk_path,
                    kind="transcription_chunk",
                    provenance={
                        "start": chunk.start,
                        "duration": chunk.duration,
                        "sample_rate": 16000,
                        "channels": 1,
                        "stream_index": stream_index,
                    },
                )
            if chunk_path.stat().st_size > SAFE_UPLOAD_BYTES:
                raise RuntimeError(
                    f"{chunk_path.name} is too close to the 25 MB upload limit; "
                    "reduce --chunk-seconds."
                )

            raw_name = f"{chunk.unit_id}_raw"
            raw_path = store.path(f"raw-responses/{chunk.unit_id}.json")
            _, should_submit = store.begin_unit(
                "transcribe",
                "openai",
                chunk.unit_id,
                parameters={
                    "start": chunk.start,
                    "duration": chunk.duration,
                    "audio_artifact": chunk_artifact,
                },
                output_artifacts=[raw_name],
                resume=compatible_resume,
            )
            if not should_submit:
                try:
                    response = _load_raw(raw_path)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    store.fail_unit("transcribe", "openai", chunk.unit_id, error)
                    _, should_submit = store.begin_unit(
                        "transcribe",
                        "openai",
                        chunk.unit_id,
                        output_artifacts=[raw_name],
                    )
            if should_submit:
                try:
                    response = _submit_chunk(
                        client,
                        chunk_path,
                        model=model,
                        language=language,
                        prompt=prompt,
                    )
                except Exception as error:
                    error_path = store.write_json(
                        f"raw-responses/{chunk.unit_id}.error.json",
                        _error_payload(error),
                        sanitize=True,
                    )
                    store.register_artifact(
                        f"{chunk.unit_id}_error",
                        error_path,
                        kind="provider_error",
                        provenance=provenance,
                    )
                    store.fail_unit(
                        "transcribe",
                        "openai",
                        chunk.unit_id,
                        error,
                        retryable=_retryable_error(error),
                    )
                    raise TranscriptionProviderError(
                        f"OpenAI failed for {chunk.unit_id}; inspect {error_path}",
                        retryable=_retryable_error(error),
                    ) from error
                raw_path = store.write_json(f"raw-responses/{chunk.unit_id}.json", response)
                store.register_artifact(
                    raw_name,
                    raw_path,
                    kind="raw_provider_response",
                    provenance=provenance,
                )
                store.complete_unit("transcribe", "openai", chunk.unit_id)
            responses.append((chunk, response))
            raw_index.append(
                {
                    "chunk_index": chunk.index,
                    "start": chunk.start,
                    "duration": chunk.duration,
                    "audio_artifact": chunk_artifact,
                    "response_artifact": raw_name,
                }
            )

        segments = _merge_responses(responses)
        canonical = {
            "schema_version": 1,
            "recording_id": store.recording_id,
            "source": str(recording_path),
            "source_fingerprint": manifest["source_fingerprint"],
            "provider": "openai",
            "model": model,
            "language": language,
            "duration_secs": duration,
            "segments": segments,
            "text": " ".join(segment["text"] for segment in segments),
        }
        raw_manifest_path = store.write_json(
            "transcript.raw.json",
            {"provider": "openai", "model": model, "chunks": raw_index},
        )
        transcript_path = store.write_json("transcript.json", canonical)
        markdown_path = store.write_text("transcript.md", _render_markdown(segments))
        srt_path = store.write_text("transcript.srt", _render_srt(segments))
        for name, path, kind in (
            ("transcript_raw", raw_manifest_path, "raw_response_index"),
            ("transcript_json", transcript_path, "normalized_transcript"),
            ("transcript_markdown", markdown_path, "transcript_markdown"),
            ("transcript_srt", srt_path, "subtitles"),
        ):
            store.register_artifact(name, path, kind=kind, provenance=provenance)
        store.complete_job("transcribe", "openai")
        return canonical
    except Exception as error:
        store.fail_job("transcribe", "openai", error, retryable=_retryable_error(error))
        raise
