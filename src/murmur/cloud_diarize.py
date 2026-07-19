"""Channel-aware OpenAI transcription with anchored speaker identities."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from murmur.artifacts import ArtifactStore
from murmur.cloud_transcribe import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_OVERLAP_SECONDS,
    SAFE_UPLOAD_BYTES,
    TranscriptionProviderError,
    _artifact_segment,
    _clock,
    _dedupe_boundary,
    _error_payload,
    _extract_chunk,
    _load_raw,
    _media_duration,
    _response_payload,
    _retryable_error,
    plan_chunks,
)
from murmur.speaker_profiles import load_profile, reference_payload

DEFAULT_DIARIZE_MODEL = "gpt-4o-transcribe-diarize"


@dataclass(frozen=True)
class Track:
    side: str
    stream_index: int


def _select_tracks(manifest: dict[str, Any]) -> list[Track]:
    streams = manifest.get("media", {}).get("streams", [])
    by_role = {
        stream.get("source_role"): int(stream.get("index", 0))
        for stream in streams
        if stream.get("codec_type", "audio") == "audio"
    }
    if "microphone" in by_role and "call_output" in by_role:
        return [Track("local", by_role["microphone"]), Track("remote", by_role["call_output"])]
    if "microphone" in by_role:
        return [Track("local", by_role["microphone"])]
    if "call_output" in by_role:
        return [Track("remote", by_role["call_output"])]
    mixed = next(
        (
            int(stream.get("index", 0))
            for stream in streams
            if stream.get("source_role") == "mixed" or stream.get("title") == "Mixed call"
        ),
        int(streams[0].get("index", 0)) if streams else 0,
    )
    return [Track("unknown", mixed)]


def _submit_diarized_chunk(
    client: Any,
    chunk_path: Path,
    *,
    model: str,
    language: str | None,
    known_names: list[str],
    known_references: list[str],
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "response_format": "diarized_json",
        "chunking_strategy": "auto",
    }
    if language:
        request["language"] = language
    if known_names:
        request["known_speaker_names"] = known_names
        request["known_speaker_references"] = known_references
    with chunk_path.open("rb") as audio:
        response = client.audio.transcriptions.create(file=audio, **request)
    return _response_payload(response)


def _normalize_segments(
    payload: dict[str, Any],
    *,
    side: str,
    stream_index: int,
    chunk_index: int,
    chunk_start: float,
    chunk_duration: float,
    known_names: set[str],
) -> list[dict[str, Any]]:
    normalized = []
    raw_segments = payload.get("segments", [])
    if not isinstance(raw_segments, list):
        return normalized
    for segment_index, segment in enumerate(raw_segments):
        text = str(segment.get("text", "")).strip()
        if _artifact_segment(text):
            continue
        local_start = max(0.0, float(segment.get("start", 0.0)))
        local_end = min(chunk_duration, float(segment.get("end", chunk_duration)))
        if local_end <= local_start:
            continue
        raw_speaker = str(segment.get("speaker") or "speaker")
        speaker = (
            raw_speaker
            if raw_speaker in known_names
            else f"unknown:{side}:chunk-{chunk_index:04d}:{raw_speaker}"
        )
        normalized.append(
            {
                "id": f"{side}:chunk-{chunk_index:04d}:{segment_index}",
                "chunk_index": chunk_index,
                "side": side,
                "stream_index": stream_index,
                "speaker": speaker,
                "raw_speaker": raw_speaker,
                "start": round(chunk_start + local_start, 3),
                "end": round(chunk_start + local_end, 3),
                "text": text,
            }
        )
    return normalized


def _merge_track_segments(chunks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for incoming in chunks:
        if merged and incoming:
            first = incoming[0]
            previous = merged[-1]
            first["text"] = _dedupe_boundary(previous["text"], first["text"])
            first["start"] = max(first["start"], previous["end"])
            if _artifact_segment(first["text"]) or first["end"] <= first["start"]:
                incoming.pop(0)
        merged.extend(incoming)
    return merged


def _render_markdown(segments: list[dict[str, Any]]) -> str:
    lines = ["# Speaker transcript", ""]
    lines.extend(
        f"[{_clock(segment['start'])}] **{segment['speaker']}** "
        f"({segment['side']}): {segment['text']}"
        for segment in segments
    )
    return "\n".join(lines) + "\n"


def _render_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for index, segment in enumerate(segments, 1):
        blocks.append(
            f"{index}\n{_clock(segment['start'], srt=True)} --> "
            f"{_clock(segment['end'], srt=True)}\n"
            f"[{segment['speaker']}] {segment['text']}\n"
        )
    return "\n".join(blocks)


def _profile_digest(profile: dict[str, Any]) -> str:
    identity = {
        "name": profile.get("name"),
        "speakers": [
            {
                "id": speaker.get("id"),
                "display_name": speaker.get("display_name"),
                "side": speaker.get("side"),
                "references": [
                    {
                        "path": reference.get("path"),
                        "duration_secs": reference.get("duration_secs"),
                        "fingerprint": reference.get("fingerprint"),
                    }
                    for reference in speaker.get("references", [])
                ],
            }
            for speaker in profile.get("speakers", [])
        ],
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def transcribe_openai_diarized(
    recording: str | Path,
    *,
    profile_name: str = "default",
    model: str = DEFAULT_DIARIZE_MODEL,
    language: str | None = None,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    resume: bool = True,
    client: Any | None = None,
) -> dict[str, Any]:
    """Diarize independent source tracks and merge them without losing overlap."""
    recording_path = Path(recording).expanduser().resolve()
    store = ArtifactStore(recording_path)
    manifest = store.ensure_manifest()
    duration = float(
        manifest.get("media", {}).get("duration_secs") or _media_duration(recording_path)
    )
    chunks = plan_chunks(duration, chunk_seconds, overlap_seconds)
    tracks = _select_tracks(manifest)
    profile = load_profile(profile_name)
    profile_digest = _profile_digest(profile)
    job_parameters = {
        "profile": profile_name,
        "profile_sha256": profile_digest,
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "tracks": [track.__dict__ for track in tracks],
        "channel_aware": len(tracks) > 1,
    }
    if language is not None:
        job_parameters["language"] = language
    artifact_names = [
        "diarized_transcript_raw",
        "diarized_transcript_json",
        "diarized_transcript_markdown",
        "diarized_transcript_srt",
    ]
    output_paths = [
        store.path("transcript.diarized.raw.json"),
        store.path("transcript.json"),
        store.path("transcript.md"),
        store.path("transcript.srt"),
    ]
    previous = store.jobs().get("jobs", {}).get("transcribe:openai-diarize", {})
    compatible_resume = bool(
        resume and previous.get("model") == model and previous.get("parameters") == job_parameters
    )
    _, should_run = store.begin_job(
        "transcribe",
        "openai-diarize",
        model=model,
        parameters=job_parameters,
        output_paths=output_paths,
        output_artifacts=artifact_names,
        resume=compatible_resume,
    )
    if not should_run:
        return _load_raw(store.path("transcript.json"))

    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            store.fail_job(
                "transcribe", "openai-diarize", "OPENAI_API_KEY is not set.", retryable=False
            )
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI diarization.")
        if api_key.startswith("op://"):
            store.fail_job(
                "transcribe",
                "openai-diarize",
                "OPENAI_API_KEY contains an unresolved op:// reference.",
                retryable=False,
            )
            raise RuntimeError("Resolve the op:// key with `op run` before starting Murmur.")
        try:
            from openai import OpenAI
        except ImportError as error:
            store.fail_job(
                "transcribe",
                "openai-diarize",
                "The optional openai package is not installed.",
                retryable=False,
            )
            raise RuntimeError("Install Murmur with the `cloud` extra.") from error
        client = OpenAI(api_key=api_key)

    all_segments: list[dict[str, Any]] = []
    raw_index = []
    provenance = {
        "provider": "openai",
        "model": model,
        "profile": profile_name,
        "profile_sha256": profile_digest,
    }
    try:
        for track in tracks:
            known_names, known_references = reference_payload(profile_name, track.side)
            track_chunks: list[list[dict[str, Any]]] = []
            for chunk in chunks:
                unit_id = f"{track.side}-{chunk.unit_id}"
                chunk_path = store.path(f"chunks/diarize/{track.side}/{chunk.unit_id}.wav")
                chunk_artifact = f"diarize_{unit_id}_audio"
                if not (compatible_resume and store.artifact_valid(chunk_artifact)):
                    _extract_chunk(recording_path, chunk_path, chunk, track.stream_index)
                    store.register_artifact(
                        chunk_artifact,
                        chunk_path,
                        kind="diarization_chunk",
                        provenance={
                            "side": track.side,
                            "stream_index": track.stream_index,
                            "start": chunk.start,
                            "duration": chunk.duration,
                            "sample_rate": 16000,
                            "channels": 1,
                        },
                    )
                if chunk_path.stat().st_size > SAFE_UPLOAD_BYTES:
                    raise RuntimeError(
                        f"{chunk_path.name} is too close to the upload limit; "
                        "reduce --chunk-seconds."
                    )
                raw_name = f"diarize_{unit_id}_raw"
                raw_path = store.path(f"raw-responses/diarize/{track.side}/{chunk.unit_id}.json")
                _, should_submit = store.begin_unit(
                    "transcribe",
                    "openai-diarize",
                    unit_id,
                    parameters={
                        "side": track.side,
                        "stream_index": track.stream_index,
                        "start": chunk.start,
                        "duration": chunk.duration,
                        "known_speakers": known_names,
                    },
                    output_artifacts=[raw_name],
                    resume=compatible_resume,
                )
                if not should_submit:
                    try:
                        response = _load_raw(raw_path)
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        store.fail_unit("transcribe", "openai-diarize", unit_id, error)
                        _, should_submit = store.begin_unit(
                            "transcribe",
                            "openai-diarize",
                            unit_id,
                            output_artifacts=[raw_name],
                        )
                if should_submit:
                    try:
                        response = _submit_diarized_chunk(
                            client,
                            chunk_path,
                            model=model,
                            language=language,
                            known_names=known_names,
                            known_references=known_references,
                        )
                    except Exception as error:
                        error_path = store.write_json(
                            f"raw-responses/diarize/{track.side}/{chunk.unit_id}.error.json",
                            _error_payload(error),
                            sanitize=True,
                        )
                        store.register_artifact(
                            f"diarize_{unit_id}_error",
                            error_path,
                            kind="provider_error",
                            provenance=provenance,
                        )
                        store.fail_unit(
                            "transcribe",
                            "openai-diarize",
                            unit_id,
                            error,
                            retryable=_retryable_error(error),
                        )
                        raise TranscriptionProviderError(
                            f"OpenAI failed for {unit_id}; inspect {error_path}",
                            retryable=_retryable_error(error),
                        ) from error
                    raw_path = store.write_json(
                        f"raw-responses/diarize/{track.side}/{chunk.unit_id}.json", response
                    )
                    store.register_artifact(
                        raw_name,
                        raw_path,
                        kind="raw_provider_response",
                        provenance=provenance,
                    )
                    store.complete_unit("transcribe", "openai-diarize", unit_id)
                track_chunks.append(
                    _normalize_segments(
                        response,
                        side=track.side,
                        stream_index=track.stream_index,
                        chunk_index=chunk.index,
                        chunk_start=chunk.start,
                        chunk_duration=chunk.duration,
                        known_names=set(known_names),
                    )
                )
                raw_index.append(
                    {
                        "side": track.side,
                        "stream_index": track.stream_index,
                        "chunk_index": chunk.index,
                        "start": chunk.start,
                        "duration": chunk.duration,
                        "response_artifact": raw_name,
                    }
                )
            all_segments.extend(_merge_track_segments(track_chunks))

        all_segments.sort(key=lambda segment: (segment["start"], segment["end"], segment["side"]))
        for index, segment in enumerate(all_segments, 1):
            segment["id"] = f"segment-{index:06d}"
        canonical = {
            "schema_version": 1,
            "recording_id": store.recording_id,
            "source": str(recording_path),
            "source_fingerprint": manifest["source_fingerprint"],
            "provider": "openai",
            "model": model,
            "speaker_profile": profile_name,
            "speaker_profile_sha256": profile_digest,
            "channel_aware": len(tracks) > 1,
            "language": language,
            "duration_secs": duration,
            "segments": all_segments,
            "text": " ".join(segment["text"] for segment in all_segments),
        }
        raw_manifest = store.write_json(
            "transcript.diarized.raw.json",
            {"provider": "openai", "model": model, "profile": profile_name, "chunks": raw_index},
        )
        transcript = store.write_json("transcript.json", canonical)
        markdown = store.write_text("transcript.md", _render_markdown(all_segments))
        subtitles = store.write_text("transcript.srt", _render_srt(all_segments))
        for name, path, kind in (
            ("diarized_transcript_raw", raw_manifest, "raw_response_index"),
            ("diarized_transcript_json", transcript, "normalized_speaker_transcript"),
            ("diarized_transcript_markdown", markdown, "speaker_transcript_markdown"),
            ("diarized_transcript_srt", subtitles, "speaker_subtitles"),
        ):
            store.register_artifact(name, path, kind=kind, provenance=provenance)
        store.complete_job("transcribe", "openai-diarize")
        return canonical
    except Exception as error:
        store.fail_job("transcribe", "openai-diarize", error, retryable=_retryable_error(error))
        raise
