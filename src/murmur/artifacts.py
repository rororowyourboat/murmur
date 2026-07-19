"""Canonical recording artifacts and durable processing job state."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from murmur.config import get_section

SCHEMA_VERSION = 1
SECRET_MARKERS = ("secret", "token", "api_key", "apikey", "authorization", "credential")


class CorruptStateError(RuntimeError):
    """Raised when a persisted manifest or jobs file is not valid JSON state."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a JSON file without exposing a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError) as error:
        raise CorruptStateError(f"Could not read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CorruptStateError(f"Expected a JSON object in {path}.")
    return payload


def _sanitize(value: Any, key: str = "") -> Any:
    """Remove credentials and embedded audio from persisted parameters."""
    lowered = key.lower()
    if any(marker in lowered for marker in SECRET_MARKERS):
        return None
    if isinstance(value, str):
        if value.lstrip().lower().startswith("data:audio/"):
            return "<redacted-audio-data-url>"
        sanitized = re.sub(r"\b(?:sk-|hf_)[A-Za-z0-9_-]{8,}\b", "<redacted-secret>", value)
        sanitized = re.sub(r"(?i)\bBearer\s+\S+", "Bearer <redacted-secret>", sanitized)
        return re.sub(r"op://[^\s]+", "<redacted-secret-reference>", sanitized)
    if isinstance(value, dict):
        return {
            child_key: sanitized
            for child_key, child_value in value.items()
            if (sanitized := _sanitize(child_value, str(child_key))) is not None
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def fingerprint_file(path: Path) -> dict[str, Any]:
    """Return a stable content fingerprint for a source or artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "algorithm": "sha256",
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def default_artifacts_root(source: Path | None = None) -> Path:
    """Return the configured artifact root beside the recordings directory."""
    config = get_section("recording")
    configured = config.get("artifacts_dir")
    if configured:
        return Path(configured).expanduser()
    recordings_dir = Path(config.get("output_dir", "~/Recordings/meetings")).expanduser()
    if source is not None and source.resolve().parent != recordings_dir.resolve():
        return source.resolve().parent / "artifacts"
    return recordings_dir.parent / "artifacts"


class ArtifactStore:
    """Manage one recording's canonical manifest, artifacts, and jobs."""

    def __init__(self, recording: str | Path, root: str | Path | None = None):
        self.recording = Path(recording).expanduser().resolve()
        self.recording_id = self.recording.stem
        artifacts_root = (
            Path(root).expanduser() if root is not None else default_artifacts_root(self.recording)
        )
        artifacts_root = artifacts_root.resolve()
        self.directory = artifacts_root / self.recording_id
        existing_manifest = self.directory / "manifest.json"
        if existing_manifest.exists():
            existing = _read_json(existing_manifest, {})
            if existing.get("recording") not in (None, str(self.recording)):
                path_hash = hashlib.sha256(str(self.recording).encode()).hexdigest()[:8]
                self.recording_id = f"{self.recording.stem}-{path_hash}"
                self.directory = artifacts_root / self.recording_id
        self.manifest_path = self.directory / "manifest.json"
        self.jobs_path = self.directory / "jobs.json"

    def path(self, relative: str | Path) -> Path:
        candidate = (self.directory / relative).resolve()
        if not candidate.is_relative_to(self.directory.resolve()):
            raise ValueError("Artifact path must remain inside the recording artifact directory.")
        return candidate

    @classmethod
    def for_input(cls, path: str | Path) -> ArtifactStore:
        """Resolve an audio file or canonical artifact back to its recording."""
        candidate = Path(path).expanduser().resolve()
        for parent in (candidate.parent, *candidate.parents):
            manifest_path = parent / "manifest.json"
            if manifest_path.is_file():
                manifest = _read_json(manifest_path, {})
                recording = manifest.get("recording")
                if recording:
                    return cls(recording, root=parent.parent)
        return cls(candidate)

    def write_text(self, relative: str | Path, content: str) -> Path:
        output = self.path(relative)
        _atomic_write_text(output, content)
        return output

    def write_json(self, relative: str | Path, payload: dict[str, Any]) -> Path:
        output = self.path(relative)
        _atomic_write_json(output, payload)
        return output

    def _source_fingerprint(self, previous: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.recording.is_file():
            raise FileNotFoundError(f"Recording not found: {self.recording}")
        stat = self.recording.stat()
        if (
            previous
            and previous.get("size_bytes") == stat.st_size
            and previous.get("mtime_ns") == stat.st_mtime_ns
            and previous.get("sha256")
        ):
            return previous
        return fingerprint_file(self.recording)

    def ensure_manifest(self, media_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create or refresh the canonical manifest for the recording."""
        existing = _read_json(self.manifest_path, {})
        now = _now()
        if media_metadata is None:
            sidecar = self.recording.with_suffix(".json")
            media_metadata = _read_json(sidecar, {}) if sidecar.exists() else {}
        media_fields = (
            "format",
            "duration_secs",
            "file_size_bytes",
            "streams",
            "source",
            "sink_id",
            "mic_source",
            "mic_id",
            "capture_mode",
        )
        media = dict(existing.get("media", {}))
        media.update(
            {field: media_metadata[field] for field in media_fields if field in media_metadata}
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "recording_id": self.recording_id,
            "recording": str(self.recording),
            "source_fingerprint": self._source_fingerprint(existing.get("source_fingerprint")),
            "media": media,
            "artifacts": existing.get("artifacts", {}),
            "created_at": existing.get("created_at", now),
            "updated_at": now,
        }
        _atomic_write_json(self.manifest_path, manifest)
        if not self.jobs_path.exists():
            _atomic_write_json(
                self.jobs_path,
                {"schema_version": SCHEMA_VERSION, "jobs": {}, "updated_at": now},
            )
        return manifest

    def manifest(self) -> dict[str, Any]:
        return _read_json(self.manifest_path, {})

    def jobs(self) -> dict[str, Any]:
        return _read_json(
            self.jobs_path,
            {"schema_version": SCHEMA_VERSION, "jobs": {}, "updated_at": _now()},
        )

    def register_artifact(
        self,
        name: str,
        path: str | Path,
        *,
        kind: str,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a completed artifact and its checksum to the manifest inventory."""
        artifact_path = Path(path).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        if not artifact_path.is_relative_to(self.directory.resolve()):
            raise ValueError("Registered artifacts must live in the canonical artifact directory.")
        manifest = self.ensure_manifest()
        entry = {
            "path": str(artifact_path.relative_to(self.directory)),
            "kind": kind,
            "fingerprint": fingerprint_file(artifact_path),
            "provenance": _sanitize(provenance or {}),
            "created_at": _now(),
        }
        manifest["artifacts"][name] = entry
        manifest["updated_at"] = _now()
        _atomic_write_json(self.manifest_path, manifest)
        return entry

    def artifact_valid(self, name: str) -> bool:
        entry = self.manifest().get("artifacts", {}).get(name)
        if not entry:
            return False
        path = self.path(entry["path"])
        if not path.is_file():
            return False
        expected = entry.get("fingerprint", {}).get("sha256")
        return bool(expected and fingerprint_file(path)["sha256"] == expected)

    @staticmethod
    def job_id(job_type: str, provider: str) -> str:
        return f"{job_type}:{provider}"

    def begin_job(
        self,
        job_type: str,
        provider: str,
        *,
        model: str | None = None,
        parameters: dict[str, Any] | None = None,
        input_paths: list[str | Path] | None = None,
        output_paths: list[str | Path] | None = None,
        output_artifacts: list[str] | None = None,
        retryable: bool = True,
        resume: bool = True,
    ) -> tuple[dict[str, Any], bool]:
        """Persist a running job; return ``(job, should_run)``."""
        manifest = self.ensure_manifest()
        payload = self.jobs()
        identifier = self.job_id(job_type, provider)
        previous = payload["jobs"].get(identifier, {})
        outputs = output_artifacts or []
        source_fingerprint = manifest["source_fingerprint"]
        if (
            resume
            and previous.get("status") == "complete"
            and outputs
            and previous.get("source_fingerprint", {}).get("sha256")
            == source_fingerprint["sha256"]
            and all(self.artifact_valid(name) for name in outputs)
        ):
            return previous, False
        now = _now()
        job = {
            "id": identifier,
            "type": job_type,
            "provider": provider,
            "model": model,
            "parameters": _sanitize(parameters or {}),
            "status": "running",
            "attempt_count": int(previous.get("attempt_count", 0)) + 1,
            "created_at": previous.get("created_at", now),
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
            "input_paths": [_sanitize(str(path)) for path in (input_paths or [self.recording])],
            "output_paths": [_sanitize(str(path)) for path in (output_paths or [])],
            "output_artifacts": outputs,
            "source_fingerprint": source_fingerprint,
            "last_error": None,
            "retryable": retryable,
            "units": previous.get("units", {}),
        }
        payload["jobs"][identifier] = job
        payload["updated_at"] = now
        _atomic_write_json(self.jobs_path, payload)
        return job, True

    def complete_job(self, job_type: str, provider: str) -> dict[str, Any]:
        identifier = self.job_id(job_type, provider)
        job = self.jobs().get("jobs", {}).get(identifier)
        if job is None:
            raise KeyError(f"Unknown job: {identifier}")
        invalid = [
            name for name in job.get("output_artifacts", []) if not self.artifact_valid(name)
        ]
        if invalid:
            raise ValueError(
                f"Cannot complete {identifier}; invalid artifacts: {', '.join(invalid)}"
            )
        return self._finish_job(job_type, provider, status="complete")

    def fail_job(
        self, job_type: str, provider: str, error: Exception | str, *, retryable: bool = True
    ) -> dict[str, Any]:
        return self._finish_job(
            job_type,
            provider,
            status="failed",
            error=_sanitize(str(error)),
            retryable=retryable,
        )

    def _finish_job(
        self,
        job_type: str,
        provider: str,
        *,
        status: str,
        error: str | None = None,
        retryable: bool | None = None,
    ) -> dict[str, Any]:
        payload = self.jobs()
        identifier = self.job_id(job_type, provider)
        if identifier not in payload["jobs"]:
            raise KeyError(f"Unknown job: {identifier}")
        now = _now()
        job = payload["jobs"][identifier]
        job.update(
            {
                "status": status,
                "updated_at": now,
                "completed_at": now if status == "complete" else None,
                "last_error": error,
            }
        )
        if retryable is not None:
            job["retryable"] = retryable
        payload["updated_at"] = now
        _atomic_write_json(self.jobs_path, payload)
        return job

    def begin_unit(
        self,
        job_type: str,
        provider: str,
        unit_id: str,
        *,
        parameters: dict[str, Any] | None = None,
        output_artifacts: list[str] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Start one resumable job unit, skipping checksum-valid completed work."""
        payload = self.jobs()
        identifier = self.job_id(job_type, provider)
        if identifier not in payload["jobs"]:
            raise KeyError(f"Unknown job: {identifier}")
        job = payload["jobs"][identifier]
        units = job.setdefault("units", {})
        previous = units.get(unit_id, {})
        outputs = output_artifacts or []
        if (
            previous.get("status") == "complete"
            and outputs
            and all(self.artifact_valid(name) for name in outputs)
        ):
            return previous, False
        now = _now()
        unit = {
            "id": unit_id,
            "status": "running",
            "attempt_count": int(previous.get("attempt_count", 0)) + 1,
            "created_at": previous.get("created_at", now),
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
            "parameters": _sanitize(parameters or {}),
            "output_artifacts": outputs,
            "last_error": None,
            "retryable": True,
        }
        units[unit_id] = unit
        job["updated_at"] = now
        payload["updated_at"] = now
        _atomic_write_json(self.jobs_path, payload)
        return unit, True

    def complete_unit(self, job_type: str, provider: str, unit_id: str) -> dict[str, Any]:
        return self._finish_unit(job_type, provider, unit_id, status="complete")

    def fail_unit(
        self,
        job_type: str,
        provider: str,
        unit_id: str,
        error: Exception | str,
        *,
        retryable: bool = True,
    ) -> dict[str, Any]:
        return self._finish_unit(
            job_type,
            provider,
            unit_id,
            status="failed",
            error=_sanitize(str(error)),
            retryable=retryable,
        )

    def _finish_unit(
        self,
        job_type: str,
        provider: str,
        unit_id: str,
        *,
        status: str,
        error: str | None = None,
        retryable: bool | None = None,
    ) -> dict[str, Any]:
        payload = self.jobs()
        identifier = self.job_id(job_type, provider)
        try:
            job = payload["jobs"][identifier]
            unit = job["units"][unit_id]
        except KeyError as missing:
            raise KeyError(f"Unknown job unit: {identifier}/{unit_id}") from missing
        if status == "complete":
            invalid = [
                name for name in unit.get("output_artifacts", []) if not self.artifact_valid(name)
            ]
            if invalid:
                raise ValueError(
                    f"Cannot complete {identifier}/{unit_id}; invalid artifacts: "
                    f"{', '.join(invalid)}"
                )
        now = _now()
        unit.update(
            {
                "status": status,
                "updated_at": now,
                "completed_at": now if status == "complete" else None,
                "last_error": error,
            }
        )
        if retryable is not None:
            unit["retryable"] = retryable
        job["updated_at"] = now
        payload["updated_at"] = now
        _atomic_write_json(self.jobs_path, payload)
        return unit

    def retry_failed(self, job_type: str | None = None) -> list[str]:
        """Reset retryable failed jobs to pending and return their IDs."""
        payload = self.jobs()
        retried = []
        now = _now()
        for identifier, job in payload["jobs"].items():
            if job.get("status") != "failed" or not job.get("retryable", False):
                continue
            if job_type and job.get("type") != job_type and identifier != job_type:
                continue
            job.update({"status": "pending", "updated_at": now})
            for unit in job.get("units", {}).values():
                if unit.get("status") == "failed" and unit.get("retryable", False):
                    unit.update({"status": "pending", "updated_at": now})
            retried.append(identifier)
        if retried:
            payload["updated_at"] = now
            _atomic_write_json(self.jobs_path, payload)
        return retried
