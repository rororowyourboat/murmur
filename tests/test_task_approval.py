"""Tests for preview-first, explicit task approval."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import click
import pytest

from murmur.artifacts import ArtifactStore, fingerprint_file
from murmur.plugins.tasks_extract import _apply_task_preview, _extract_tasks


def _summary(tmp_path):
    recording = tmp_path / "meeting.mka"
    recording.write_bytes(b"recording")
    store = ArtifactStore(recording)
    store.ensure_manifest()
    summary = store.write_json(
        "summary.json",
        {
            "action_items": [
                {
                    "task": "Send the plan",
                    "segment_ids": ["segment-1"],
                    "citations": [{"segment_id": "segment-1"}],
                }
            ]
        },
    )
    return recording, store, summary


def _task(title, segment_ids):
    return SimpleNamespace(
        title=title,
        owner="Rohan",
        deadline="Friday",
        priority="normal",
        project="Launch",
        source_excerpt="I will send the plan Friday.",
        confidence=0.95,
        commitment="explicit",
        source_segment_ids=segment_ids,
    )


def test_extraction_writes_preview_and_filters_uncited_tasks_without_mutation(tmp_path):
    _recording, store, summary = _summary(tmp_path)
    analysis = SimpleNamespace(
        new_tasks=[_task("Send the plan", ["segment-1"]), _task("Invented task", [])],
        blockers_raised=[],
        blockers_resolved=[],
    )
    fake_dspy = ModuleType("dspy")
    fake_dspy.LM = lambda *args, **kwargs: object()
    fake_dspy.JSONAdapter = lambda: object()
    fake_dspy.configure = lambda **kwargs: None

    def extractor(**kwargs):
        return SimpleNamespace(analysis=analysis)

    with (
        patch.dict(sys.modules, {"dspy": fake_dspy}),
        patch("murmur.plugins.tasks_extract._build_extractor", return_value=extractor),
        patch("murmur.plugins.tasks.load_tasks", return_value=[]),
        patch("murmur.plugins.tasks.save_tasks") as save,
    ):
        result = _extract_tasks(summary, model="test/model")

    assert [task.title for task in result.new_tasks] == ["Send the plan"]
    save.assert_not_called()
    preview = json.loads(store.path("tasks.preview.json").read_text())
    assert preview["approval_required"] is True
    assert preview["rejected_uncited_tasks"] == 1
    assert preview["new_tasks"][0]["source_segment_ids"] == ["segment-1"]


def test_apply_requires_saved_unchanged_preview_and_is_idempotent(tmp_path):
    _recording, store, summary = _summary(tmp_path)
    preview = {
        "schema_version": 1,
        "kind": "task_change_preview",
        "preview_id": "preview-1",
        "source": str(summary),
        "source_fingerprint": fingerprint_file(summary),
        "approval_required": True,
        "applied_at": None,
        "new_tasks": [
            {
                "title": "Send the plan",
                "owner": "Rohan",
                "deadline": "Friday",
                "priority": "normal",
                "project": "Launch",
                "commitment": "explicit",
                "source_segment_ids": ["segment-1"],
            }
        ],
    }
    store.write_json("tasks.preview.json", preview)

    with (
        patch("murmur.plugins.tasks.load_tasks", return_value=[]),
        patch("murmur.plugins.tasks.save_tasks") as save,
    ):
        assert _apply_task_preview(summary) == 1
        assert _apply_task_preview(summary) == 0

    save.assert_called_once()
    saved_tasks = save.call_args.args[0]
    assert saved_tasks[0].title == "Send the plan"
    assert "explicit" in saved_tasks[0].tags
    applied = json.loads(store.path("tasks.preview.json").read_text())
    assert applied["approval_required"] is False
    assert applied["applied_at"]


def test_apply_rejects_changed_source(tmp_path):
    _recording, store, summary = _summary(tmp_path)
    store.write_json(
        "tasks.preview.json",
        {
            "preview_id": "preview-1",
            "source": str(summary),
            "source_fingerprint": fingerprint_file(summary),
            "applied_at": None,
            "new_tasks": [],
        },
    )
    summary.write_text('{"changed": true}')

    with pytest.raises(click.ClickException, match="Source changed"):
        _apply_task_preview(summary)
