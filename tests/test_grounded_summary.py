"""Tests for transcript cleanup, grounded citation validation, and map-reduce."""

from murmur.grounded_summary import (
    clean_transcript,
    generate_grounded_summary,
    render_summary,
    validate_and_ground_summary,
)


def _source(segments):
    return {"provider": "openai", "model": "test", "segments": segments}


def test_cleanup_preserves_words_and_surfaces_uncertainty():
    cleaned = clean_transcript(
        _source(
            [
                {
                    "id": "segment-1",
                    "start": 1,
                    "end": 3,
                    "speaker": "unknown:remote:chunk-0000:A",
                    "text": "  launch   is [inaudible] Friday  ",
                }
            ]
        )
    )
    segment = cleaned["segments"][0]
    assert segment["text"] == "launch is [inaudible] Friday"
    assert segment["raw_text"] == "  launch   is [inaudible] Friday  "
    assert segment["uncertainty_reasons"] == [
        "unresolved_speaker",
        "uncertain_transcription",
    ]


def test_grounding_resolves_valid_citations_and_rejects_unsupported_claims():
    cleaned = clean_transcript(
        _source(
            [
                {
                    "id": "segment-1",
                    "start": 10,
                    "end": 14,
                    "speaker": "Rohan",
                    "text": "I will send the launch plan Friday.",
                }
            ]
        )
    )
    candidate = {
        "title": "Launch planning",
        "attendees": [{"name": "Rohan", "segment_ids": ["segment-1"]}],
        "executive_summary": [
            {"text": "A launch plan was discussed.", "segment_ids": ["segment-1"]}
        ],
        "topics": [{"text": "Unsupported budget topic", "segment_ids": []}],
        "decisions": [
            {
                "text": "The plan will be sent Friday.",
                "commitment": "explicit",
                "confidence": 0.95,
                "segment_ids": ["segment-1"],
            }
        ],
        "open_questions": [],
        "action_items": [
            {
                "task": "Send the launch plan",
                "owner": "Rohan",
                "deadline": "Friday",
                "commitment": "explicit",
                "confidence": 0.98,
                "segment_ids": ["segment-1"],
            }
        ],
    }
    summary = validate_and_ground_summary(candidate, cleaned)
    action = summary["action_items"][0]
    assert action["citations"][0]["timestamp"] == "00:00:10"
    assert action["citations"][0]["segment_id"] == "segment-1"
    assert action["source_excerpts"] == ["I will send the launch plan Friday."]
    assert summary["topics"] == []
    assert summary["rejected_claims"] == [
        {
            "field": "topics",
            "reason": "no_valid_source_citation",
            "value": "Unsupported budget topic",
        }
    ]
    markdown = render_summary(summary)
    assert "transcript.cleaned.md#segment-1" in markdown


def test_no_source_means_no_model_call_and_no_claims():
    called = False

    def generator(stage, content, glossary):
        nonlocal called
        called = True
        return {"title": "hallucinated"}

    summary, metadata = generate_grounded_summary(clean_transcript(_source([])), generator)
    assert called is False
    assert metadata["strategy"] == "no-source"
    assert summary["decisions"] == []
    assert summary["action_items"] == []


def test_long_transcript_uses_map_reduce_and_retains_source_ids():
    cleaned = clean_transcript(
        _source(
            [
                {
                    "id": f"segment-{index}",
                    "start": index,
                    "end": index + 1,
                    "speaker": "Abby",
                    "text": "discussion " * 15,
                }
                for index in range(4)
            ]
        )
    )
    stages = []

    def generator(stage, content, glossary):
        stages.append(stage)
        if stage == "reduce":
            return {
                "title": "Long call",
                "attendees": [],
                "executive_summary": [
                    {"text": "Discussion occurred.", "segment_ids": ["segment-0"]}
                ],
                "topics": [],
                "decisions": [],
                "open_questions": [],
                "action_items": [],
            }
        return {"title": "Partial", "topics": []}

    summary, metadata = generate_grounded_summary(
        cleaned, generator, glossary={"K8s": "Kubernetes"}, max_characters=200
    )
    assert stages.count("map") == 4
    assert stages[-1] == "reduce"
    assert metadata["strategy"] == "map-reduce"
    assert summary["executive_summary"][0]["segment_ids"] == ["segment-0"]
