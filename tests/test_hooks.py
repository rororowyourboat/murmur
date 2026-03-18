"""Tests for the pub/sub event system."""

from murmur import hooks


def setup_function():
    hooks.clear()


def test_on_and_emit():
    received = []
    hooks.on("test_event", lambda **kw: received.append(kw))

    hooks.emit("test_event", foo="bar", n=42)

    assert received == [{"foo": "bar", "n": 42}]


def test_multiple_listeners():
    calls = []
    hooks.on("evt", lambda **kw: calls.append("a"))
    hooks.on("evt", lambda **kw: calls.append("b"))

    hooks.emit("evt")

    assert calls == ["a", "b"]


def test_emit_unknown_event_is_noop():
    hooks.emit("nonexistent", data=1)


def test_clear():
    hooks.on("evt", lambda **kw: None)
    hooks.clear()

    # Should not raise even though listener was cleared
    hooks.emit("evt")


def test_separate_events_are_independent():
    a_calls = []
    b_calls = []
    hooks.on("a", lambda **kw: a_calls.append(1))
    hooks.on("b", lambda **kw: b_calls.append(1))

    hooks.emit("a")

    assert len(a_calls) == 1
    assert len(b_calls) == 0
