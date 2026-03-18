"""Minimal pub/sub event system for Murmur plugins."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

_listeners: dict[str, list[Callable[..., Any]]] = defaultdict(list)


def on(event: str, callback: Callable[..., Any]) -> None:
    """Register a callback for an event."""
    _listeners[event].append(callback)


def emit(event: str, **payload: Any) -> None:
    """Fire all callbacks registered for an event."""
    for callback in _listeners[event]:
        callback(**payload)


def clear() -> None:
    """Remove all listeners. Mainly useful for testing."""
    _listeners.clear()
