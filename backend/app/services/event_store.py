from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class EventStore:
    """Small JSON-backed event log for an MVP. Replace with SQLite for production."""

    def __init__(self, path: Path, max_events: int = 300) -> None:
        self.path = path
        self.max_events = max_events
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._items = [item for item in data if isinstance(item, dict)][-self.max_events :]
        except (OSError, json.JSONDecodeError):
            # Preserve the corrupted file for manual inspection; start a fresh in-memory log.
            self._items = []

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._items[-self.max_events :], ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._items.insert(0, dict(event))
            self._items = self._items[: self.max_events]
            self._save()
            return dict(self._items[0])

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._items[: max(1, min(limit, self.max_events))]]

    def acknowledge(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            for event in self._items:
                if event.get("id") == event_id:
                    event["acknowledged"] = True
                    self._save()
                    return dict(event)
        return None
