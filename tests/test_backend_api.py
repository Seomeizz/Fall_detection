from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep unit tests independent of a physical webcam or a trained checkpoint.
os.environ["FALLGUARD_AUTO_START_CAMERA"] = "false"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_status_and_events_contract() -> None:
    with TestClient(app) as client:
        status = client.get("/api/status")
        events = client.get("/api/events?limit=5")
    assert status.status_code == 200
    assert "camera" in status.json()
    assert "ai" in status.json()
    assert status.json()["stream_url"] == "/stream.mjpg"
    assert events.status_code == 200
    assert isinstance(events.json()["items"], list)
