"""Run FallGuard with camera release initiated as soon as Ctrl+C is received."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import FrameType

import uvicorn

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app, engine  # noqa: E402

logger = logging.getLogger("fallguard.backend.runner")


class FallGuardServer(uvicorn.Server):
    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        first_shutdown = not self.should_exit
        # Set should_exit first; a second Ctrl+C can still force-exit Uvicorn.
        super().handle_exit(sig, frame)
        if first_shutdown:
            logger.info("Shutdown requested; releasing webcam before waiting for MJPEG clients")
            try:
                engine.stop()
            except Exception:
                logger.exception("Could not stop camera cleanly during shutdown")


def main() -> None:
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
    FallGuardServer(config).run()


if __name__ == "__main__":
    main()
