from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT / "backend" / ".env", override=False)
except ImportError:
    pass


def _path_env(name: str, default: Path) -> Path:
    raw = Path(os.getenv(name, str(default))).expanduser()
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _pose_weights_value() -> str:
    """Prefer a local weights file bundled in scripts/, otherwise let Ultralytics download by name."""
    raw = os.getenv("FALLGUARD_POSE_WEIGHTS", "yolo11l-pose.pt").strip()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    rooted = (REPO_ROOT / candidate).resolve()
    if rooted.is_file():
        return str(rooted)
    in_scripts = (REPO_ROOT / "scripts" / candidate.name).resolve()
    if in_scripts.is_file() and candidate.parent == Path("."):
        return str(in_scripts)
    # Returning the bare model name lets Ultralytics download it if not found locally.
    return raw


@dataclass(frozen=True)
class Settings:
    repo_root: Path = REPO_ROOT
    host: str = os.getenv("FALLGUARD_HOST", "0.0.0.0")
    port: int = int(os.getenv("FALLGUARD_PORT", "8000"))
    camera_index: int = int(os.getenv("FALLGUARD_CAMERA_INDEX", "0"))
    camera_width: int = int(os.getenv("FALLGUARD_CAMERA_WIDTH", "640"))
    camera_height: int = int(os.getenv("FALLGUARD_CAMERA_HEIGHT", "480"))
    camera_fps: float = float(os.getenv("FALLGUARD_CAMERA_FPS", "30"))
    mirror_camera: bool = _env_bool("FALLGUARD_MIRROR_CAMERA", True)
    checkpoint: Path = _path_env("FALLGUARD_CHECKPOINT", REPO_ROOT / "artifacts" / "best_transformer_8class.pth")
    pose_weights: str = _pose_weights_value()
    imgsz: int = int(os.getenv("FALLGUARD_IMGSZ", "960"))
    detection_confidence: float = float(os.getenv("FALLGUARD_DETECTION_CONF", "0.10"))
    clip_seconds: float = float(os.getenv("FALLGUARD_CLIP_SECONDS", "2.5"))
    stride_frames: int = int(os.getenv("FALLGUARD_STRIDE_FRAMES", "15"))
    jpeg_quality: int = int(os.getenv("FALLGUARD_JPEG_QUALITY", "78"))
    auto_start_camera: bool = _env_bool("FALLGUARD_AUTO_START_CAMERA", True)
    event_log: Path = _path_env("FALLGUARD_EVENT_LOG", REPO_ROOT / "backend" / "runtime" / "events.json")
    tracker_yaml: Path = REPO_ROOT / "backend" / "runtime" / "botsort_custom_no_reid.yaml"
    event_cooldown_seconds: float = float(os.getenv("FALLGUARD_EVENT_COOLDOWN_SECONDS", "20"))


settings = Settings()
