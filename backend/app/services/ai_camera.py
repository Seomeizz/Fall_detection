from __future__ import annotations

import importlib
import logging
import math
import os
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


class AICameraEngine:
    """Owns the laptop webcam, AI inference loop and latest annotated JPEG frame.

    It reuses the model/preprocessing/fall-event functions from scripts/run_demo.py
    so the service does not define a second, drifting inference pipeline.
    """

    def __init__(self, settings: Settings, on_event: Callable[[dict[str, Any]], None]):
        self.settings = settings
        self.on_event = on_event
        self._state_lock = threading.RLock()
        self._frame_condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._jpeg: bytes | None = None
        self._jpeg_version = 0
        self._started_monotonic: float | None = None
        self._last_event_by_track: dict[int, float] = {}
        self._status: dict[str, Any] = {
            "ok": True,
            "camera": "stopped",
            "ai": "stopped",
            "model": "Temporal Transformer 8-class",
            "device": "not loaded",
            "fps": 0.0,
            "people_count": 0,
            "predictions": [],
            "last_error": None,
            "last_event": None,
            "uptime_seconds": 0,
            "checkpoint": str(settings.checkpoint),
        }

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            result = dict(self._status)
            result["predictions"] = [dict(p) for p in self._status.get("predictions", [])]
            if self._started_monotonic is not None:
                result["uptime_seconds"] = int(time.monotonic() - self._started_monotonic)
            result["stream_url"] = "/stream.mjpg"
            return result

    def start(self) -> dict[str, Any]:
        with self._state_lock:
            if self._thread and self._thread.is_alive():
                return {"started": False, "message": "AI camera worker đang chạy."}
            self._stop_event.clear()
            self._started_monotonic = time.monotonic()
            # Clear the previous session's image so clients never receive a stale frame after restart.
            with self._frame_condition:
                self._jpeg = None
                self._jpeg_version += 1
                self._frame_condition.notify_all()
            self._status.update({
                "ok": True,
                "camera": "starting",
                "ai": "loading",
                "last_error": None,
                "checkpoint": str(self.settings.checkpoint),
            })
            self._thread = threading.Thread(target=self._worker, name="fallguard-ai-camera", daemon=True)
            self._thread.start()
        return {"started": True, "message": "Đã yêu cầu khởi động camera và AI."}

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        # Wake stream generators immediately so stopped streams can close cleanly.
        with self._frame_condition:
            self._frame_condition.notify_all()
        with self._state_lock:
            cap = self._capture
            thread = self._thread
            self._status.update({"camera": "stopping", "ai": "stopping"})
        if cap is not None:
            try:
                cap.release()
            except Exception:
                logger.debug("Error releasing camera while stopping", exc_info=True)
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        with self._state_lock:
            if not self._thread or not self._thread.is_alive():
                self._status.update({"camera": "stopped", "ai": "stopped", "people_count": 0})
        return {"stopped": True, "message": "Đã gửi yêu cầu dừng camera."}

    def wait_for_frame(self, last_version: int, timeout: float = 1.0) -> tuple[int, bytes | None]:
        """Block until a new JPEG frame exists or timeout expires."""
        with self._frame_condition:
            if self._jpeg_version <= last_version and not self._stop_event.is_set():
                self._frame_condition.wait(timeout=timeout)
            return self._jpeg_version, self._jpeg

    def _publish_frame(self, frame: np.ndarray) -> None:
        quality = int(max(40, min(95, self.settings.jpeg_quality)))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return
        with self._frame_condition:
            self._jpeg = encoded.tobytes()
            self._jpeg_version += 1
            self._frame_condition.notify_all()

    @staticmethod
    def _new_track_state(track_id: int, clip_frames: int, fps: float) -> dict[str, Any]:
        return {
            "track_id": int(track_id),
            "xy_buffer": deque(maxlen=clip_frames),
            "conf_buffer": deque(maxlen=clip_frames),
            "found_buffer": deque(maxlen=clip_frames),
            "prob_history": deque(maxlen=3),
            "model_fall_probs": deque(maxlen=3),
            "current_probs": None,
            "current_label": "collecting pose",
            "frames_elapsed": 0,
            "last_seen_frame": 0,
            "window_number": 0,
            "geometry_history": deque(maxlen=max(60, int(round(fps * 2.0)))),
            "candidate_history": deque(maxlen=5),
            "baseline_shapes": deque(maxlen=max(30, int(round(fps * 1.2)))),
            "baseline_aspect": None,
            "baseline_angle": None,
            "event_active": False,
            "event_start_frame": None,
            "last_event_cleared_frame": None,
            "event_min_until": 0,
            "recovery_frames": 0,
            "max_event_score": 0.0,
            "event_debug": {},
        }

    def _event_args(self) -> SimpleNamespace:
        # Keep the same heuristic defaults as scripts/run_demo.py.
        return SimpleNamespace(
            event_lag_sec=0.45,
            event_hold_sec=1.5,
            recovery_sec=0.65,
            recovery_max_aspect=0.90,
            recovery_max_torso_angle=52.0,
            min_fall_score=0.48,
            event_move_threshold=0.18,
            event_hip_drop_threshold=0.12,
            event_angle_threshold=22.0,
            event_aspect_ratio=1.28,
            event_height_ratio=0.78,
            event_horizontal_aspect=0.82,
            event_horizontal_angle=55.0,
            event_horizontal_aspect_strong=1.25,
            model_fall_threshold=0.62,
            model_fall_single_threshold=0.88,
            recovery_model_threshold=0.55,
        )

    def _import_demo_helpers(self):
        repo_root = self.settings.repo_root
        scripts_dir = repo_root / "scripts"
        src_dir = repo_root / "src"
        for path in (str(src_dir), str(scripts_dir)):
            if path not in sys.path:
                sys.path.insert(0, path)
        # Importing this script is safe: its CLI is guarded by if __name__ == '__main__'.
        return importlib.import_module("run_demo")

    def _worker(self) -> None:
        cap = None
        try:
            demo = self._import_demo_helpers()
            import torch
            from ultralytics import YOLO

            if not self.settings.checkpoint.is_file():
                raise FileNotFoundError(
                    f"Không tìm thấy checkpoint Transformer: {self.settings.checkpoint}. "
                    "Hãy tải best_transformer_8class.pth từ Kaggle và đặt vào artifacts/ hoặc cấu hình FALLGUARD_CHECKPOINT."
                )

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model, class_names, seq_len, checkpoint = demo.load_checkpoint(self.settings.checkpoint, device)
            logger.info("Loaded Transformer checkpoint epoch=%s classes=%s", checkpoint.get("epoch", "unknown"), class_names)
            pose_model = YOLO(self.settings.pose_weights)

            self.settings.tracker_yaml.parent.mkdir(parents=True, exist_ok=True)
            tracker_cfg = demo.make_tracker_config(self.settings.tracker_yaml, track_buffer=120)

            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.settings.camera_index, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_height)
            cap.set(cv2.CAP_PROP_FPS, self.settings.camera_fps)
            if not cap.isOpened():
                raise RuntimeError(
                    f"Không mở được camera index {self.settings.camera_index}. "
                    "Đóng ứng dụng khác đang dùng camera, kiểm tra quyền Camera trên Windows hoặc đổi FALLGUARD_CAMERA_INDEX."
                )

            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 1e-3:
                fps = self.settings.camera_fps
            fps = max(1.0, fps)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.settings.camera_width
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.settings.camera_height
            clip_frames = max(seq_len, int(round(self.settings.clip_seconds * fps)))
            stride = max(1, self.settings.stride_frames)
            args = self._event_args()
            zero_xy = np.zeros((17, 2), dtype=np.float32)
            zero_conf = np.zeros((17,), dtype=np.float32)
            tracks: dict[int, dict[str, Any]] = {}
            frame_index = 0
            frame_count_since_fps = 0
            fps_clock = time.monotonic()
            processed_fps = 0.0
            device_name = str(device)

            with self._state_lock:
                self._capture = cap
                self._status.update({
                    "ok": True,
                    "camera": "online",
                    "ai": "running",
                    "model": "Temporal Transformer 8-class",
                    "device": device_name,
                    "width": width,
                    "height": height,
                    "source": f"Laptop camera index {self.settings.camera_index}",
                    "last_error": None,
                })

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Không đọc được frame từ camera laptop.")
                if self.settings.mirror_camera:
                    frame = cv2.flip(frame, 1)
                frame_index += 1
                result = pose_model.track(
                    frame,
                    persist=True,
                    tracker=str(tracker_cfg),
                    verbose=False,
                    conf=self.settings.detection_confidence,
                    imgsz=self.settings.imgsz,
                    device=device_name,
                )[0]
                people, draw_items = demo.extract_tracked_people(result)
                seen_ids = set(people.keys())
                stale_ids = []
                for tid, state in list(tracks.items()):
                    if tid not in seen_ids:
                        state["xy_buffer"].append(zero_xy.copy())
                        state["conf_buffer"].append(zero_conf.copy())
                        state["found_buffer"].append(False)
                        state["frames_elapsed"] += 1
                    if frame_index - state["last_seen_frame"] > 180:
                        stale_ids.append(tid)
                for tid in stale_ids:
                    del tracks[tid]

                for tid, (xy, conf, bbox) in people.items():
                    if tid not in tracks:
                        tracks[tid] = self._new_track_state(tid, clip_frames, fps)
                    state = tracks[tid]
                    state["xy_buffer"].append(xy.copy())
                    state["conf_buffer"].append(conf.copy())
                    state["found_buffer"].append(True)
                    state["frames_elapsed"] += 1
                    state["last_seen_frame"] = frame_index
                    has_clip = len(state["xy_buffer"]) == clip_frames
                    should_infer = (
                        has_clip
                        and (state["frames_elapsed"] - clip_frames) % stride == 0
                        and sum(state["found_buffer"]) >= int(0.70 * clip_frames)
                    )
                    if should_infer:
                        clip_xy = np.stack(state["xy_buffer"], axis=0)
                        clip_conf = np.stack(state["conf_buffer"], axis=0)
                        xy_sampled, conf_sampled = demo.resample_clip(clip_xy, clip_conf, seq_len)
                        features = demo.common_window_normalize(xy_sampled, conf_sampled)
                        tensor = torch.from_numpy(features).unsqueeze(0).to(device)
                        with torch.inference_mode():
                            logits = model(tensor)
                            probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
                        state["prob_history"].append(probs)
                        state["current_probs"] = np.mean(np.stack(state["prob_history"], axis=0), axis=0)
                        state["current_label"] = str(class_names[int(np.argmax(state["current_probs"]))])
                        state["model_fall_probs"].append(
                            demo.get_class_probability(state["current_probs"], class_names, "falling")
                        )
                        state["window_number"] += 1

                    debug = demo.update_fall_event(
                        state, xy, conf, bbox, frame, frame_index, fps, class_names, args
                    )
                    state["event_debug"] = debug
                    if debug.get("event_triggered"):
                        now_mono = time.monotonic()
                        last_event = self._last_event_by_track.get(int(tid), 0.0)
                        if now_mono - last_event >= self.settings.event_cooldown_seconds:
                            self._last_event_by_track[int(tid)] = now_mono
                            p_fall = float(debug.get("p_falling", 0.0))
                            event = {
                                "id": uuid.uuid4().hex,
                                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                                "camera_id": f"laptop-camera-{self.settings.camera_index}",
                                "track_id": int(tid),
                                "label": "falling",
                                "confidence": round(p_fall, 6),
                                "event_score": round(float(debug.get("event_score_max", debug.get("event_score", 0.0))), 6),
                                "confirmed": True,
                                "acknowledged": False,
                                "reason": str(debug.get("event_reason", "temporal_geometry_guard")),
                            }
                            with self._state_lock:
                                self._status["last_event"] = event
                            try:
                                self.on_event(event)
                            except Exception:
                                logger.exception("Failed to publish fall event")

                predictions: list[dict[str, Any]] = []
                for tid, xy, conf, bbox in draw_items:
                    if tid is not None and tid in tracks:
                        state = tracks[tid]
                        debug = state.get("event_debug", {})
                        current_probs = state.get("current_probs")
                        p_fall = float(debug.get("p_falling", 0.0))
                        model_conf = float(np.max(current_probs)) if current_probs is not None else 0.0
                        if state["event_active"]:
                            label = f"ID {tid} | FALL ALERT"
                            is_falling = True
                            status_label = "FALL_ALERT"
                        elif p_fall >= args.model_fall_threshold and bool(debug.get("horizontal_pose_now", False)):
                            label = f"ID {tid} | FALLING? {p_fall:.0%}"
                            is_falling = True
                            status_label = "FALLING_CANDIDATE"
                        elif current_probs is None:
                            label = f"ID {tid} | PROCESSING"
                            is_falling = False
                            status_label = "PROCESSING"
                        else:
                            label = f"ID {tid} | {state['current_label'].upper()} {model_conf:.0%}"
                            is_falling = False
                            status_label = state["current_label"].upper()
                        demo.draw_box(frame, bbox, label, is_falling)
                        predictions.append({
                            "track_id": int(tid),
                            "label": str(state["current_label"]),
                            "confidence": round(model_conf, 6),
                            "p_falling": round(p_fall, 6),
                            "fall_alert": bool(state["event_active"]),
                            "status": status_label,
                        })
                    else:
                        demo.draw_box(frame, bbox, "DETECTING", False)

                frame_count_since_fps += 1
                elapsed = time.monotonic() - fps_clock
                if elapsed >= 2.0:
                    processed_fps = frame_count_since_fps / elapsed
                    frame_count_since_fps = 0
                    fps_clock = time.monotonic()
                self._publish_frame(frame)
                with self._state_lock:
                    self._status.update({
                        "ok": True,
                        "camera": "online",
                        "ai": "running",
                        "device": device_name,
                        "fps": round(processed_fps, 2),
                        "people_count": len(people),
                        "predictions": predictions,
                        "last_error": None,
                    })

        except Exception as exc:
            if self._stop_event.is_set():
                logger.info("AI camera worker stopped on request")
                with self._state_lock:
                    self._status.update({"ok": True, "camera": "stopped", "ai": "stopped", "people_count": 0, "last_error": None})
            else:
                logger.exception("AI camera worker failed")
                with self._state_lock:
                    self._status.update({
                        "ok": False,
                        "camera": "error",
                        "ai": "error",
                        "last_error": f"{type(exc).__name__}: {exc}",
                    })
                try:
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "FALLGUARD BACKEND ERROR", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(error_frame, str(exc)[:75], (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    self._publish_frame(error_frame)
                except Exception:
                    logger.debug("Unable to publish error frame", exc_info=True)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            with self._state_lock:
                self._capture = None
                if self._stop_event.is_set():
                    self._status.update({"camera": "stopped", "ai": "stopped", "people_count": 0})
                elif self._status.get("ai") != "error":
                    self._status.update({"camera": "offline", "ai": "stopped", "people_count": 0})
            with self._frame_condition:
                self._frame_condition.notify_all()
