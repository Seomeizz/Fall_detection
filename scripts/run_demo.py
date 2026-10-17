"""Test video với model Transformer 8 lớp từ notebook NTU60 (clip-level).

Pipeline: YOLO11n-Pose (or another compatible YOLO Pose model) + BoT-SORT -> keypoints COCO-17 theo từng người (track ID)
-> gom ~CLIP_SECONDS giây gần nhất -> lấy mẫu đều thành 30 frame (giống notebook)
-> chuẩn hóa -> best_transformer_8class.pth -> 8 lớp:
walking, standing, sitting, bending, falling, getting_up, stumbling, abnormal_movement.

Mỗi track có thêm state machine hình học để xác nhận cảnh báo té ngã
(hộp ĐỎ = fall alert, XANH = bình thường).
Chỉ lưu video MP4 đầu ra; không xuất file CSV. Ngưỡng hình học là heuristic, cần kiểm chứng trên video thật có nhãn.
"""

from __future__ import annotations

import argparse
import sys
import math
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path

# These values match the training notebook.
# Shared definitions are imported from the project package so training and inference
# use one source of truth for class order, architecture, and preprocessing.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from fall_detection.labels import (
    CLASS_NAMES as DEFAULT_CLASSES, FEATURE_DIM, NUM_CLASSES, SEQUENCE_LENGTH,
)
from fall_detection.model import TemporalTransformerClassifier
from fall_detection.preprocessing import (
    KEYPOINT_CONF_THRESHOLD as CONF_THRESHOLD,
    MAX_NORMALIZED_COORD,
    common_clip_normalize as common_window_normalize,
    resample_full_clip as resample_clip,
)

WINDOW_SIZE = SEQUENCE_LENGTH
WINDOW_STRIDE = 15


def load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {path}\n"
            "Hãy tải best_transformer_8class.pth từ "
            "/kaggle/working/ntu60_8class_cliplevel_comparison/ rồi đặt cạnh file .py, "
            "hoặc dùng --checkpoint để chỉ định đường dẫn."
        )
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch cũ
        checkpoint = torch.load(path, map_location=device)

    model_name = checkpoint.get("model_name", "Transformer")
    if model_name != "Transformer":
        raise ValueError(f"Checkpoint là '{model_name}', script này chỉ nạp Transformer.")
    class_names = list(checkpoint.get("class_names", DEFAULT_CLASSES))
    if class_names != DEFAULT_CLASSES:
        raise ValueError(
            "Thứ tự class của checkpoint không khớp project. "
            f"Checkpoint={class_names}; expected={DEFAULT_CLASSES}"
        )
    seq_len = int(checkpoint.get("sequence_length", WINDOW_SIZE))
    feature_dim = int(checkpoint.get("feature_dim", FEATURE_DIM))
    # Hyperparameter kiến trúc không được lưu trong checkpoint: dùng đúng giá trị CONFIG của notebook.
    model = TemporalTransformerClassifier(
        feature_dim=feature_dim, d_model=128, nhead=4, num_layers=4,
        dim_feedforward=256, dropout=0.15, num_classes=len(class_names),
        window_size=seq_len,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, seq_len, checkpoint


def extract_tracked_people(result):
    """Return {track_id: (xy[17,2], conf[17], bbox[4])} for this frame."""
    people = {}
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return people, []

    boxes = result.boxes.xyxy.cpu().numpy()
    xy_all = result.keypoints.xy.cpu().numpy()
    if result.keypoints.conf is not None:
        conf_all = result.keypoints.conf.cpu().numpy()
    elif result.keypoints.data.shape[-1] >= 3:
        conf_all = result.keypoints.data[..., 2].cpu().numpy()
    else:
        conf_all = np.ones(xy_all.shape[:2], dtype=np.float32)

    ids = None
    if result.boxes.id is not None:
        ids = result.boxes.id.cpu().numpy().astype(int)

    # All detections are returned for drawing. Classification requires a stable track ID.
    draw_items = []
    for i in range(min(len(boxes), len(xy_all))):
        xy = np.asarray(xy_all[i], dtype=np.float32)
        conf = np.asarray(conf_all[i], dtype=np.float32)
        bbox = boxes[i].astype(int)
        track_id = int(ids[i]) if ids is not None and i < len(ids) else None
        draw_items.append((track_id, xy, conf, bbox))
        if track_id is not None and xy.shape == (17, 2):
            people[track_id] = (xy, conf, bbox)
    return people, draw_items


def get_class_probability(probs, class_names, target_name: str) -> float:
    """Safely get a class probability when class order comes from checkpoint metadata."""
    if probs is None:
        return 0.0
    for i, name in enumerate(class_names):
        if str(name).lower() == target_name.lower() and i < len(probs):
            return float(probs[i])
    return 0.0


def pose_geometry(xy, conf, bbox, frame_shape):
    """Extract view-relative geometry from one person's raw keypoints and box."""
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    aspect = bw / bh
    scale = max(1.0, math.sqrt(bw * bh))

    xy = np.asarray(xy, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    valid = np.isfinite(xy).all(axis=1) & (conf >= CONF_THRESHOLD)

    def joint_center(indices):
        valid_ids = [j for j in indices if j < len(valid) and valid[j]]
        if not valid_ids:
            return None
        return np.mean(xy[valid_ids], axis=0).astype(np.float32)

    shoulder = joint_center([5, 6])
    hip = joint_center([11, 12])
    torso_angle = float("nan")
    hip_y = float("nan")
    if hip is not None:
        hip_y = float(hip[1])
    if shoulder is not None and hip is not None:
        dx = float(shoulder[0] - hip[0])
        dy = float(shoulder[1] - hip[1])
        # 0 degrees = torso axis vertical in image; 90 = horizontal.
        torso_angle = float(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))

    return {
        "center_x": cx,
        "center_y": cy,
        "box_w": bw,
        "box_h": bh,
        "aspect": aspect,
        "scale": scale,
        "torso_angle": torso_angle,
        "hip_y": hip_y,
        "valid_joints": int(valid.sum()),
        "frame_w": int(frame_w),
        "frame_h": int(frame_h),
    }


def _finite_angle(value) -> bool:
    return value is not None and np.isfinite(value)


def update_fall_event(state, xy, conf, bbox, frame, frame_index, fps,
                      class_names, args):
    """Update per-track event state; return geometry diagnostics for on-screen status/debugging.

    The Transformer estimates posture/action probabilities. This state machine
    independently checks for a rapid per-track change in center/hip position and
    body shape. Two candidate frames within a short temporal vote are required to
    avoid firing on one noisy pose estimate.
    """
    g = pose_geometry(xy, conf, bbox, frame.shape)
    g["frame_index"] = frame_index
    hist = state["geometry_history"]
    hist.append(g)

    probs = state.get("current_probs")
    p_fall = get_class_probability(probs, class_names, "falling")
    p_standing = get_class_probability(probs, class_names, "standing")
    p_sitting = get_class_probability(probs, class_names, "sitting")
    p_getting_up = get_class_probability(probs, class_names, "getting_up")
    p_walking = get_class_probability(probs, class_names, "walking")
    p_stumbling = get_class_probability(probs, class_names, "stumbling")

    # A geometric check of the CURRENT pose. A single cue is not enough:
    # fast running can tilt the torso, and a wide pose can occur during gait.
    # Require BOTH a near-horizontal torso and a horizontally-wide person box.
    # A very wide box is allowed as a fallback when torso keypoints are noisy.
    angle_is_horizontal = (
        _finite_angle(g["torso_angle"])
        and g["torso_angle"] >= args.event_horizontal_angle
    )
    box_is_horizontal = g["aspect"] >= args.event_horizontal_aspect
    horizontal_pose_now = bool(
        (angle_is_horizontal and box_is_horizontal)
        or (g["aspect"] >= args.event_horizontal_aspect_strong
            and g["valid_joints"] >= 5)
    )
    # Running/leaning should still count as upright for recovery when the body
    # box is tall and the torso angle is not close to horizontal.
    upright_recovery_pose = (
        g["valid_joints"] >= 6
        and g["aspect"] <= args.recovery_max_aspect
        and (
            not _finite_angle(g["torso_angle"])
            or g["torso_angle"] <= args.recovery_max_torso_angle
        )
    )

    debug = {
        "event_score": 0.0,
        "geometry_candidate": False,
        "event_triggered": False,
        "center_shift_norm": 0.0,
        "hip_drop_norm": 0.0,
        "torso_angle_deg": g["torso_angle"],
        "torso_angle_change_deg": 0.0,
        "bbox_aspect_ratio": g["aspect"],
        "bbox_aspect_growth": 1.0,
        "bbox_height_ratio": 1.0,
        "event_reason": "",
        "horizontal_pose_now": bool(horizontal_pose_now),
        "upright_recovery_pose": bool(upright_recovery_pose),
    }

    lag_frames = max(3, int(round(fps * args.event_lag_sec)))
    target_frame = frame_index - lag_frames
    previous = None
    # Find the nearest observation at or before the target lag time.
    for item in reversed(hist):
        if item["frame_index"] <= target_frame:
            previous = item
            break

    score = 0.0
    candidate = False
    motion_cue = False
    shape_cue = False
    horizontal_evidence = horizontal_pose_now
    if previous is not None and frame_index - previous["frame_index"] <= max(lag_frames * 2, int(fps * 0.9)):
        center_delta = math.hypot(g["center_x"] - previous["center_x"],
                                  g["center_y"] - previous["center_y"])
        center_shift = center_delta / max(previous["scale"], 1.0)
        hip_drop = 0.0
        if _finite_angle(g["hip_y"]) and _finite_angle(previous["hip_y"]):
            hip_drop = (g["hip_y"] - previous["hip_y"]) / max(previous["scale"], previous["box_h"], 1.0)
        angle_change = 0.0
        if _finite_angle(g["torso_angle"]) and _finite_angle(previous["torso_angle"]):
            angle_change = abs(g["torso_angle"] - previous["torso_angle"])
        aspect_growth = g["aspect"] / max(previous["aspect"], 0.05)
        height_ratio = g["box_h"] / max(previous["box_h"], 1.0)

        move_by_center = center_shift >= args.event_move_threshold
        move_by_hip = hip_drop >= args.event_hip_drop_threshold
        angle_change_cue = angle_change >= args.event_angle_threshold
        aspect_change_cue = aspect_growth >= args.event_aspect_ratio or height_ratio <= args.event_height_ratio
        motion_cue = move_by_center or move_by_hip
        shape_cue = angle_change_cue or aspect_change_cue
        # Require direct evidence from the CURRENT geometry. Do not use a stale
        # Transformer lying probability plus ordinary running shape changes to
        # pretend the body is horizontal.
        horizontal_evidence = horizontal_pose_now

        if move_by_center:
            score += 0.28
        if move_by_hip:
            score += 0.22
        if angle_change_cue:
            score += 0.30
        if aspect_growth >= args.event_aspect_ratio:
            score += 0.24
        if height_ratio <= args.event_height_ratio:
            score += 0.20
        if horizontal_evidence:
            score += 0.08
        if p_fall >= 0.35:
            score += min(p_fall, 1.0) * 0.15
        score = float(min(score, 1.0))

        candidate = motion_cue and shape_cue and horizontal_evidence and score >= args.min_fall_score
        debug.update({
            "center_shift_norm": float(center_shift),
            "hip_drop_norm": float(hip_drop),
            "torso_angle_change_deg": float(angle_change),
            "bbox_aspect_growth": float(aspect_growth),
            "bbox_height_ratio": float(height_ratio),
        })
        reasons = []
        if move_by_center:
            reasons.append("center_move")
        if move_by_hip:
            reasons.append("hip_drop")
        if angle_change_cue:
            reasons.append("torso_angle_change")
        if aspect_growth >= args.event_aspect_ratio:
            reasons.append("box_widened")
        if height_ratio <= args.event_height_ratio:
            reasons.append("box_height_shrank")
        if p_fall >= 0.35:
            reasons.append("transformer_fall_support")
        debug["event_reason"] = "+".join(reasons)

    # A strong Transformer fall prediction can support the event, but low-confidence
    # class argmax alone cannot activate the alarm.
    fall_prob_history = state["model_fall_probs"]
    model_consensus = (
        len(fall_prob_history) >= 2
        and sum(float(p) >= args.model_fall_threshold for p in fall_prob_history) >= 2
    )
    model_candidate = (
        model_consensus and (motion_cue or shape_cue) and horizontal_evidence
    ) or (
        p_fall >= args.model_fall_single_threshold
        and (motion_cue or shape_cue)
        and horizontal_evidence
    )
    if model_candidate:
        score = max(score, min(p_fall, 1.0))
        if not debug["event_reason"]:
            debug["event_reason"] = "transformer_fall_consensus" if model_consensus else "transformer_high_probability"
    candidate = bool(candidate or model_candidate)
    state["candidate_history"].append(candidate)
    debug["event_score"] = float(score)
    debug["geometry_candidate"] = bool(candidate)

    # Build a robust pre-event reference from calm frames only. It is not updated
    # once the fall alert is active or during a candidate transition.
    if not state["event_active"] and not candidate and g["valid_joints"] >= 6:
        state["baseline_shapes"].append((g["aspect"], g["torso_angle"]))
        if state["baseline_shapes"]:
            aspects = [v[0] for v in state["baseline_shapes"] if np.isfinite(v[0])]
            angles = [v[1] for v in state["baseline_shapes"] if _finite_angle(v[1])]
            state["baseline_aspect"] = float(np.median(aspects)) if aspects else None
            state["baseline_angle"] = float(np.median(angles)) if angles else None

    # Two candidate frames in a 5-frame vote are required before confirming.
    if (not state["event_active"] and sum(state["candidate_history"]) >= 2):
        state["event_active"] = True
        state["event_start_frame"] = frame_index
        state["event_min_until"] = frame_index + int(round(args.event_hold_sec * fps))
        state["recovery_frames"] = 0
        state["max_event_score"] = score
        debug["event_triggered"] = True
        print(
            f"[FALL EVENT] ID={state['track_id']} at {frame_index / fps:.2f}s "
            f"score={score:.2f} evidence={debug['event_reason'] or 'model_consensus'}"
        )

    if state["event_active"]:
        state["max_event_score"] = max(state.get("max_event_score", 0.0), score)

        # Recovery is primarily geometry-based. The original logic required
        # matching the exact pre-fall baseline, a recovery class probability,
        # AND no candidate flag; those gates could keep red boxes active while
        # a child was already upright and walking. Require a stable upright pose
        # for a short interval instead, even if the classifier is still wrong.
        recovered_now = (
            frame_index >= state.get("event_min_until", frame_index + 1)
            and upright_recovery_pose
            and not horizontal_pose_now
        )
        if recovered_now:
            state["recovery_frames"] += 1
        else:
            state["recovery_frames"] = 0

        if state["recovery_frames"] >= max(1, int(round(args.recovery_sec * fps))):
            state["event_active"] = False
            state["candidate_history"].clear()
            state["recovery_frames"] = 0
            state["max_event_score"] = 0.0
            state["last_event_cleared_frame"] = frame_index
            print(f"[FALL RECOVERED] ID={state['track_id']} at {frame_index / fps:.2f}s")

    debug["fall_event_active"] = bool(state["event_active"])
    debug["event_score_max"] = float(state.get("max_event_score", 0.0))
    debug["p_falling"] = float(p_fall)
    debug["p_stumbling"] = float(p_stumbling)
    debug["baseline_aspect"] = state.get("baseline_aspect")
    debug["baseline_angle"] = state.get("baseline_angle")
    return debug


GREEN = (0, 200, 0)  # OpenCV BGR
RED = (0, 0, 255)


def make_tracker_config(config_path: Path, track_buffer: int) -> Path:
    """Write a BoT-SORT config using motion/geometry only (no ReID weights)."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.25
track_buffer: {int(track_buffer)}
match_thresh: 0.80
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.50
appearance_thresh: 0.80
with_reid: False
model: auto
"""
    config_path.write_text(text, encoding="utf-8")
    return config_path


def draw_box(frame, bbox, label: str, is_falling: bool):
    """Draw only a clean red/green person box and compact label."""
    color = RED if is_falling else GREEN
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 0.58, 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_top = max(0, y1 - th - baseline - 8)
        label_bottom = max(th + baseline + 4, y1)
        label_right = min(w - 1, x1 + tw + 10)
        cv2.rectangle(frame, (x1, label_top), (label_right, label_bottom), color, -1)
        cv2.putText(frame, label, (x1 + 5, label_bottom - baseline - 3),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(
        description="Multi-person tracking + 8-class NTU60 Transformer + horizontal-pose fall-event guard"
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", default=None, help="Đường dẫn video, ví dụ F:/video/test.mp4")
    source_group.add_argument("--camera", type=int, default=None, help="Chỉ số camera laptop, thường là 0; thử 1 nếu có nhiều camera")
    parser.add_argument("--camera-width", type=int, default=640, help="Chiều rộng yêu cầu khi dùng camera")
    parser.add_argument("--camera-height", type=int, default=480, help="Chiều cao yêu cầu khi dùng camera")
    parser.add_argument("--camera-fps", type=float, default=30.0, help="FPS dự phòng cho camera nếu driver không báo FPS")
    parser.add_argument("--checkpoint", default=str("../artifacts/best_transformer_8class.pth"))
    parser.add_argument("--clip-seconds", type=float, default=2.5,
                        help="Độ dài đoạn video (giây) được resample thành 30 frame; clip NTU dài ~2-4 s")
    parser.add_argument("--stride-frames", type=int, default=15,
                        help="Chạy suy luận mỗi N frame video")
    parser.add_argument("--pose-weights", default="yolo11l-pose.pt",
                        help="YOLO11l-Pose weights; downloaded automatically by Ultralytics if needed")
    parser.add_argument("--tracker", default="botsort",
                        help="'botsort' (default, no separate ReID model) or a tracker YAML path such as bytetrack.yaml")
    parser.add_argument("--track-buffer", type=int, default=120,
                        help="Keep lost tracks for this many frames in BoT-SORT (120 ~= 4 s at 30 FPS)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference size; increase for small/far people if hardware allows")
    parser.add_argument("--output", default="output_8class.mp4")
    parser.add_argument("--no-display", action="store_true", help="Không mở cửa sổ xem video")
    parser.add_argument(
        "--no-mirror-camera", action="store_true",
        help="Không lật ngang hình ảnh khi dùng camera; mặc định camera được lật ngang"
    )
    parser.add_argument("--conf", type=float, default=0.10,
                        help="Lower detection threshold helps recover weak/partly occluded people")
    parser.add_argument("--max-age", type=int, default=180,
                        help="Keep classifier state this many frames while a track is missing")
    # Fall-event fusion parameters. They are intentionally exposed for video validation.
    parser.add_argument("--event-lag-sec", type=float, default=0.45,
                        help="Compare current pose/box against this many seconds earlier")
    parser.add_argument("--event-hold-sec", type=float, default=1.5,
                        help="Minimum duration of a confirmed fall alert before recovery can clear it")
    parser.add_argument("--recovery-sec", type=float, default=0.65,
                        help="Continuous upright duration required to clear the fall alert")
    parser.add_argument("--recovery-max-aspect", type=float, default=0.90,
                        help="Maximum bbox width/height ratio considered upright for recovery")
    parser.add_argument("--recovery-max-torso-angle", type=float, default=52.0,
                        help="Maximum shoulder-hip axis angle (degrees from vertical) considered upright")
    parser.add_argument("--min-fall-score", type=float, default=0.48,
                        help="Minimum combined geometry score for a fall candidate")
    parser.add_argument("--event-move-threshold", type=float, default=0.18,
                        help="Center displacement normalized by box scale")
    parser.add_argument("--event-hip-drop-threshold", type=float, default=0.12,
                        help="Downward hip displacement normalized by body scale")
    parser.add_argument("--event-angle-threshold", type=float, default=22.0,
                        help="Torso-axis angle change in degrees")
    parser.add_argument("--event-aspect-ratio", type=float, default=1.28,
                        help="Box aspect-ratio growth indicating a transition")
    parser.add_argument("--event-height-ratio", type=float, default=0.78,
                        help="Current/previous box-height ratio indicating a collapse")
    parser.add_argument("--event-horizontal-aspect", type=float, default=0.82,
                        help="Aspect ratio used as contextual horizontal-pose evidence")
    parser.add_argument("--event-horizontal-angle", type=float, default=55.0,
                        help="Torso angle used as contextual horizontal-pose evidence")
    parser.add_argument("--event-horizontal-aspect-strong", type=float, default=1.25,
                        help="Very wide bbox fallback for a horizontal pose when torso keypoints are noisy")
    parser.add_argument("--model-fall-threshold", type=float, default=0.62,
                        help="Per-inference-clip Transformer p(falling) threshold for consensus")
    parser.add_argument("--model-fall-single-threshold", type=float, default=0.88,
                        help="Very high Transformer p(falling) can support a single-inference trigger")
    parser.add_argument("--recovery-model-threshold", type=float, default=0.55,
                        help="P(standing/sitting) required to confirm recovery")
    args = parser.parse_args()

    is_camera = args.camera is not None
    video_path = Path(args.video).expanduser() if args.video else None
    checkpoint_path = Path(args.checkpoint).expanduser()
    output_path = Path(args.output).expanduser()
    if not is_camera and (video_path is None or not video_path.is_file()):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, class_names, window_size, ckpt = load_checkpoint(checkpoint_path, device)
    print(f"Loaded Transformer checkpoint: {checkpoint_path}")
    print(f"Best epoch: {ckpt.get('epoch', 'unknown')} | classes: {class_names}")
    print(f"Loading YOLO Pose: {args.pose_weights}")
    pose_model = YOLO(args.pose_weights)
    if args.tracker.lower() == "botsort":
        tracker_cfg = make_tracker_config(
            output_path.parent / "botsort_custom_no_reid.yaml",
            args.track_buffer,
        )
    else:
        tracker_cfg = args.tracker
    print(f"Tracker config: {tracker_cfg}")
    if args.tracker.lower() == "botsort":
        print(f"BoT-SORT without a separate ReID model | track_buffer={args.track_buffer}")

    if is_camera:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(args.camera, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.camera_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.camera_height))
        cap.set(cv2.CAP_PROP_FPS, float(args.camera_fps))
        source_label = f"Laptop camera index {args.camera}"
    else:
        cap = cv2.VideoCapture(str(video_path))
        source_label = f"Video file {video_path}"

    if not cap.isOpened():
        if is_camera:
            raise RuntimeError(
                f"Không mở được camera index {args.camera}. Hãy đóng ứng dụng đang dùng camera, "
                "kiểm tra quyền Camera của Windows, hoặc thử --camera 1."
            )
        raise RuntimeError(f"OpenCV không mở được video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = float(args.camera_fps) if is_camera else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0:
        width = int(args.camera_width) if is_camera else 640
    if height <= 0:
        height = int(args.camera_height) if is_camera else 480
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_frames = max(window_size, int(round(args.clip_seconds * fps)))
    stride = max(1, args.stride_frames)
    if is_camera:
        print(f"Source: {source_label} | {width}x{height}, nominal {fps:.2f} FPS")
    else:
        print(f"Source: {source_label} | {width}x{height}, {fps:.2f} FPS, {frame_count} frames")
    print(f"Multi-person: mỗi track ID gom {clip_frames} frame (~{clip_frames / fps:.1f}s) -> resample {window_size} frame -> Transformer.")
    print("Overlay: green box for non-alert classes, red box for falling/fall alert; no skeleton or top-left status.")

    video_writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not video_writer.isOpened():
        cap.release()
        raise RuntimeError(f"Không tạo được file output: {output_path}")

    zero_xy = np.zeros((17, 2), dtype=np.float32)
    zero_conf = np.zeros((17,), dtype=np.float32)
    # Each ID owns an independent temporal window and probability smoother.
    tracks = {}
    inferred_track_ids = set()
    frame_index = 0
    windows_inferred = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror the camera frame BEFORE detection and drawing.
            # This makes the camera view mirror-like while keeping text labels readable.
            # Existing video files are never mirrored here.
            if is_camera and not args.no_mirror_camera:
                frame = cv2.flip(frame, 1)

            frame_index += 1
            time_sec = frame_index / fps

            result = pose_model.track(
                frame,
                persist=True,
                tracker=str(tracker_cfg),
                verbose=False,
                conf=args.conf,
                imgsz=args.imgsz,
                device=str(device),
            )[0]
            people, draw_items = extract_tracked_people(result)
            seen_ids = set(people.keys())

            # Update gaps for IDs not detected in this frame, keeping temporal alignment.
            stale_ids = []
            for tid, state in list(tracks.items()):
                if tid not in seen_ids:
                    state["xy_buffer"].append(zero_xy.copy())
                    state["conf_buffer"].append(zero_conf.copy())
                    state["found_buffer"].append(False)
                    state["frames_elapsed"] += 1
                if frame_index - state["last_seen_frame"] > args.max_age:
                    stale_ids.append(tid)
            for tid in stale_ids:
                del tracks[tid]

            for tid, (xy, conf, bbox) in people.items():
                if tid not in tracks:
                    tracks[tid] = {
                        "track_id": int(tid),
                        "xy_buffer": deque(maxlen=clip_frames),
                        "conf_buffer": deque(maxlen=clip_frames),
                        "found_buffer": deque(maxlen=clip_frames),
                        "prob_history": deque(maxlen=3),
                        "model_fall_probs": deque(maxlen=3),
                        "current_probs": None,
                        "current_label": "collecting pose",
                        "frames_elapsed": 0,
                        "last_seen_frame": frame_index,
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
                state = tracks[tid]
                state["xy_buffer"].append(xy.copy())
                state["conf_buffer"].append(conf.copy())
                state["found_buffer"].append(True)
                state["frames_elapsed"] += 1
                state["last_seen_frame"] = frame_index

                has_window = len(state["xy_buffer"]) == clip_frames
                should_infer = (
                    has_window
                    and (state["frames_elapsed"] - clip_frames) % stride == 0
                    and sum(state["found_buffer"]) >= int(0.70 * clip_frames)
                )
                if should_infer:
                    win_xy = np.stack(state["xy_buffer"], axis=0)
                    win_conf = np.stack(state["conf_buffer"], axis=0)
                    xy_s, conf_s = resample_clip(win_xy, win_conf, window_size)
                    features = common_window_normalize(xy_s, conf_s)
                    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        logits = model(tensor)
                        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
                    state["prob_history"].append(probs)
                    state["current_probs"] = np.mean(
                        np.stack(state["prob_history"], axis=0), axis=0
                    )
                    probs_smoothed = state["current_probs"]
                    state["current_label"] = str(class_names[int(np.argmax(probs_smoothed))])
                    state["model_fall_probs"].append(
                        get_class_probability(probs_smoothed, class_names, "falling")
                    )
                    state["window_number"] += 1
                    windows_inferred += 1
                    inferred_track_ids.add(int(tid))
                    confidence = float(np.max(probs_smoothed))
                    print(
                        f"t={time_sec:7.2f}s | ID={tid:3d} | "
                        f"Transformer={state['current_label']:17s} {confidence:.3f} | "
                        + " | ".join(f"{name}={p:.3f}"
                                      for name, p in zip(class_names, probs_smoothed))
                    )

                # Event detector runs every frame on this ID's raw pose + box history.
                debug = update_fall_event(
                    state, xy, conf, bbox, frame, frame_index, fps, class_names, args
                )
                state["event_debug"] = debug
                p_fall = float(debug.get("p_falling", 0.0))
                model_conf = float(np.max(state["current_probs"])) if state["current_probs"] is not None else 0.0
                if state["event_active"]:
                    display_status = "FALL_ALERT"
                elif p_fall >= args.model_fall_threshold and bool(debug.get("horizontal_pose_now", False)):
                    display_status = "FALLING_CANDIDATE"
                elif state["current_probs"] is not None:
                    display_status = state["current_label"].upper()
                else:
                    display_status = "PROCESSING"


            # Minimal visualization: only green/red boxes and one compact label per person.
            for tid, xy, conf, bbox in draw_items:
                if tid is not None and tid in tracks:
                    state = tracks[tid]
                    debug = state.get("event_debug", {})
                    p_fall = float(debug.get("p_falling", 0.0))
                    model_conf = float(np.max(state["current_probs"])) if state["current_probs"] is not None else 0.0
                    if state["event_active"]:
                        label = f"ID {tid} | FALL ALERT"
                        is_falling = True
                    elif p_fall >= args.model_fall_threshold and bool(debug.get("horizontal_pose_now", False)):
                        label = f"ID {tid} | FALLING? {p_fall:.0%}"
                        is_falling = True
                    elif (
                        state.get("last_event_cleared_frame") is not None
                        and frame_index - state["last_event_cleared_frame"] <= int(round(2.0 * fps))
                        and bool(debug.get("upright_recovery_pose", False))
                    ):
                        label = f"ID {tid} | RECOVERED"
                        is_falling = False
                    elif (
                        state["current_probs"] is not None
                        and state["current_label"] == "falling"
                        and bool(debug.get("upright_recovery_pose", False))
                    ):
                        label = f"ID {tid} | FALL? {model_conf:.0%}"
                        is_falling = False
                    elif state["current_probs"] is None:
                        label = f"ID {tid} | PROCESSING"
                        is_falling = False
                    else:
                        label = f"ID {tid} | {state['current_label'].upper()} {model_conf:.0%}"
                        is_falling = False
                else:
                    label = "DETECTING"
                    is_falling = False
                draw_box(frame, bbox, label, is_falling)

            # Deliberately no global tracker/status line and no skeleton overlay.
            video_writer.write(frame)
            if not args.no_display:
                cv2.imshow("Multi-person pose tracking + posture/fall", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        cap.release()
        video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print("\nDone.")
    print(f"Annotated output video (MP4): {output_path.resolve()}")
    print(f"Track IDs with at least one prediction: {sorted(inferred_track_ids)}")
    print(f"Total inference windows: {windows_inferred}")
    if windows_inferred == 0:
        print("Chưa có dự đoán: video cần dài hơn --clip-seconds và YOLO phải gán track ID.")
    print("Fall fusion: 8-class Transformer + per-track geometry + horizontal-pose gate.")
    print(f"Alert recovery: min hold={args.event_hold_sec:.2f}s, upright recovery={args.recovery_sec:.2f}s; running/leaning should not trigger unless current body geometry is horizontally down.")
    print("Lưu ý: model 8 lớp không có lying/immobile; ngưỡng hình học cần kiểm chứng trên video có nhãn.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user.")
        sys.exit(130)