"""Inference preprocessing matched to train_compare_8class.ipynb.

Important: temporal resampling happens BEFORE confidence-based interpolation and
normalization, matching make_fixed_length_clip_sequence() in the notebook.
"""
from __future__ import annotations

import numpy as np

from .labels import FEATURE_DIM, SEQUENCE_LENGTH

KEYPOINT_CONF_THRESHOLD = 0.20
MAX_NORMALIZED_COORD = 8.0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


def resample_full_clip(
    xy: np.ndarray,
    conf: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample raw COCO-17 keypoints from the complete clip."""
    xy = np.asarray(xy, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[1:] != (17, 2):
        raise ValueError(f"Expected xy [T,17,2], got {xy.shape}")
    if conf.shape != xy.shape[:2]:
        raise ValueError(f"Expected conf [T,17], got {conf.shape}")
    n = xy.shape[0]
    if n <= 0:
        raise ValueError("Cannot resample an empty sequence")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if n == 1:
        return (
            np.repeat(xy, sequence_length, axis=0).astype(np.float32),
            np.repeat(conf, sequence_length, axis=0).astype(np.float32),
        )
    target_t = np.linspace(0, n - 1, num=sequence_length, dtype=np.float32)
    left = np.floor(target_t).astype(np.int64)
    right = np.minimum(left + 1, n - 1)
    alpha = (target_t - left).reshape(-1, 1, 1)
    xy_sampled = (1.0 - alpha) * xy[left] + alpha * xy[right]
    conf_sampled = (1.0 - alpha[..., 0]) * conf[left] + alpha[..., 0] * conf[right]
    return xy_sampled.astype(np.float32), conf_sampled.astype(np.float32)


def interpolate_missing_xy(
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thr: float = KEYPOINT_CONF_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate coordinates across time using joints over the confidence threshold."""
    xy = np.asarray(xy, dtype=np.float32).copy()
    conf = np.asarray(conf, dtype=np.float32).copy()
    if xy.ndim != 3 or xy.shape[1:] != (17, 2):
        raise ValueError(f"Expected xy [T,17,2], got {xy.shape}")
    if conf.shape != xy.shape[:2]:
        raise ValueError(f"Expected conf [T,17], got {conf.shape}")
    t_all = np.arange(xy.shape[0])
    for j in range(17):
        valid = np.isfinite(xy[:, j]).all(axis=1) & (conf[:, j] > conf_thr)
        if valid.sum() >= 2:
            for coord in range(2):
                xy[:, j, coord] = np.interp(t_all, t_all[valid], xy[valid, j, coord])
        elif valid.sum() == 1:
            xy[:, j, :] = xy[valid, j, :][0]
        else:
            xy[:, j, :] = 0.0
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.clip(conf, 0.0, 1.0)
    return xy, conf


def common_clip_normalize(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Create 51 features/frame with one clip-level hip origin and torso scale."""
    xy, conf = interpolate_missing_xy(xy, conf)
    valid_conf = conf > KEYPOINT_CONF_THRESHOLD

    hip_centers = []
    for t in range(xy.shape[0]):
        ids = [j for j in (L_HIP, R_HIP) if valid_conf[t, j]]
        if ids:
            hip_centers.append(xy[t, ids].mean(axis=0))
    if hip_centers:
        origin = np.asarray(hip_centers[0], dtype=np.float32)
    else:
        visible = valid_conf[0]
        origin = (
            np.median(xy[0, visible], axis=0).astype(np.float32)
            if visible.any()
            else xy[0].mean(axis=0)
        )

    torso_lengths = []
    for t in range(xy.shape[0]):
        shoulders_ok = valid_conf[t, L_SHOULDER] and valid_conf[t, R_SHOULDER]
        hips_ok = valid_conf[t, L_HIP] and valid_conf[t, R_HIP]
        if shoulders_ok and hips_ok:
            shoulder_mid = 0.5 * (xy[t, L_SHOULDER] + xy[t, R_SHOULDER])
            hip_mid = 0.5 * (xy[t, L_HIP] + xy[t, R_HIP])
            length = float(np.linalg.norm(shoulder_mid - hip_mid))
            if np.isfinite(length) and length > 1e-3:
                torso_lengths.append(length)

    if torso_lengths:
        scale = float(np.median(torso_lengths))
    else:
        visible_xy = xy[valid_conf]
        if len(visible_xy) >= 2:
            spread = np.ptp(visible_xy, axis=0)
            scale = float(max(np.max(spread) * 0.25, 1.0))
        else:
            scale = 1.0
    scale = max(scale, 1.0)

    norm_xy = np.clip((xy - origin[None, None, :]) / scale,
                      -MAX_NORMALIZED_COORD, MAX_NORMALIZED_COORD)
    norm_xy[~valid_conf] = 0.0
    clean_conf = conf.copy()
    clean_conf[~valid_conf] = 0.0
    features = np.concatenate(
        [norm_xy.reshape(xy.shape[0], 34), clean_conf.reshape(xy.shape[0], 17)],
        axis=1,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def pose_sequence_to_features(
    xy: np.ndarray,
    conf: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Convert one tracked person's complete temporal sequence into [30,51]."""
    xy_sampled, conf_sampled = resample_full_clip(xy, conf, sequence_length)
    features = common_clip_normalize(xy_sampled, conf_sampled)
    expected = (sequence_length, FEATURE_DIM)
    if features.shape != expected:
        raise RuntimeError(f"Expected feature shape {expected}, got {features.shape}")
    return features
