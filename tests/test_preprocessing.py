import numpy as np

from fall_detection.labels import FEATURE_DIM, SEQUENCE_LENGTH
from fall_detection.preprocessing import pose_sequence_to_features, resample_full_clip


def synthetic_pose(n=12):
    # Stable, simple COCO-17 coordinates with visible shoulders and hips.
    xy = np.zeros((n, 17, 2), dtype=np.float32)
    base = np.array([
        [0,0],[0,0],[0,0],[0,0],[0,0],[-10,20],[10,20],[-15,35],[15,35],[-15,50],[15,50],
        [-7,55],[7,55],[-7,75],[7,75],[-7,95],[7,95]
    ], dtype=np.float32)
    for t in range(n):
        xy[t] = base + np.array([t * 0.5, t * 0.25], dtype=np.float32)
    conf = np.ones((n, 17), dtype=np.float32)
    return xy, conf


def test_resample_shape_and_dtype():
    xy, conf = synthetic_pose()
    xs, cs = resample_full_clip(xy, conf)
    assert xs.shape == (SEQUENCE_LENGTH, 17, 2)
    assert cs.shape == (SEQUENCE_LENGTH, 17)
    assert xs.dtype == np.float32


def test_feature_shape_and_finite_values():
    xy, conf = synthetic_pose()
    conf[3:5, 9] = 0.0  # missing/low-confidence joint across two frames
    features = pose_sequence_to_features(xy, conf)
    assert features.shape == (SEQUENCE_LENGTH, FEATURE_DIM)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
