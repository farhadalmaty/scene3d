from __future__ import annotations
import numpy as np
from sklearn.cluster import DBSCAN


def backproject(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, pose_wc: np.ndarray,
                max_depth: float = np.inf) -> np.ndarray:
    """Lift masked depth pixels to world-frame 3D points using pose and intrinsics K."""
    ok = mask & (depth > 0) & (depth < max_depth)
    v, u = np.nonzero(ok)
    if len(u) == 0:
        return np.empty((0, 3))
    z = depth[v, u].astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    cam = np.stack([x, y, z, np.ones_like(z)], 1)
    return (pose_wc @ cam.T).T[:, :3]


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def largest_cluster(pts: np.ndarray, eps: float, min_samples: int = 5) -> np.ndarray:
    """DBSCAN noise removal; keeps the biggest cluster (review Sec. 1.2)."""
    if len(pts) < min_samples:
        return np.empty((0, 3))
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    good = lab[lab >= 0]
    if len(good) == 0:
        return np.empty((0, 3))
    return pts[lab == np.bincount(good).argmax()]


def bbox_iou(a_min, a_max, b_min, b_max) -> float:
    lo = np.maximum(a_min, b_min); hi = np.minimum(a_max, b_max)
    inter = np.prod(np.clip(hi - lo, 0, None))
    va = np.prod(a_max - a_min); vb = np.prod(b_max - b_min)
    union = va + vb - inter
    return float(inter / union) if union > 0 else 0.0
