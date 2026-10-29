"""Find the camera field of view that makes the reconstruction consistent with the images.

Problem: the depth/pose model GUESSES the camera's focal length. For an ultra-wide (0.5x) phone camera it guesses a
normal-lens value, which bends the whole reconstruction. The model accepts known intrinsics, so we can sweep candidate
fields of view and keep the one that is most consistent with the footage.

Consistency measure (independent of the depth maps): for neighbouring keyframes, SIFT matches that agree with ANY
single epipolar geometry (RANSAC on the image evidence alone) are compared with the epipolar geometry implied by the
MODEL's relative pose and intrinsics (Sampson distance, pixels). Wrong focal length => the model's geometry
contradicts the image matches => larger error."""
from __future__ import annotations
import cv2
import numpy as np


def _skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def sampson(F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    x1 = np.c_[p1, np.ones(len(p1))]; x2 = np.c_[p2, np.ones(len(p2))]
    Fx1 = x1 @ F.T; Ftx2 = x2 @ F
    num = np.sum(x2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def _features(frames, n_features: int = 1500):
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02); out = []
    for f in frames:
        kp, des = sift.detectAndCompute(cv2.cvtColor(np.asarray(f.rgb), cv2.COLOR_RGB2GRAY), None)
        out.append((np.array([k.pt for k in kp], np.float64) if kp else np.empty((0, 2)), des))
    return out


def epipolar_consistency(frames, min_matches: int = 25, min_baseline: float = 1e-3, feats=None) -> dict:
    """Median Sampson error (px) of RANSAC-consistent matches under the model's own geometry, over neighbouring pairs."""
    feats = feats or _features(frames); bf = cv2.BFMatcher(cv2.NORM_L2); per_pair = []
    for i in range(len(frames) - 1):
        (p1, d1), (p2, d2) = feats[i], feats[i + 1]
        if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8: continue
        good = [m[0] for m in bf.knnMatch(d1, d2, k=2) if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < min_matches: continue
        a = p1[[m.queryIdx for m in good]]; b = p2[[m.trainIdx for m in good]]
        Fr, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.5, 0.999)
        if Fr is None or mask is None or mask.sum() < min_matches: continue
        a, b = a[mask.ravel() == 1], b[mask.ravel() == 1]
        T = np.linalg.inv(frames[i + 1].pose_wc) @ frames[i].pose_wc            # camera i -> camera j
        R, t = T[:3, :3], T[:3, 3]
        base = np.linalg.norm(t) / max(np.linalg.norm(frames[i].pose_wc[:3, 3]) + 1.0, 1.0)
        if base < min_baseline: continue                                         # pure rotation: epipolar geometry degenerate
        F = np.linalg.inv(frames[i + 1].K).T @ (_skew(t) @ R) @ np.linalg.inv(frames[i].K)
        per_pair.append(float(np.median(sampson(F, a, b))))
    if not per_pair:
        return {"median_px": float("nan"), "pairs": 0}
    return {"median_px": float(np.median(per_pair)), "pairs": len(per_pair), "per_pair": per_pair}


def sweep_fov(rgbs, geometry, fovs, progress=None) -> list[dict]:
    """Run the geometry model once per candidate field of view (None = let the model guess) and score each result.
    `geometry` must expose a settable attribute fov_long (degrees along the long side of the frame)."""
    rows = []
    for k, fov in enumerate(fovs):
        geometry.fov_long = fov
        frames = geometry.estimate(rgbs)
        c = epipolar_consistency(frames)
        K = np.mean([f.K for f in frames], axis=0)
        h, w = frames[0].depth.shape
        rows.append({"fov_long": fov, "median_px": c["median_px"], "pairs": c["pairs"],
                     "fy_over_fx": float(K[1, 1] / K[0, 0]),
                     "model_fov_long": float(2 * np.degrees(np.arctan(max(h, w) / 2 / (K[1, 1] if h >= w else K[0, 0]))))})
        if progress: progress((k + 1) / len(fovs), f"fov {fov}: {rows[-1]['median_px']:.2f} px")
    geometry.fov_long = None
    return rows
