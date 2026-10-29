"""Camera pose for EVERY video frame, from the image itself (not interpolated between keyframes).

Keyframes have a reconstructed pose and a depth map, so each of their SIFT keypoints is a known 3D point in the map
frame. For any other frame we match its keypoints against the two neighbouring keyframes and solve PnP+RANSAC:
the result is a pose in the SAME map frame (aligned, scaled). Frames where this fails (blur, no texture) fall back
to interpolation between their tracked neighbours. Light temporal smoothing removes jitter."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _backproject_pts(uv: np.ndarray, depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray):
    u, v = uv[:, 0], uv[:, 1]
    iu = np.clip(np.round(u).astype(int), 0, depth.shape[1] - 1); iv = np.clip(np.round(v).astype(int), 0, depth.shape[0] - 1)
    z = depth[iv, iu]; ok = z > 0
    x = (u - K[0, 2]) * z / K[0, 0]; y = (v - K[1, 2]) * z / K[1, 1]
    cam = np.stack([x, y, z, np.ones_like(z)], 1)
    return (T_wc @ cam.T).T[:, :3], ok


class PoseTracker:
    def __init__(self, geo: dict, keyframes_rgb: list, n_features: int = 1500, ratio: float = 0.75,
                 min_inliers: int = 25, reproj_px: float = 3.0):
        self.kidx = np.asarray(geo["frame_index"], float); self.poses = geo["poses_wc"]
        self.Kk = geo["K"]; self.K = geo["K"].mean(0)
        self.hw = tuple(int(x) for x in geo["hw"][0])
        self.sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02)   # sensitive: indoor walls are low-contrast
        self.ratio, self.min_inl, self.reproj = ratio, min_inliers, reproj_px
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.ref = []
        for i, rgb in enumerate(keyframes_rgb):
            kp, des = self.sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
            if des is None or len(kp) == 0:
                self.ref.append((np.empty((0, 3)), np.empty((0, 128), np.float32))); continue
            uv = np.array([k.pt for k in kp], np.float32)
            P, ok = _backproject_pts(uv, geo["depth"][i].astype(np.float32), self.Kk[i], self.poses[i])
            self.ref.append((P[ok], des[ok]))

    def _neighbors(self, f: float, n: int = 2) -> list[int]:
        return list(np.argsort(np.abs(self.kidx - f))[:n])

    def localize(self, rgb: np.ndarray, f: float):
        """-> (T_wc 4x4, n_inliers) or None. rgb must already be at keyframe (processed) resolution."""
        kp, des = self.sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
        if des is None or len(kp) < self.min_inl: return None
        uv = np.array([k.pt for k in kp], np.float64)
        obj, img = [], []
        for j in self._neighbors(f):
            P, D = self.ref[j]
            if len(D) < 2: continue
            for m in self.matcher.knnMatch(des, D, k=2):
                if len(m) == 2 and m[0].distance < self.ratio * m[1].distance:
                    obj.append(P[m[0].trainIdx]); img.append(uv[m[0].queryIdx])
        if len(obj) < self.min_inl: return None
        obj, img = np.array(obj, np.float64), np.array(img, np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, self.K, None, iterationsCount=300,
                                                 reprojectionError=self.reproj, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < self.min_inl: return None
        inl = inl[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        T_cw = np.eye(4); T_cw[:3, :3] = R; T_cw[:3, 3] = tvec[:, 0]
        return np.linalg.inv(T_cw), int(len(inl))


def _interp(poses: np.ndarray, idx: np.ndarray, f: float) -> np.ndarray:
    if f <= idx[0]: return poses[0]
    if f >= idx[-1]: return poses[-1]
    j = int(np.searchsorted(idx, f, side="right") - 1); t = (f - idx[j]) / (idx[j + 1] - idx[j])
    T = np.eye(4)
    T[:3, :3] = Slerp([0, 1], Rotation.from_matrix([poses[j][:3, :3], poses[j + 1][:3, :3]]))(t).as_matrix()
    T[:3, 3] = (1 - t) * poses[j][:3, 3] + t * poses[j + 1][:3, 3]
    return T


def smooth_poses(poses: np.ndarray, fixed: np.ndarray, win: int = 5) -> np.ndarray:
    """Moving average of position and rotation (chordal mean); `fixed` frames (keyframes) stay exact."""
    out = poses.copy(); n = len(poses); h = win // 2
    for i in range(n):
        if fixed[i]: continue
        a, b = max(0, i - h), min(n, i + h + 1)
        out[i, :3, 3] = poses[a:b, :3, 3].mean(0)
        out[i, :3, :3] = Rotation.from_matrix(poses[a:b, :3, :3]).mean().as_matrix()
    return out


def track_video_poses(frames_iter, total: int, tracker: PoseTracker, progress=None, step: int = 1,
                      max_dev_trans: float = 1.0, max_dev_deg: float = 40.0):
    """frames_iter yields RGB frames at keyframe resolution, in order. Returns (poses (total,4,4), info)."""
    H, W = tracker.hw
    poses = np.full((total, 4, 4), np.nan); inl = np.zeros(total, int)
    kmap = {int(k): i for i, k in enumerate(tracker.kidx)}
    order = np.argsort(tracker.kidx); kp_sorted, kp_idx = tracker.poses[order], tracker.kidx[order]
    rejected = 0
    for i, rgb in enumerate(frames_iter):
        if i >= total: break
        if i in kmap:
            poses[i] = tracker.poses[kmap[i]]; inl[i] = 10 ** 6                       # keyframes are exact
        elif i % step == 0:
            r = tracker.localize(rgb, i)
            if r is not None:                                  # sanity gate: a wild PnP result is worse than interpolation
                prior = _interp(kp_sorted, kp_idx, float(i))
                dang = np.degrees(np.arccos(np.clip((np.trace(prior[:3, :3].T @ r[0][:3, :3]) - 1) / 2, -1, 1)))
                if np.linalg.norm(prior[:3, 3] - r[0][:3, 3]) <= max_dev_trans and dang <= max_dev_deg:
                    poses[i], inl[i] = r
                else:
                    rejected += 1
        if progress and i % 15 == 0: progress(i / max(total, 1), f"tracking {i}/{total}")
    good = np.isfinite(poses[:, 0, 0])
    gidx = np.nonzero(good)[0]
    if len(gidx) == 0:
        raise RuntimeError("no frame could be tracked")
    for i in np.nonzero(~good)[0]:
        poses[i] = _interp(poses[gidx], gidx.astype(float), float(i))                 # fill gaps between tracked frames
    fixed = np.zeros(total, bool); fixed[[int(k) for k in tracker.kidx if int(k) < total]] = True
    poses = smooth_poses(poses, fixed)
    n_key = int(fixed.sum()); n_tracked = int((good & ~fixed).sum())
    return poses, {"tracked_mask": (good & ~fixed).tolist(), "frames": total, "keyframes": n_key, "tracked": n_tracked,
                   "interpolated": int(total - n_key - n_tracked), "rejected": rejected,
                   "median_inliers": int(np.median(inl[good & ~fixed])) if n_tracked else 0}
