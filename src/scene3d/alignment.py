"""Put a monocular reconstruction into a usable frame: gravity-aligned (Z up) and with sane scale.

A feed-forward reconstructor returns poses in an arbitrary world frame (usually the first camera:
x right, y down, z forward) and arbitrary depth scale. The rest of the system assumes Z is up and its
distance thresholds (2.5 cm voxels etc.) assume metres, so we fix both before building the map."""
from __future__ import annotations
import numpy as np
from .datatypes import Frame
from .geometry import backproject


def up_from_cameras(poses_wc: np.ndarray) -> np.ndarray:
    """Handheld video: the camera 'down' axis (y in OpenCV convention) points roughly to the floor."""
    y = np.stack([p[:3, 1] for p in poses_wc])
    up = -y.mean(0)
    n = np.linalg.norm(up)
    return up / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])


def sample_cloud(frames: list[Frame], stride: int = 8, max_points: int = 60000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed); pts = []
    for fr in frames:
        m = np.zeros(fr.depth.shape, bool); m[::stride, ::stride] = True
        pts.append(backproject(fr.depth, m, fr.K, fr.pose_wc))
    P = np.vstack(pts) if pts else np.empty((0, 3))
    if len(P) > max_points:
        P = P[rng.choice(len(P), max_points, replace=False)]
    return P


def refine_up_ransac(points: np.ndarray, up0: np.ndarray, max_angle_deg: float = 35.0,
                     iters: int = 400, seed: int = 0) -> np.ndarray:
    """Refine 'up' with the biggest roughly-horizontal plane (floor / table top / ceiling).
    Only planes within max_angle_deg of the camera-based guess are considered."""
    if len(points) < 200:
        return up0
    rng = np.random.default_rng(seed)
    lo, hi = np.percentile(points, 5, axis=0), np.percentile(points, 95, axis=0)
    tol = 0.01 * float(np.linalg.norm(hi - lo))
    cos_min = np.cos(np.deg2rad(max_angle_deg))
    best_n, best_cnt = None, 0
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-12: continue
        n /= nn
        if n @ up0 < 0: n = -n
        if n @ up0 < cos_min: continue
        cnt = int((np.abs((points - a) @ n) < tol).sum())
        if cnt > best_cnt: best_cnt, best_n = cnt, n
    if best_n is None or best_cnt < 0.03 * len(points):
        return up0
    # least-squares polish on inliers
    inl = points[np.abs((points - points.mean(0)) @ best_n) < 3 * tol]
    if len(inl) >= 50:
        _, _, vt = np.linalg.svd(inl - inl.mean(0), full_matrices=False)
        n = vt[-1]; n = n if n @ best_n > 0 else -n
        best_n = n
    return best_n / np.linalg.norm(best_n)


def rotation_to_z(up: np.ndarray) -> np.ndarray:
    """Rotation R with R @ up = (0,0,1)."""
    up = up / np.linalg.norm(up); z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z); s = np.linalg.norm(v); c = float(up @ z)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def align_gravity(frames: list[Frame], refine: bool = True) -> tuple[np.ndarray, float]:
    """Rotate the world so that estimated 'up' becomes +Z (in place). Returns (R, tilt_deg_of_initial_guess)."""
    poses = np.stack([f.pose_wc for f in frames])
    up = up_from_cameras(poses)
    up0 = up.copy()
    if refine:
        up = refine_up_ransac(sample_cloud(frames), up)
    R = rotation_to_z(up)
    T = np.eye(4); T[:3, :3] = R
    for f in frames:
        f.pose_wc = T @ f.pose_wc
    # shift so the floor-ish bottom of the scene (2nd percentile of height) sits at z = 0
    z0 = float(np.percentile(sample_cloud(frames)[:, 2], 2)) if frames else 0.0
    for f in frames:
        f.pose_wc = f.pose_wc.copy(); f.pose_wc[2, 3] -= z0
    tilt = float(np.degrees(np.arccos(np.clip(up0 @ up, -1, 1))))
    return R, tilt


def normalize_scale(frames: list[Frame], target_median_depth: float = 2.0) -> float:
    """Scale depth and camera translation so the median valid depth equals target (a plausible indoor value).
    NOT a metric calibration: use AgentTools.calibrate_scale for real units."""
    d = np.concatenate([f.depth[f.depth > 0].ravel() for f in frames])
    if len(d) == 0:
        return 1.0
    s = target_median_depth / float(np.median(d))
    for f in frames:
        f.depth = (f.depth * s).astype(np.float32)
        p = f.pose_wc.copy(); p[:3, 3] *= s; f.pose_wc = p
    return s


def clean_depth_edges(depth: np.ndarray, rel_grad: float = 0.06) -> np.ndarray:
    """Zero out depth where it jumps between neighbouring pixels (object borders). Monocular models smear depth
    across such borders, which produces long streaks of 'flying' 3D points."""
    d = depth.astype(np.float32).copy()
    valid = d > 0
    z = np.where(valid, d, np.nan)
    gx = np.abs(np.diff(z, axis=1, prepend=z[:, :1])); gy = np.abs(np.diff(z, axis=0, prepend=z[:1, :]))
    with np.errstate(invalid="ignore"):
        edge = (gx / z > rel_grad) | (gy / z > rel_grad)
        edge |= np.roll(edge, 1, 1) | np.roll(edge, -1, 1) | np.roll(edge, 1, 0) | np.roll(edge, -1, 0)   # 1-px safety ring
    d[np.nan_to_num(edge, nan=False).astype(bool)] = 0
    return d
