"""Cross-view association (review Sec. 3.2): phi = phi_geo + phi_sem, greedy assignment."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .datatypes import Detection, ObjectNode
from .geometry import voxel_downsample


def nn_ratio(p_det: np.ndarray, p_obj: np.ndarray, delta_nn: float) -> float:
    """Fraction of detection points that have a map point within delta_nn."""
    if len(p_det) == 0 or len(p_obj) == 0:
        return 0.0
    d, _ = cKDTree(p_obj).query(p_det, k=1)
    return float((d < delta_nn).mean())


def sem_sim(f_a: np.ndarray, f_b: np.ndarray) -> float:
    c = float(f_a @ f_b / (np.linalg.norm(f_a) * np.linalg.norm(f_b) + 1e-9))
    return 0.5 * c + 0.5


def _diag(p: np.ndarray) -> float:
    return float(np.linalg.norm(p.max(0) - p.min(0))) if len(p) else 0.0


def similarity_parts(det: Detection, obj: ObjectNode, delta_nn: float, rel: float = 0.05):
    """(geo, sem) or None when the two do not overlap spatially at all.
    The distance tolerance grows with the SMALLER of the two objects (so big surfaces tolerate pose error,
    but a mug is not swallowed by the table it stands on)."""
    tol = max(delta_nn, rel * min(_diag(det.points), _diag(obj.points)))
    if np.any(det.points.max(0) < obj.bbox_min - tol) or np.any(det.points.min(0) > obj.bbox_max + tol):
        return None
    return nn_ratio(det.points, obj.points, tol), sem_sim(det.feature, obj.feature)


def similarity(det: Detection, obj: ObjectNode, delta_nn: float) -> float:
    p = similarity_parts(det, obj, delta_nn)
    return 0.0 if p is None else p[0] + p[1]


def fuse(obj: ObjectNode, det: Detection, voxel: float) -> None:
    """Running weighted average of features + point-set union (review Sec. 3.2)."""
    obj.feature = (obj.n_obs * obj.feature + det.feature) / (obj.n_obs + 1)
    obj.feature /= np.linalg.norm(obj.feature) + 1e-9
    obj.points = voxel_downsample(np.vstack([obj.points, det.points]), voxel)
    obj.n_obs += 1
    obj.last_frame = det.frame_index
    obj.status = "present"


def associate(det: Detection, objects: list[ObjectNode], delta_nn: float, delta_sim: float):
    """Return (best_object or None, score, info). info = {overlapping, geo, sem} of the best candidate (diagnostics)."""
    best, best_s, info, n_ov = None, -1.0, {"overlapping": 0, "geo": None, "sem": None}, 0
    for o in objects:
        p = similarity_parts(det, o, delta_nn)
        if p is None: continue
        n_ov += 1
        s = p[0] + p[1]
        if s > best_s: best, best_s, info["geo"], info["sem"] = o, s, p[0], p[1]
    info["overlapping"] = n_ov
    return (best, best_s, info) if best_s > delta_sim else (None, best_s, info)
