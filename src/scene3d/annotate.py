"""Draw object labels onto the video (or its keyframes) from the 3D map.

Idea: every object is a 3D point cloud in the map frame; for a frame with camera pose T (camera->world) and
intrinsics K we project the cloud into the image, get a 2D box, decide whether the object is hidden behind
another object (z-buffer over all object points), and draw '#id tag'.

Keyframes have an exact pose (from the reconstruction). In-between video frames get an interpolated pose
(SLERP for rotation, linear for position) between the neighbouring keyframes, so labels glide smoothly but are
approximate there."""
from __future__ import annotations
import functools
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .graph import SceneGraph
from .imageio_safe import open_video, imread_rgb, imwrite_rgb

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 150, 0), (0, 130, 200), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (170, 110, 40), (128, 128, 0), (0, 128, 128), (255, 100, 100), (100, 100, 255)]


# ------------------------------------------------------------------ geometry
def load_geometry(scene_dir) -> dict:
    p = Path(scene_dir) / "geometry.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Re-run `scene3d process` with this version (poses are saved since v0.3).")
    g = np.load(p)
    return {k: g[k] for k in g.files}


def interp_pose(poses: np.ndarray, kidx: np.ndarray, f: float) -> np.ndarray:
    """Camera-to-world pose at (fractional) video frame f. Holds the nearest keyframe outside their range."""
    order = np.argsort(kidx); poses, kidx = poses[order], np.asarray(kidx)[order]
    if f <= kidx[0]: return poses[0]
    if f >= kidx[-1]: return poses[-1]
    j = int(np.searchsorted(kidx, f, side="right") - 1)
    t = (f - kidx[j]) / (kidx[j + 1] - kidx[j])
    rots = Rotation.from_matrix([poses[j][:3, :3], poses[j + 1][:3, :3]])
    T = np.eye(4)
    T[:3, :3] = Slerp([0, 1], rots)(t).as_matrix()
    T[:3, 3] = (1 - t) * poses[j][:3, 3] + t * poses[j + 1][:3, 3]
    return T


def object_overlays(graph: SceneGraph, T_wc: np.ndarray, K: np.ndarray, W: int, H: int, only=None, show_unknown: bool = False,
                    min_px: int | None = None, max_labels: int = 30, vis_thresh: float = 0.3, max_pts: int = 1500) -> list[dict]:
    """2D boxes of the objects visible in this view: [{id, tag, box=(x0,y0,x1,y1), vis, depth}].
    min_px: smallest box side to label (default scales with the image: ~1.2% of its long side, at least 6 px)."""
    if min_px is None: min_px = max(6, int(0.012 * max(W, H)))
    objs = [o for o in graph.shown(show_unknown) if only is None or o.tag in only]
    if not objs:
        return []
    Tcw = np.linalg.inv(T_wc); R, t = Tcw[:3, :3], Tcw[:3, 3]
    cams = []
    for o in objs:
        P = o.points[:: max(1, len(o.points) // max_pts)]
        pc = P @ R.T + t; z = pc[:, 2]; front = z > 0.05
        zs = np.where(front, z, 1.0)
        u = K[0, 0] * pc[:, 0] / zs + K[0, 2]; v = K[1, 1] * pc[:, 1] / zs + K[1, 2]
        inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        cams.append((o, u, v, z, front, inside))
    s = 0.25                                           # coarse z-buffer over ALL objects (occlusion between objects)
    hh, ww = int(H * s) + 1, int(W * s) + 1
    zb = np.full(hh * ww, 1e6, np.float32)
    for o, u, v, z, front, inside in cams:
        if inside.any():
            idx = (v[inside] * s).astype(int) * ww + (u[inside] * s).astype(int)
            np.minimum.at(zb, idx, z[inside].astype(np.float32))
    zb = cv2.erode(zb.reshape(hh, ww), np.ones((5, 5), np.uint8))      # min filter: closes gaps between sparse points
    out = []
    for o, u, v, z, front, inside in cams:
        if inside.sum() < 5: continue
        cu, cv_ = (u[inside] * s).astype(int), (v[inside] * s).astype(int); zi = z[inside]
        zo = np.full(hh * ww, 1e6, np.float32)                       # this object's OWN nearest surface
        np.minimum.at(zo, cv_ * ww + cu, zi.astype(np.float32))
        zo = cv2.erode(zo.reshape(hh, ww), np.ones((5, 5), np.uint8))
        tol = 0.05 + 0.04 * zi
        own_front = zi <= zo[cv_, cu] + tol                          # skip the object's far side (hidden by itself)
        vis = own_front & (zi <= zb[cv_, cu] + tol)                  # ... and points hidden by OTHER objects
        if own_front.sum() < 5: continue
        vis_score = float(vis.sum()) / float(own_front.sum())
        if vis_score < vis_thresh or vis.sum() < 5: continue
        uu, vv = u[inside][vis], v[inside][vis]
        x0, x1 = np.percentile(uu, [2, 98]); y0, y1 = np.percentile(vv, [2, 98])
        if x1 - x0 < min_px or y1 - y0 < min_px: continue
        out.append({"id": o.id, "tag": o.tag, "box": (float(x0), float(y0), float(x1), float(y1)),
                    "vis": round(vis_score, 2), "depth": float(np.median(zi[vis]))})
    out.sort(key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
    return out[:max_labels]


# ------------------------------------------------------------------ drawing
@functools.lru_cache(maxsize=8)
def _font(size: int):
    from PIL import ImageFont
    for p in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try: return ImageFont.truetype(p, size)
        except OSError: continue
    return ImageFont.load_default()


def draw_overlays(rgb: np.ndarray, overlays: list[dict], with_depth: bool = False) -> np.ndarray:
    """Boxes + '#id tag' labels (PIL, so non-Latin tags work too). Far objects are drawn first."""
    from PIL import Image, ImageDraw
    H, W = rgb.shape[:2]
    img = Image.fromarray(rgb); dr = ImageDraw.Draw(img)
    font = _font(max(13, H // 38)); lw = max(2, H // 240)
    placed: list[tuple] = []
    for d in sorted(overlays, key=lambda d: -d["depth"]):
        col = PALETTE[d["id"] % len(PALETTE)]
        x0, y0, x1, y1 = d["box"]
        dr.rectangle([x0, y0, x1, y1], outline=col, width=lw)
        text = f"#{d['id']} {d['tag']}" + (f" {d['depth']:.1f}" if with_depth else "")
        l, t, r, b = dr.textbbox((0, 0), text, font=font)
        tw, th = r - l + 8, b - t + 6
        lx, ly = min(max(0, x0), max(0, W - tw)), y0 - th
        if ly < 0: ly = y0 + 2                                    # no room above: put inside the box
        for _ in range(8):                                        # nudge down while overlapping earlier labels
            if any(not (lx + tw < a or lx > c or ly + th < bb or ly > e) for a, bb, c, e in placed): ly += th + 1
            else: break
        ly = min(ly, H - th)
        placed.append((lx, ly, lx + tw, ly + th))
        dr.rectangle([lx, ly, lx + tw, ly + th], fill=col)
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        dr.text((lx + 4, ly + 2 - t), text, fill=(0, 0, 0) if lum > 150 else (255, 255, 255), font=font)
    return np.asarray(img)


# ------------------------------------------------------------------ drivers
def _only(only):
    return None if not only else {x.strip() for x in (only.split(",") if isinstance(only, str) else only) if x.strip()}


def _tracked_poses(video, scene_dir, geo, rotation, progress, step):
    """Pass 1: a pose per video frame from the image content. Returns (poses, info) or (None, reason)."""
    from .track import PoseTracker, track_video_poses
    if "depth" not in geo:
        return None, "geometry.npz has no depth (made by an older version)"
    fdir = Path(scene_dir) / "frames"
    kf = []
    for i in range(len(geo["poses_wc"])):
        p = fdir / f"{i:05d}.png"
        if not p.exists(): return None, f"{p} missing"
        kf.append(imread_rgb(p))
    tracker = PoseTracker(geo, kf)
    hp, wp = tracker.hw
    cap = open_video(video, rotation)
    def frames():
        while True:
            ok, bgr = cap.read()
            if not ok: return
            yield cv2.cvtColor(cv2.resize(bgr, (wp, hp), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    total = int(geo["total_frames"])
    try:
        return track_video_poses(frames(), total, tracker, (lambda fr, m: progress(fr * 0.5, m)) if progress else None, step)
    finally:
        cap.release()


def annotate_video(video, scene_dir, out_path, only=None, progress=None, with_depth: bool = False,
                   track: bool = True, track_step: int = 1, show_unknown: bool = False) -> dict:
    """Write a copy of the ORIGINAL video with labels. Needs <scene_dir>/graph and geometry.npz.
    track=True: camera pose of every frame is estimated from the image (accurate, slower);
    track=False or on failure: poses are interpolated between keyframes (fast, approximate)."""
    scene_dir = Path(scene_dir)
    graph = SceneGraph.load(scene_dir / "graph"); geo = load_geometry(scene_dir)
    only = _only(only)
    rotation = int(geo["rotation"]) if "rotation" in geo else None
    kidx = geo["frame_index"].astype(float); poses = geo["poses_wc"]
    Kp = geo["K"].mean(0); hp, wp = [float(x) for x in geo["hw"][0]]
    pose_all, tinfo, mode = None, None, "interpolated"
    if track:
        pose_all, tinfo = _tracked_poses(video, scene_dir, geo, rotation, progress, track_step)
        if pose_all is None:
            tinfo = {"fallback_reason": tinfo}
        else:
            mode = "tracked"
    cap = open_video(video, rotation)
    ok, bgr = cap.read()
    if not ok: raise ValueError("video has no readable frames")
    H0, W0 = bgr.shape[:2]
    K = Kp.copy(); K[0] *= W0 / wp; K[1] *= H0 / hp                  # processed-size intrinsics -> original size
    fps = float(cap.get(cv2.CAP_PROP_FPS) or geo["fps"] or 25.0)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W0, H0))
    if not vw.isOpened(): raise RuntimeError(f"cannot write video: {out_path}")
    total = int(geo["total_frames"]); i = n_lab = 0
    off = 0.5 if pose_all is not None else 0.0
    while ok:
        T = pose_all[i] if (pose_all is not None and i < len(pose_all)) else interp_pose(poses, kidx, i)
        ov = object_overlays(graph, T, K, W0, H0, only, show_unknown)
        n_lab += len(ov)
        rgb = draw_overlays(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ov, with_depth)
        vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if progress and i % 15 == 0: progress(off + (1 - off) * i / max(total, 1), f"labelling {i}/{total}")
        i += 1; ok, bgr = cap.read()
    cap.release(); vw.release()
    return {"frames": i, "labels_drawn": n_lab, "out": str(out_path), "pose_mode": mode, "tracking": tinfo,
            "rotation": rotation}


def annotate_keyframes(scene_dir, out_dir=None, only=None, with_depth: bool = False, show_unknown: bool = False) -> list[Path]:
    """Label the saved keyframes with their EXACT reconstructed poses (most accurate view of the result)."""
    scene_dir = Path(scene_dir); out_dir = Path(out_dir or scene_dir / "annotated"); out_dir.mkdir(parents=True, exist_ok=True)
    graph = SceneGraph.load(scene_dir / "graph"); geo = load_geometry(scene_dir); only = _only(only)
    paths = []
    for i, T in enumerate(geo["poses_wc"]):
        fp = scene_dir / "frames" / f"{i:05d}.png"
        if not fp.exists(): raise FileNotFoundError(f"{fp} missing (was the run made with save_keyframes?)")
        rgb = imread_rgb(fp); H, W = rgb.shape[:2]
        ov = object_overlays(graph, T, geo["K"][i], W, H, only, show_unknown)
        p = out_dir / f"{i:05d}.png"; imwrite_rgb(p, draw_overlays(rgb, ov, with_depth)); paths.append(p)
    return paths
