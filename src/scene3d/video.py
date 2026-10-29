from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from .imageio_safe import open_video, imwrite_rgb


def _resize(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]; s = max_side / max(h, w)
    return cv2.resize(rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else rgb


def sample_keyframes(video, n_frames: int = 24, max_side: int = 640, probe_every: int = 3, return_meta: bool = False,
                     rotate: int | None = None, hi_side: int | None = None):
    """Pick n_frames spread evenly over the video; inside each time window take the sharpest frame
    (variance of the Laplacian), which discards motion-blurred frames typical for handheld phone video.

    Two passes: pass 1 scores every `probe_every`-th frame (nothing is kept in memory), pass 2 re-reads the video and
    keeps only the chosen frames. Returns RGB uint8 frames with the long side <= max_side (these go to the depth/pose
    model). With hi_side the same frames are also returned at that larger size in meta["hi_frames"]: segmentation and
    object recognition run on those, because small objects are unrecognisable at depth-model resolution.
    With return_meta=True returns (frames, meta): original frame index of every keyframe, fps, original size, rotation."""
    cap = open_video(video, rotate)
    scores, idx, i, orig_wh = [], [], 0, None
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if orig_wh is None: orig_wh = (bgr.shape[1], bgr.shape[0])
        if i % probe_every == 0:
            g = cv2.cvtColor(_resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_side), cv2.COLOR_RGB2GRAY)
            scores.append(cv2.Laplacian(g, cv2.CV_64F).var()); idx.append(i)
        i += 1
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0); rot = cap.rotation_total; cap.release()
    if not idx:
        raise ValueError("video has no readable frames")
    n = min(n_frames, len(idx)); edges = np.linspace(0, len(idx), n + 1).astype(int); sel = []
    for a, b in zip(edges[:-1], edges[1:]):
        b = max(b, a + 1); sel.append(idx[a + int(np.argmax(scores[a:b]))])
    cap = open_video(video, rotate); want = set(sel); lo, hi, j = {}, {}, 0
    while j <= max(sel):
        ok, bgr = cap.read()
        if not ok: break
        if j in want:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lo[j] = _resize(rgb, max_side)
            if hi_side: hi[j] = _resize(rgb, hi_side)
        j += 1
    cap.release()
    sel = [k for k in sel if k in lo]
    out = [lo[k] for k in sel]
    if return_meta:
        meta = {"indices": sel, "fps": fps, "total_frames": i, "orig_wh": list(orig_wh), "rotation": int(rot)}
        if hi_side: meta["hi_frames"] = [hi[k] for k in sel]
        return out, meta
    return out


def extract_keyframes(video, out_dir, every: int = 15, min_motion: float = 0.0, max_side: int = 960,
                      rotate: int | None = None) -> list[Path]:
    """Save every N-th frame as PNG (manual workflow / debugging)."""
    cap = open_video(video, rotate)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    kept, last, i = [], None, 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if i % every == 0:
            rgb = _resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_side)
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            if last is None or np.abs(g - last).mean() >= min_motion:
                p = out / f"{len(kept):05d}.png"; imwrite_rgb(p, rgb); kept.append(p); last = g
        i += 1
    cap.release()
    return kept
