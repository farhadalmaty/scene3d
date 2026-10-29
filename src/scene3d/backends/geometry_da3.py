"""Pose + depth straight from video frames with Depth Anything 3 (feed-forward, CPU-capable).

Model card (DA3-SMALL, 0.08B params, Apache-2.0): model.inference(images) returns
  .depth [N,H,W], .conf [N,H,W], .extrinsics [N,3,4] world-to-camera, .intrinsics [N,3,3].
Not exercised in the dev sandbox (no network): verify on your machine, in particular the axis convention
of `extrinsics` (assumed OpenCV: x right, y down, z forward)."""
from __future__ import annotations
import numpy as np
import cv2
from ..datatypes import Frame


def _np(x):
    if hasattr(x, "detach"): x = x.detach().cpu().numpy()
    return np.asarray(x)


class DepthAnything3Geometry:
    def __init__(self, model_id: str = "depth-anything/da3-small", device: str | None = None,
                 conf_percentile: float = 20.0):
        import torch
        from .da3_compat import import_da3
        DepthAnything3 = import_da3()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            torch.set_num_threads(max(1, __import__("os").cpu_count() or 1))
        self.model = DepthAnything3.from_pretrained(model_id).to(device=torch.device(self.device))
        self.conf_percentile = conf_percentile
        self.process_res = 504            # long side the depth model works at (library default 504); settable

    def estimate(self, rgbs):
        pred = self.model.inference([np.asarray(r) for r in rgbs], process_res=self.process_res)
        depth, conf = _np(pred.depth).astype(np.float32), _np(pred.conf)
        w2c, K = _np(pred.extrinsics), _np(pred.intrinsics)
        frames = []
        for i, rgb in enumerate(rgbs):
            d = depth[i].copy()
            H, W = d.shape
            thr = np.percentile(conf[i], self.conf_percentile)
            d[(conf[i] < thr) | ~np.isfinite(d) | (d <= 0)] = 0
            T = np.eye(4); T[:3, :] = w2c[i]
            frames.append(Frame(i, cv2.resize(np.asarray(rgb), (W, H), interpolation=cv2.INTER_AREA), d,
                                K[i].astype(np.float64), np.linalg.inv(T)))
        return frames
