"""Pose + depth straight from video frames with Depth Anything 3 (feed-forward, CPU-capable).

Model card (DA3-SMALL, 0.08B params, Apache-2.0): model.inference(images, process_res=504, intrinsics=None, ...) returns
  .depth [N,H,W], .conf [N,H,W], .extrinsics [N,3,4] world-to-camera, .intrinsics [N,3,3] (at the processed resolution).
The model GUESSES the camera's focal length. For an ultra-wide (0.5x) phone camera the guess is too narrow and bends the
reconstruction, so a known field of view can be passed in (fov_long, degrees along the frame's long side); `intrinsics`
is an optional input of the API (docs: "enables pose-conditioned depth estimation").
Axis convention of `extrinsics` is assumed to be OpenCV (x right, y down, z forward)."""
from __future__ import annotations
import warnings
import numpy as np
import cv2
from ..datatypes import Frame


def _np(x):
    if hasattr(x, "detach"): x = x.detach().cpu().numpy()
    return np.asarray(x)


def intrinsics_from_fov(fov_long_deg: float, h: int, w: int, n: int) -> np.ndarray:
    """Square pixels, centred principal point; fov_long is the angle across the LONG side of the frame."""
    f = (max(h, w) / 2.0) / np.tan(np.radians(fov_long_deg) / 2.0)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    return np.tile(K, (n, 1, 1))


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
        self.fov_long = None              # known field of view (deg, long side); None = let the model guess
        self.known_K = None               # or exact intrinsics from the camera: {fx, fy, cx, cy, width, height} (pixels at that size)
        self._scale_input_k = True        # does the library rescale given intrinsics when it resizes the images? (self-checked)
        self.notes: list[str] = []

    def _infer(self, imgs, K=None):
        kw = {"process_res": self.process_res}
        if K is not None: kw["intrinsics"] = K
        return self.model.inference(imgs, **kw)

    def _input_intrinsics(self, h: int, w: int, n: int):
        """Intrinsics for the frames we hand to the model (at THEIR size), or None to let the model guess."""
        k = getattr(self, "known_K", None)
        if k:                                          # exact values from the device, scaled to this frame size
            sx, sy = w / float(k["width"]), h / float(k["height"])
            K = np.array([[k["fx"] * sx, 0, k["cx"] * sx], [0, k["fy"] * sy, k["cy"] * sy], [0, 0, 1.0]])
            return np.tile(K, (n, 1, 1))
        fov = getattr(self, "fov_long", None)
        if fov:
            return intrinsics_from_fov(fov, h, w, n)
        return None

    def estimate(self, rgbs):
        imgs = [np.asarray(r) for r in rgbs]; h, w = imgs[0].shape[:2]; n = len(imgs)
        K_in = self._input_intrinsics(h, w, n)
        if K_in is not None:
            pred = self._infer(imgs, K_in)
            hp, wp = _np(pred.depth).shape[1:]
            exp_fx = K_in[0, 0, 0] * wp / w; got_fx = float(_np(pred.intrinsics)[:, 0, 0].mean())
            if abs(got_fx - exp_fx) / exp_fx > 0.12:
                if abs(got_fx - K_in[0, 0, 0]) / K_in[0, 0, 0] <= 0.12:       # library took our K as already at processed size
                    K_proc = K_in.copy(); K_proc[:, 0] *= wp / w; K_proc[:, 1] *= hp / h
                    self.notes.append("DA3 did not rescale the given intrinsics: re-ran with intrinsics at processed size")
                    pred = self._infer(imgs, K_proc)
                else:
                    msg = (f"given intrinsics (fx_expected {exp_fx:.0f}) do not match what the model returned ({got_fx:.0f}); "
                           "it may be ignoring them")
                    self.notes.append(msg); warnings.warn(msg)
        else:
            pred = self._infer(imgs)
        depth, conf = _np(pred.depth).astype(np.float32), _np(pred.conf)
        w2c, K = _np(pred.extrinsics), _np(pred.intrinsics)
        frames = []
        for i, rgb in enumerate(rgbs):
            d = depth[i].copy()
            H, W = d.shape
            thr = np.percentile(conf[i], self.conf_percentile)
            d[(conf[i] < thr) | ~np.isfinite(d) | (d <= 0)] = 0
            T = np.eye(4); T[:3, :] = w2c[i][:3]
            frames.append(Frame(i, cv2.resize(np.asarray(rgb), (W, H), interpolation=cv2.INTER_AREA), d,
                                K[i].astype(np.float64), np.linalg.inv(T)))
        return frames
