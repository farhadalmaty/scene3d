"""Windows-safe image / video I/O.

OpenCV's imread/imwrite/VideoCapture fail on non-ASCII paths on Windows (e.g. C:\\Users\\Фархад\\...).
Images go through imdecode/imencode + numpy file I/O, which handle Unicode paths on every OS.
VideoCapture has no such workaround, so non-ASCII video paths are rejected with a clear message."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def imread_rgb(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path, rgb: np.ndarray) -> None:
    ok, buf = cv2.imencode(Path(path).suffix or ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"cannot encode image for {path}")
    buf.tofile(str(path))


_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class VideoReader:
    """cv2.VideoCapture + correct orientation.

    Phone videos carry a rotation flag; whether OpenCV honours it depends on the build/backend, which led to sideways
    frames on some machines. We read the flag from the file ourselves and rotate deterministically.
    rotate: None = use the file's flag, or 0/90/180/270 to override (clockwise degrees)."""
    def __init__(self, cap, meta_rotation: int, rotate: int | None):
        self.cap = cap
        auto_on = False
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            try: cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)          # we rotate ourselves
            except Exception: pass
            auto_on = cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) > 0.5   # backend insists on auto-rotating: trust it
        self.meta_rotation = meta_rotation
        already = meta_rotation if auto_on else 0                    # rotation the backend has applied for us
        want = meta_rotation if rotate is None else rotate % 360
        self.rotation = (want - already) % 360                       # what is still left for us to apply
        self.rotation_total = want                                   # what the frames end up rotated by vs the raw stream

    def read(self):
        ok, fr = self.cap.read()
        if ok and self.rotation:
            fr = cv2.rotate(fr, _ROT[self.rotation])
        return ok, fr

    def get(self, prop): return self.cap.get(prop)
    def release(self): self.cap.release()


def open_video(path, rotate: int | None = None) -> VideoReader:
    p = str(path)
    if not p.isascii():
        raise ValueError(
            f"Video path contains non-ASCII characters: {p!r}. OpenCV cannot open such paths on Windows. "
            "Copy the video to a folder with a Latin-only path (e.g. C:\\scene3d\\video.mp4) and try again.")
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {p}")
    from .videometa import mp4_rotation
    return VideoReader(cap, mp4_rotation(p), rotate)
