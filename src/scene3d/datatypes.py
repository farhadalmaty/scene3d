from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Frame:
    """One posed RGB-D observation I_t = <rgb, depth, theta> (review, Sec. 1.2).

    pose_wc: 4x4 camera-to-world transform. depth: HxW, in map units.
    K: 3x3 pinhole intrinsics.
    """
    index: int
    rgb: np.ndarray            # HxWx3 uint8
    depth: np.ndarray          # HxW float32, 0 = invalid
    K: np.ndarray
    pose_wc: np.ndarray
    rgb_hi: np.ndarray | None = None   # higher-resolution copy of the frame for segmentation/tagging (depth stays low-res)


@dataclass
class Detection:
    """A 2D mask lifted to a candidate 3D object: <p_{t,i}, f_{t,i}>."""
    frame_index: int
    mask: np.ndarray           # HxW bool
    points: np.ndarray         # Nx3 world coords (filtered, downsampled)
    feature: np.ndarray        # d, L2-normalised
    crop: np.ndarray | None = None
    label_hint: str | None = None   # only set by synthetic/closed-set backends


@dataclass
class ObjectNode:
    """Mapped object o_j = <p, f, c, B> (review, Sec. 1.2)."""
    id: int
    points: np.ndarray
    feature: np.ndarray
    n_obs: int = 1
    first_frame: int = 0
    last_frame: int = 0
    captions: list[str] = field(default_factory=list)
    tag: str = "unknown"
    caption: str = ""
    status: str = "present"    # present | missing
    obs: list = field(default_factory=list)          # [[frame, x0, y0, x1, y1], ...]: where each mask was seen
    tag_scores: list = field(default_factory=list)   # [[word, prob], ...] top candidates from the tagger
    tag_guess: str = ""                              # best guess when the tag was withheld as "unknown"

    @property
    def bbox_min(self): return self.points.min(0)
    @property
    def bbox_max(self): return self.points.max(0)
    @property
    def center(self): return (self.bbox_min + self.bbox_max) / 2
    @property
    def extent(self): return self.bbox_max - self.bbox_min
