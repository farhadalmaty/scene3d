"""Pluggable model interfaces. Swap implementations without touching the pipeline."""
from __future__ import annotations
from typing import Protocol, Sequence
import numpy as np


class GeometryEstimator(Protocol):
    """Monocular video has no depth/pose: this stage supplies them.
    Takes RGB frames, returns Frames (rgb possibly resized to the depth resolution) with depth, K and
    camera-to-world pose in ONE consistent world frame. Scale may be arbitrary (relative depth)."""
    def estimate(self, rgbs: Sequence[np.ndarray]) -> list: ...


class Segmenter(Protocol):
    """Class-agnostic 2D instance masks (SAM / EfficientSAM / FastSAM)."""
    def segment(self, rgb: np.ndarray) -> list[np.ndarray]: ...


class Embedder(Protocol):
    """Vision-language features per mask (CLIP / DINOv2) + text encoder for queries."""
    def embed(self, rgb: np.ndarray, masks: Sequence[np.ndarray]) -> np.ndarray: ...   # (M, d), L2-normalised
    def embed_crops(self, crops: Sequence[np.ndarray]) -> np.ndarray: ...                # (M, d), L2-normalised
    def embed_text(self, text: str) -> np.ndarray: ...                                  # (d,), L2-normalised


class Captioner(Protocol):
    """LVLM crops -> (tag, caption); optional LLM summarisation inside (review Sec. 3.3)."""
    def describe(self, crops: Sequence[np.ndarray]) -> tuple[str, str]: ...
