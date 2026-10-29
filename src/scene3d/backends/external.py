"""Load poses/depth produced by ANY external reconstruction tool (COLMAP+depth, VGGT, MASt3R, ...).

Folder layout (one entry per keyframe, same order as the keyframe PNGs):
    geometry/intrinsics.json   {"fx":..,"fy":..,"cx":..,"cy":..}   (at keyframe resolution)
    geometry/poses.npy         (N,4,4) camera-to-world, OpenCV axes (x right, y down, z forward)
    geometry/depth/00000.npy   (H,W) float32 depth along z, 0 = invalid
Depth is up-to-scale for monocular methods -> use AgentTools.calibrate_scale afterwards.
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from ..datatypes import Frame
from ..imageio_safe import imread_rgb


def load_frames(frames_dir: str, geometry_dir: str):
    fd, gd = Path(frames_dir), Path(geometry_dir)
    imgs = sorted(fd.glob("*.png"))
    poses = np.load(gd / "poses.npy")
    ki = json.loads((gd / "intrinsics.json").read_text(encoding="utf-8"))
    K = np.array([[ki["fx"], 0, ki["cx"]], [0, ki["fy"], ki["cy"]], [0, 0, 1.0]])
    if len(imgs) != len(poses):
        raise ValueError(f"{len(imgs)} frames but {len(poses)} poses")
    for i, p in enumerate(imgs):
        rgb = imread_rgb(p)
        depth = np.load(gd / "depth" / f"{i:05d}.npy").astype(np.float32)
        if depth.shape != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        yield Frame(i, rgb, depth, K, poses[i])
