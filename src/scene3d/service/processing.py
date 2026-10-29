"""End-to-end: video file -> keyframes -> pose+depth -> aligned frames -> object map on disk."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
from ..alignment import align_gravity, normalize_scale, clean_depth_edges
from ..imageio_safe import imwrite_rgb
from ..pipeline import Pipeline, PipelineConfig
from ..video import sample_keyframes
from ..compress import prepare_source


@dataclass
class ProcessConfig:
    camera_intrinsics: dict | None = None  # exact intrinsics reported by the camera: {fx, fy, cx, cy, width, height}
    fov_long: float | None = None      # known camera field of view in degrees along the long side of the frame (None: model guesses)
    hi_side: int | None = 960          # long side of the frames used for segmentation/recognition (None: depth resolution)
    da3_res: int | None = None         # long side the depth model works at (library default 504); larger = more detail, slower
    save_crops: bool = True            # keep tagging crops so `scene3d retag` can rename objects without re-running the depth model
    compress: bool = True              # shrink big / non-ASCII-path videos first (see compress.py)
    compress_threshold_mb: float = 50.0
    compress_max_side: int = 960
    compress_fps: float = 15.0
    compress_crf: int = 23             # H.264 quality for the working copy (lower = sharper/bigger); 28 was too lossy for small objects
    depth_conf_cut: float | None = None  # % of least-confident depth pixels dropped per frame (None: model wrapper default 20)
    rotate: int | None = None          # None = read the rotation flag from the file; or 0/90/180/270 clockwise
    n_frames: int = 24                 # keyframes handed to the reconstructor (CPU: 16-32)
    max_side: int = 640                # keyframe long side in px
    auto_up: bool = True               # estimate gravity and rotate the map to Z-up
    refine_up: bool = True             # refine gravity with the largest horizontal plane (RANSAC)
    scale_target_depth: float | None = 2.0   # rescale so median depth = this; None = keep model scale
    edge_filter: bool = True           # invalidate depth at strong discontinuities ('flying pixels')
    edge_rel_grad: float = 0.06        # |dz| / z between neighbouring pixels above this = edge
    save_keyframes: bool = True
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def process_video(video, out_dir, backends, cfg: ProcessConfig | None = None, progress=None):
    """Returns the SceneGraph; also saves it to <out_dir>/graph and keyframes to <out_dir>/frames."""
    cfg = cfg or ProcessConfig()
    cb = progress or (lambda stage, frac, msg="": None)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    cb("frames", 0.0, "preparing video")
    video, cinfo = prepare_source(video, out, cfg.compress, cfg.compress_threshold_mb, max_side=cfg.compress_max_side,
                                  fps=cfg.compress_fps, crf=cfg.compress_crf)
    if cinfo:
        cb("frames", 0.0, f"video compressed ({cinfo['engine']}): {cinfo['in_mb']} MB -> {cinfo['out_mb']} MB, x{cinfo['ratio']}")
        if cinfo["engine"] == "opencv" and cinfo["out_mb"] > 40:
            cb("frames", 0.0, "tip: install ffmpeg (pip install imageio-ffmpeg) for much smaller files")
    cb("frames", 0.0, "reading video")
    rgbs, vmeta = sample_keyframes(video, cfg.n_frames, cfg.max_side, return_meta=True, rotate=cfg.rotate, hi_side=cfg.hi_side)
    cb("frames", 1.0, f"{len(rgbs)} keyframes, frame size {vmeta['orig_wh'][0]}x{vmeta['orig_wh'][1]}, rotation applied {vmeta['rotation']} deg")

    cb("geometry", 0.0, "estimating camera poses and depth (slowest step on CPU)")
    if hasattr(backends.geometry, "fov_long"):
        backends.geometry.fov_long = cfg.fov_long
        backends.geometry.known_K = cfg.camera_intrinsics
    if cfg.depth_conf_cut is not None and hasattr(backends.geometry, "conf_percentile"):
        backends.geometry.conf_percentile = float(cfg.depth_conf_cut)
    if cfg.da3_res and hasattr(backends.geometry, "process_res"):
        backends.geometry.process_res = int(cfg.da3_res)
    frames = backends.geometry.estimate(rgbs)
    for fr, hi in zip(frames, vmeta.get("hi_frames", [])):
        fr.rgb_hi = hi
    cb("geometry", 1.0, "done")

    Kmean = np.mean([f.K for f in frames], axis=0); h_, w_ = frames[0].depth.shape
    info = {"n_keyframes": len(frames), "scale_factor": 1.0, "tilt_deg": None, "fov_long_given": cfg.fov_long, "intrinsics_given": cfg.camera_intrinsics,
            "model_fy_over_fx": round(float(Kmean[1, 1] / Kmean[0, 0]), 3),
            "model_fov_long_deg": round(float(2 * np.degrees(np.arctan(max(h_, w_) / 2 / (Kmean[1, 1] if h_ >= w_ else Kmean[0, 0])))), 1),
            "geometry_notes": list(getattr(backends.geometry, "notes", []))}
    if abs(info["model_fy_over_fx"] - 1) > 0.1:
        cb("geometry", 1.0, f"warning: fy/fx = {info['model_fy_over_fx']} (should be 1.00): the camera model is distorted; try --camera ultrawide or `scene3d calibrate-fov`")
    if cfg.auto_up:
        _, tilt = align_gravity(frames, refine=cfg.refine_up); info["tilt_deg"] = round(tilt, 1)
    if cfg.scale_target_depth:
        info["scale_factor"] = float(normalize_scale(frames, cfg.scale_target_depth))
    if cfg.edge_filter:
        for fr in frames: fr.depth = clean_depth_edges(fr.depth, cfg.edge_rel_grad)
    cb("align", 1.0, "gravity-aligned, scale normalised, depth edges cleaned")

    np.savez_compressed(out / "geometry.npz",             # aligned poses/intrinsics: lets us draw labels back onto the video
                        poses_wc=np.stack([f.pose_wc for f in frames]), K=np.stack([f.K for f in frames]),
                        hw=np.array([f.depth.shape for f in frames]), frame_index=np.array(vmeta["indices"]),
                        fps=vmeta["fps"], total_frames=vmeta["total_frames"], orig_wh=np.array(vmeta["orig_wh"]), rotation=vmeta["rotation"],
                        depth=np.stack([f.depth for f in frames]).astype(np.float16))
    info["video"] = str(Path(video).resolve()); info["compression"] = cinfo; info["rotation_applied"] = vmeta["rotation"]; info["frame_size"] = vmeta["orig_wh"]
    if cfg.save_keyframes:
        fd = out / "frames"; fd.mkdir(exist_ok=True)
        for i, fr in enumerate(frames):
            imwrite_rgb(fd / f"{i:05d}.png", fr.rgb)

    pipe = Pipeline(backends.segmenter, backends.embedder, backends.captioner, cfg.pipeline)
    for k, fr in enumerate(frames):
        pipe.process_frame(fr)
        cb("objects", (k + 1) / len(frames), f"frame {k + 1}/{len(frames)}: {len(pipe.graph.objects)} candidates")
    cb("finalize", 0.0, "merging duplicates, naming objects, building relations")
    geo = {"poses_wc": np.stack([f.pose_wc for f in frames]), "K": np.stack([f.K for f in frames]),
           "hw": np.array([f.depth.shape for f in frames])}
    graph = pipe.finalize(geo)
    if cfg.save_crops:
        pipe.save_crops(out / "crops")
    graph.save(out / "graph")
    info["diagnostics"] = pipe.diagnostics()
    (out / "process_info.json").write_text(json.dumps({**info, "config": asdict(cfg)}, indent=2, default=str), encoding="utf-8")
    d = pipe.stats
    cb("finalize", 1.0, f"{d['shown']} objects shown ({d['unknown_name']} unsure names, {d['unreliable']} unreliable, {d['dropped_structure']} walls/floors dropped)")
    return graph
