"""Shrink / simplify a video BEFORE processing.

Why: phone videos are 100+ MB (4K/1080p, 30-60 fps, HEVC), while the pipeline only needs ~24 sharp frames at
<= 640 px and the labelled output should be reasonably small. Downscaling and lowering the frame rate cuts the size
by an order of magnitude and speeds up every later step (decoding, tracking, labelling, uploading).

Two engines:
  * ffmpeg (best: H.264 with a quality target, reads HEVC/MOV/any codec, honours the rotation flag, accepts
    non-ASCII input paths on Windows). Found on PATH, or from the pip package `imageio-ffmpeg`.
  * OpenCV fallback (MPEG-4 part 2): works everywhere, larger files, cannot open non-ASCII paths."""
from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path
import cv2
from .imageio_safe import open_video


def find_ffmpeg() -> str | None:
    p = shutil.which("ffmpeg")
    if p: return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _mb(p) -> float: return os.path.getsize(p) / 1e6


def ffmpeg_command(ffmpeg: str, src, dst, max_side: int, fps: float, crf: int, max_seconds: float | None) -> list[str]:
    """Landscape: width <= max_side; portrait: height <= max_side; never upscale; even dimensions; no audio."""
    vf = (f"scale=w='if(gt(iw,ih),min({max_side},iw),-2)':h='if(gt(iw,ih),-2,min({max_side},ih))',"
          f"fps={fps:g}")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    if max_seconds: cmd += ["-t", f"{max_seconds:g}"]
    return cmd + ["-vf", vf, "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p",
                  "-an", "-movflags", "+faststart", str(dst)]


def _compress_opencv(src, dst, max_side: int, fps: float, max_seconds: float | None) -> dict:
    cap = open_video(src)                                   # applies the rotation flag; frames come out upright
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round(src_fps / fps))); out_fps = src_fps / step
    ok, fr = cap.read()
    if not ok: raise ValueError("video has no readable frames")
    h, w = fr.shape[:2]; s = min(1.0, max_side / max(h, w))
    size = (int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2)
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, size)
    if not vw.isOpened(): raise RuntimeError(f"cannot write {dst}")
    i = n = 0; limit = int(max_seconds * src_fps) if max_seconds else None
    while ok:
        if i % step == 0:
            vw.write(cv2.resize(fr, size, interpolation=cv2.INTER_AREA) if s < 1.0 else fr); n += 1
        i += 1
        if limit and i >= limit: break
        ok, fr = cap.read()
    cap.release(); vw.release()
    return {"frames_written": n, "size": size, "fps": out_fps}


def compress_video(src, dst, max_side: int = 960, fps: float = 15.0, crf: int = 28,
                   max_seconds: float | None = None, engine: str = "auto") -> dict:
    """-> {engine, in_mb, out_mb, ratio, ...}. engine: auto | ffmpeg | opencv."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    in_mb = _mb(src); info = {"in_mb": round(in_mb, 1)}
    ff = find_ffmpeg() if engine in ("auto", "ffmpeg") else None
    if engine == "ffmpeg" and not ff: raise RuntimeError("ffmpeg not found (install it or: pip install imageio-ffmpeg)")
    done = False
    if ff:
        r = subprocess.run(ffmpeg_command(ff, src, dst, max_side, fps, crf, max_seconds), capture_output=True, text=True)
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            info.update(engine="ffmpeg", fps=fps, max_side=max_side, crf=crf); done = True
        elif engine == "ffmpeg":
            raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-400:]}")
        else:
            info["ffmpeg_error"] = r.stderr.strip()[-200:]          # fall through to OpenCV
    if not done:
        info.update(engine="opencv", **_compress_opencv(src, dst, max_side, fps, max_seconds))
    info["out_mb"] = round(_mb(dst), 2); info["ratio"] = round(in_mb / max(info["out_mb"], 1e-6), 1)
    return info


def prepare_source(video, out_dir, enabled: bool = True, threshold_mb: float = 50.0, **kw):
    """Compress when the file is big or its path is not ASCII (OpenCV cannot open such paths on Windows).
    Returns (path_to_use, info_or_None). The compressed copy is <out_dir>/source.mp4."""
    video = Path(video)
    big = _mb(video) > threshold_mb; odd = not str(video).isascii()
    if not enabled or not (big or odd):
        return video, None
    dst = Path(out_dir) / "source.mp4"
    info = compress_video(video, dst, **kw)
    info["reason"] = "non-ASCII path" if odd and not big else "large file"
    return dst, info
