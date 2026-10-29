"""SceneService: the single object the API, CLI and agents talk to.
Owns the store, a one-worker job queue (CPU inference is serial anyway) and lazily-loaded models."""
from __future__ import annotations
import shutil, threading, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ..graph import SceneGraph
from ..query import AgentTools
from .processing import ProcessConfig, process_video
from .store import SceneStore


def config_from_request(base: ProcessConfig, camera: str = "", fov_long: float = 0.0, intrinsics_json: str = "") -> ProcessConfig:
    """Per-upload camera information -> ProcessConfig. Priority: exact intrinsics > fov_long > camera preset > model guess."""
    import dataclasses, json
    from ..presets import CAMERA_FOV
    kw = {}
    if intrinsics_json:
        d = json.loads(intrinsics_json)
        need = {"fx", "fy", "cx", "cy", "width", "height"}
        if not need <= set(d): raise ValueError(f"intrinsics_json needs keys {sorted(need)}")
        kw["camera_intrinsics"] = {k: float(d[k]) for k in need}
    elif fov_long:
        kw["fov_long"] = float(fov_long)
    elif camera:
        if camera not in CAMERA_FOV: raise ValueError(f"unknown camera {camera!r}; use one of {sorted(CAMERA_FOV)}")
        kw["fov_long"] = CAMERA_FOV[camera]
    return dataclasses.replace(base, **kw) if kw else base


class SceneService:
    def __init__(self, data_dir="data", backends_factory=None, config: ProcessConfig | None = None):
        self.store = SceneStore(data_dir)
        self._factory, self._backends, self._blk = backends_factory, None, threading.Lock()
        self.config = config or ProcessConfig()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene3d-job")
        self._tools: dict[str, tuple[float, AgentTools]] = {}
        self._futures: dict[str, object] = {}

    # ---- models (loaded once, on first job)
    def backends(self):
        with self._blk:
            if self._backends is None:
                if self._factory is None:
                    from ..backends.models import default_backends
                    self._factory = default_backends
                self._backends = self._factory()
            return self._backends

    # ---- ingest
    def submit_video(self, video_path, name: str = "", copy: bool = True, config: ProcessConfig | None = None) -> str:
        sid = self.store.create(name)
        src = Path(video_path)
        dst = self.store.dir(sid) / ("video" + (src.suffix.lower() or ".mp4"))
        shutil.copyfile(src, dst) if copy else shutil.move(src, dst)
        self.store.update(sid, status="queued", message="waiting for worker")
        self._futures[sid] = self._pool.submit(self._run, sid, dst, config or self.config)
        return sid

    def _run(self, sid: str, video: Path, cfg: ProcessConfig):
        def cb(stage, frac, msg=""):
            order = {"frames": (0, .05), "geometry": (.05, .45), "align": (.45, .5), "objects": (.5, .95), "finalize": (.95, 1.0)}
            a, b = order.get(stage, (0, 1))
            self.store.update(sid, status="running", stage=stage, progress=round(a + (b - a) * frac, 3), message=msg)
        try:
            cb("frames", 0.0, "starting")
            g = process_video(video, self.store.dir(sid), self.backends(), cfg, cb)
            self.store.update(sid, status="done", stage="done", progress=1.0, message="", n_objects=len(g.present()),
                              metric_scale=g.metric_scale)
        except Exception as e:                       # report to the client instead of killing the worker
            self.store.update(sid, status="error", error=f"{type(e).__name__}: {e}", message=traceback.format_exc(limit=3))

    def wait(self, sid: str, timeout: float | None = None):
        f = self._futures.get(sid)
        if f is not None: f.result(timeout)
        return self.store.meta(sid)

    # ---- queries
    def tools(self, sid: str) -> AgentTools:
        meta = self.store.meta(sid)
        if meta["status"] != "done":
            raise RuntimeError(f"scene {sid} is not ready (status: {meta['status']})")
        gd = self.store.graph_dir(sid)
        mtime = (gd / "scene.json").stat().st_mtime
        cached = self._tools.get(sid)
        if cached is None or cached[0] != mtime:
            emb = None
            if self._backends is not None: emb = self._backends.embedder
            self._tools[sid] = (mtime, AgentTools(SceneGraph.load(gd), emb, str(gd)))
        return self._tools[sid][1]

    def annotate(self, sid: str, only: str = "", progress=None) -> Path:
        """Render <scene>/annotated.mp4 (labels drawn onto the original video)."""
        from ..annotate import annotate_video
        meta = self.store.meta(sid)
        if meta["status"] != "done": raise RuntimeError(f"scene {sid} is not ready (status: {meta['status']})")
        d = self.store.dir(sid)
        video = next(d.glob("video.*"))
        annotate_video(video, d, d / "annotated.mp4", only, progress)
        return d / "annotated.mp4"

    def shutdown(self): self._pool.shutdown(wait=False, cancel_futures=True)
