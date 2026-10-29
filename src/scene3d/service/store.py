"""On-disk scene registry. One folder per scene (ASCII ids, so OpenCV is happy on Windows):
    <root>/<scene_id>/video.<ext>  meta.json  process_info.json  frames/  graph/{scene.json,arrays.npz}"""
from __future__ import annotations
import json, shutil, threading, time, uuid
from pathlib import Path


class SceneStore:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def dir(self, scene_id: str) -> Path:
        if not scene_id.isalnum():
            raise KeyError(scene_id)
        d = self.root / scene_id
        if not d.is_dir():
            raise KeyError(f"unknown scene {scene_id}")
        return d

    def create(self, name: str = "") -> str:
        with self._lock:
            sid = uuid.uuid4().hex[:12]
            (self.root / sid).mkdir()
            self._write(sid, {"id": sid, "name": name, "status": "created", "stage": "", "progress": 0.0,
                              "message": "", "error": None, "created": time.time(), "n_objects": None,
                              "metric_scale": False})
            return sid

    def _write(self, sid, meta):
        (self.root / sid / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def meta(self, sid: str) -> dict:
        with self._lock:
            return json.loads((self.dir(sid) / "meta.json").read_text(encoding="utf-8"))

    def update(self, sid: str, **kw) -> dict:
        with self._lock:
            m = self.meta(sid); m.update(kw); self._write(sid, m); return m

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.root.iterdir()):
            if (p / "meta.json").exists():
                out.append(json.loads((p / "meta.json").read_text(encoding="utf-8")))
        return sorted(out, key=lambda m: -m["created"])

    def delete(self, sid: str):
        with self._lock:
            shutil.rmtree(self.dir(sid))

    def graph_dir(self, sid: str) -> Path: return self.dir(sid) / "graph"
