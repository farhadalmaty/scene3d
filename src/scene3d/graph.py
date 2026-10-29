"""Object-centric 3D scene graph M_t = <O_t, E_t> (review Sec. 1.2)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from .datatypes import ObjectNode
from .geometry import bbox_iou, voxel_downsample


class SceneGraph:
    def __init__(self, up=(0.0, 0.0, 1.0), metric_scale: bool = False):
        self.objects: dict[int, ObjectNode] = {}
        self.edges: list[dict] = []          # {a, b, relation}
        self.up = np.asarray(up, float) / np.linalg.norm(up)
        self.metric_scale = metric_scale     # False => units are arbitrary (monocular)
        self._next = 0

    # ---- objects
    def add(self, points, feature, frame_index) -> ObjectNode:
        o = ObjectNode(self._next, points, feature, 1, frame_index, frame_index)
        self.objects[o.id] = o
        self._next += 1
        return o

    def present(self):
        """Objects that are part of the map (not missing / unreliable / structure). May still be tagged 'unknown'."""
        return [o for o in self.objects.values() if o.status == "present"]

    def shown(self, include_unknown: bool = False):
        """Objects meant to be shown to a user or an agent: present AND confidently named."""
        return [o for o in self.present() if include_unknown or o.tag != "unknown"]

    # ---- post-processing
    def merge_duplicates(self, iou_thr=0.4, sem_thr=0.8, voxel=0.025) -> int:
        """Merge nodes that describe one physical object (over-segmentation, review Sec. 6.2)."""
        merged = 0
        changed = True
        while changed:
            changed = False
            ids = list(self.objects)
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    A, B = self.objects[a], self.objects[b]
                    iou = bbox_iou(A.bbox_min, A.bbox_max, B.bbox_min, B.bbox_max)
                    cs = float(A.feature @ B.feature / (np.linalg.norm(A.feature) * np.linalg.norm(B.feature) + 1e-9))
                    if iou > iou_thr and cs > sem_thr:
                        n = A.n_obs + B.n_obs
                        f = (A.n_obs * A.feature + B.n_obs * B.feature) / n
                        A.feature = f / (np.linalg.norm(f) + 1e-9)
                        A.points = voxel_downsample(np.vstack([A.points, B.points]), voxel)
                        A.n_obs = n
                        A.first_frame = min(A.first_frame, B.first_frame)
                        A.last_frame = max(A.last_frame, B.last_frame)
                        A.captions += B.captions; A.obs += B.obs
                        del self.objects[b]
                        merged += 1; changed = True
                        break
                if changed:
                    break
        return merged

    def build_edges(self, near_thr: float | None = None):
        """Proximity graph -> MST -> geometric relation labels (rule-based stand-in for the LLM step)."""
        objs = self.shown()
        self.edges = []
        if len(objs) < 2:
            return
        C = np.stack([o.center for o in objs])
        D = np.linalg.norm(C[:, None] - C[None], axis=-1) + 1e-6
        mst = minimum_spanning_tree(D).tocoo()
        pairs = {tuple(sorted((int(i), int(j)))) for i, j in zip(mst.row, mst.col)}   # structural neighbours
        seen = set()
        def add(a, b):
            rel = self._relation(a, b, near_thr)
            if rel == "below": a, b, rel = b, a, "on_top_of"          # canonical direction
            elif rel == "contains": a, b, rel = b, a, "inside_of"
            if (a.id, b.id) not in seen:
                seen.add((a.id, b.id)); self.edges.append({"a": a.id, "b": b.id, "relation": rel})
        for i, j in pairs:
            add(objs[i], objs[j])
        for i in range(len(objs)):                                      # plus every support / containment pair
            for j in range(i + 1, len(objs)):
                if self._relation(objs[i], objs[j], near_thr) in ("on_top_of", "below", "inside_of", "contains"):
                    add(objs[i], objs[j])

    def _relation(self, a: ObjectNode, b: ObjectNode, near_thr) -> str:
        u = self.up
        def span(o): p = o.points @ u; return p.min(), p.max()
        (a_lo, a_hi), (b_lo, b_hi) = span(a), span(b)
        h_overlap = self._horizontal_overlap(a, b)
        tol = 0.15 * max(a_hi - a_lo, b_hi - b_lo, 1e-6)
        if h_overlap and abs(a_lo - b_hi) <= tol and a_lo >= b_lo:
            return "on_top_of"
        if h_overlap and abs(b_lo - a_hi) <= tol and b_lo >= a_lo:
            return "below"
        if np.all(a.bbox_min >= b.bbox_min) and np.all(a.bbox_max <= b.bbox_max):
            return "inside_of"
        if np.all(b.bbox_min >= a.bbox_min) and np.all(b.bbox_max <= a.bbox_max):
            return "contains"
        return "near"

    def _horizontal_overlap(self, a, b) -> bool:
        u = self.up
        keep = [i for i in range(3) if abs(u[i]) < 0.7]   # axes ~perpendicular to up
        lo = np.maximum(a.bbox_min[keep], b.bbox_min[keep]); hi = np.minimum(a.bbox_max[keep], b.bbox_max[keep])
        return bool(np.all(hi > lo))

    def rescale(self, factor: float):
        """Convert arbitrary monocular units to metres (scale is unobservable from mono video)."""
        for o in self.objects.values():
            o.points = o.points * factor
        self.metric_scale = True

    # ---- persistence
    def save(self, directory):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        meta = {"up": self.up.tolist(), "metric_scale": self.metric_scale, "next": self._next,
                "edges": self.edges,
                "objects": [{"id": o.id, "n_obs": o.n_obs, "first_frame": o.first_frame, "last_frame": o.last_frame,
                             "captions": o.captions, "tag": o.tag, "caption": o.caption, "status": o.status,
                             "obs": o.obs, "tag_scores": o.tag_scores, "tag_guess": o.tag_guess}
                            for o in self.objects.values()]}
        (d / "scene.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        arrs = {}
        for o in self.objects.values():
            arrs[f"p{o.id}"] = o.points; arrs[f"f{o.id}"] = o.feature
        np.savez_compressed(d / "arrays.npz", **arrs)

    @classmethod
    def load(cls, directory) -> "SceneGraph":
        d = Path(directory)
        meta = json.loads((d / "scene.json").read_text(encoding="utf-8"))
        arr = np.load(d / "arrays.npz")
        g = cls(up=meta["up"], metric_scale=meta["metric_scale"])
        g._next = meta["next"]; g.edges = meta["edges"]
        for m in meta["objects"]:
            o = ObjectNode(m["id"], arr[f"p{m['id']}"], arr[f"f{m['id']}"], m["n_obs"], m["first_frame"], m["last_frame"],
                           m["captions"], m["tag"], m["caption"], m["status"], m.get("obs", []), m.get("tag_scores", []), m.get("tag_guess", ""))
            g.objects[o.id] = o
        return g
