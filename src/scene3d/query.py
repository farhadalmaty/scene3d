"""Agent-facing tool layer. Pure Python and JSON-serialisable, so the same methods back the
MCP server now and an in-app chat / function-calling loop later."""
from __future__ import annotations
import re
import numpy as np
from .graph import SceneGraph


def _r(v, n=3): return [round(float(x), n) for x in np.asarray(v).ravel()]


class AgentTools:
    def __init__(self, graph: SceneGraph, embedder=None, path: str | None = None):
        self.g, self.emb, self.path = graph, embedder, path

    # ---- helpers
    def _get(self, oid: int):
        if oid not in self.g.objects:
            raise KeyError(f"no object with id {oid}")
        return self.g.objects[oid]

    def _card(self, o, extra=None) -> dict:
        d = {"id": o.id, "tag": o.tag, "caption": o.caption, "status": o.status,
             "position": _r(o.center), "size": _r(o.extent),
             "bbox_min": _r(o.bbox_min), "bbox_max": _r(o.bbox_max),
             "observations": o.n_obs, "first_seen_frame": o.first_frame, "last_seen_frame": o.last_frame}
        if o.tag == "unknown" and o.tag_guess: d["tag_guess"] = o.tag_guess
        if o.tag_scores: d["tag_confidence"] = o.tag_scores[0][1]; d["tag_alternatives"] = [w for w, _ in o.tag_scores[1:]]
        if extra: d.update(extra)
        return d

    def _units(self) -> str: return "metres" if self.g.metric_scale else "arbitrary map units (scale not calibrated)"

    # ---- tools
    def list_objects(self, include_missing: bool = False, include_unknown: bool = False) -> dict:
        """List detected objects with 3D position and size. Objects whose name is uncertain are hidden unless include_unknown."""
        objs = [o for o in self.g.objects.values() if (include_missing or o.status == "present")
                and (include_unknown or o.tag != "unknown" or o.status != "present")]
        if not include_missing: objs = [o for o in objs if o.status == "present"]
        hidden = {"unknown_name": sum(1 for o in self.g.present() if o.tag == "unknown"),
                  "unreliable": sum(1 for o in self.g.objects.values() if o.status == "unreliable")}
        return {"units": self._units(), "count": len(objs), "hidden": hidden, "objects": [self._card(o) for o in objs]}

    def get_object(self, object_id: int) -> dict:
        """Full details of one object, including its relations to other objects."""
        o = self._get(object_id)
        return {"units": self._units(), **self._card(o, {"relations": self.get_relations(object_id)["relations"]})}

    def find_objects(self, query: str, top_k: int = 5) -> dict:
        """Find objects by free-text description (semantic match on features + tag/caption words)."""
        words = set(re.findall(r"\w+", query.lower()))
        qf = self.emb.embed_text(query) if self.emb is not None else None
        scored = []
        for o in self.g.shown():
            txt = set(re.findall(r"\w+", f"{o.tag} {o.caption}".lower()))
            lex = len(words & txt) / max(len(words), 1)
            sem = float(o.feature @ qf) if qf is not None else 0.0
            scored.append((0.6 * lex + 0.4 * max(sem, 0), lex, sem, o))
        scored.sort(key=lambda t: -t[0])
        return {"units": self._units(), "query": query,
                "matches": [self._card(o, {"score": round(s, 3)}) for s, _, _, o in scored[:top_k] if s > 0]}

    def where_is(self, query: str) -> dict:
        """Position of the best match for a description, plus nearby context."""
        m = self.find_objects(query, 1)["matches"]
        if not m: return {"found": False, "query": query}
        o = self._get(m[0]["id"])
        rel = self.get_relations(o.id)["relations"]
        return {"found": True, "units": self._units(), "object": m[0], "relations": rel}

    def distance_between(self, id_a: int, id_b: int) -> dict:
        """Euclidean distance between object centres and the gap between their boxes."""
        a, b = self._get(id_a), self._get(id_b)
        gap = np.clip(np.maximum(a.bbox_min - b.bbox_max, b.bbox_min - a.bbox_max), 0, None)
        return {"units": self._units(), "centre_distance": round(float(np.linalg.norm(a.center - b.center)), 3),
                "box_gap": round(float(np.linalg.norm(gap)), 3), "a": a.tag, "b": b.tag}

    def objects_near(self, object_id: int, radius: float) -> dict:
        """Objects whose centre is within `radius` of the given object's centre."""
        o = self._get(object_id)
        res = []
        for p in self.g.shown():
            if p.id != o.id:
                d = float(np.linalg.norm(p.center - o.center))
                if d <= radius: res.append((d, p))
        res.sort(key=lambda t: t[0])
        return {"units": self._units(), "near": [self._card(p, {"distance": round(d, 3)}) for d, p in res]}

    def get_relations(self, object_id: int) -> dict:
        """Spatial relations (on_top_of, below, inside_of, contains, near) involving the object."""
        o = self._get(object_id); rel = []
        for e in self.g.edges:
            if e["a"] == o.id and e["b"] in self.g.objects:
                rel.append({"relation": e["relation"], "other_id": e["b"], "other": self.g.objects[e["b"]].tag})
            elif e["b"] == o.id and e["a"] in self.g.objects:
                inv = {"on_top_of": "below", "below": "on_top_of", "inside_of": "contains", "contains": "inside_of", "near": "near"}
                rel.append({"relation": inv[e["relation"]], "other_id": e["a"], "other": self.g.objects[e["a"]].tag})
        return {"object_id": o.id, "tag": o.tag, "relations": rel}

    def scene_summary(self) -> dict:
        """Compact overview: object count, extent of the scene, and the scene-graph edges as sentences."""
        ob = self.g.shown()
        if not ob: return {"objects": 0}
        lo = np.min([o.bbox_min for o in ob], 0); hi = np.max([o.bbox_max for o in ob], 0)
        sent = [f"{self.g.objects[e['a']].tag}#{e['a']} {e['relation']} {self.g.objects[e['b']].tag}#{e['b']}"
                for e in self.g.edges if e["a"] in self.g.objects and e["b"] in self.g.objects]
        return {"units": self._units(), "objects": len(ob), "hidden_unknown_name": sum(1 for o in self.g.present() if o.tag == "unknown"), "scene_min": _r(lo), "scene_max": _r(hi), "relations": sent}

    def mark_missing(self, object_id: int) -> dict:
        """Flag an object as no longer at its recorded location (e.g. verified absent)."""
        o = self._get(object_id); o.status = "missing"; self.g.build_edges(); self._save()
        return {"id": o.id, "status": o.status}

    def update_position(self, object_id: int, new_center: list[float]) -> dict:
        """Record that an object was moved: translate its geometry to the new centre."""
        o = self._get(object_id)
        o.points = o.points + (np.asarray(new_center, float) - o.center); o.status = "present"
        self.g.build_edges(); self._save()
        return self._card(o)

    def calibrate_scale(self, id_a: int, id_b: int, real_distance_m: float) -> dict:
        """Fix the metric scale: tell the system the real distance (m) between two object centres."""
        cur = float(np.linalg.norm(self._get(id_a).center - self._get(id_b).center))
        if cur < 1e-9: raise ValueError("objects coincide; pick two distinct objects")
        k = real_distance_m / cur; self.g.rescale(k); self._save()
        return {"scale_factor": round(k, 4), "units": self._units()}

    def _save(self):
        if self.path: self.g.save(self.path)
