"""Rename objects without re-running the depth model: uses the tagging crops saved by `process` (crops/<id>/m*.png,
c*.png). Lets you compare crop modes and confidence thresholds in seconds."""
from __future__ import annotations
import collections
from pathlib import Path
import numpy as np
from .graph import SceneGraph
from .imageio_safe import imread_rgb


def load_crops(crops_dir, oid: int, mode: str) -> list:
    d = Path(crops_dir) / f"{oid:03d}"
    if not d.is_dir(): return []
    pick = {"masked": ["m"], "context": ["c"], "both": ["m", "c"]}[mode]
    return [imread_rgb(p) for pre in pick for p in sorted(d.glob(f"{pre}*.png"))]


def retag_scene(scene_dir, captioner, min_tag_prob: float = 0.35, mode: str = "masked",
                drop_tags=("wall", "floor", "ceiling"), write: bool = True) -> dict:
    scene_dir = Path(scene_dir); gdir = scene_dir / "graph"; graph = SceneGraph.load(gdir)
    crops_dir = scene_dir / "crops"
    if not crops_dir.is_dir():
        raise RuntimeError("no crops/ folder: run `process` again with this version (crops are saved since v0.5)")
    tags, confs, unknown, structure, changed = collections.Counter(), [], 0, 0, 0
    for o in graph.objects.values():
        if o.status == "missing": continue
        crops = load_crops(crops_dir, o.id, mode)
        if not crops: continue
        old = o.tag
        o.tag, o.caption, o.tag_scores = captioner.describe_scored(crops); o.tag_guess = ""
        if o.status == "structure": o.status = "present"
        if o.tag in drop_tags: o.status = "structure"; structure += 1
        elif o.tag_scores[0][1] < min_tag_prob:
            o.tag_guess, o.tag, o.caption = o.tag, "unknown", "unknown object"; unknown += 1
        confs.append(o.tag_scores[0][1]); tags[o.tag] += 1; changed += int(o.tag != old)
    graph.build_edges()
    if write: graph.save(gdir)
    c = np.array(confs) if confs else np.zeros(1)
    return {"mode": mode, "objects": len(confs), "unknown": unknown, "structure": structure, "changed": changed,
            "conf_median": round(float(np.median(c)), 2), "conf_ge_0.5": int((c >= 0.5).sum()), "tags": dict(tags.most_common())}
