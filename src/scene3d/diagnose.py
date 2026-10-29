"""Check WHY an object has a wrong name or wrong place, with data instead of guesses.

For every object we know where each of its masks was seen (frame + 2D box) and its fused 3D cloud. Projecting the
cloud back into the frames where it was seen and comparing with the original mask boxes separates the causes:

  * `cover` low   -> the 3D object does not even cover where its masks were seen: pose/time error for that frame,
                     or wrong associations (parts of OTHER things were merged in).
  * `cover` high, `iou` low -> the cloud is much bigger than the masks: several things merged into one object.
  * both high     -> geometry is fine; if the NAME is still wrong it is a tagging (vocabulary/CLIP) problem:
                     see `tag_prob` and the alternatives.

Output: ranking.csv (worst first) + one contact sheet PNG per selected object (observed frames, yellow = where the
mask was, cyan = where the fused 3D object projects)."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .graph import SceneGraph
from .imageio_safe import imread_rgb, imwrite_rgb
from .annotate import load_geometry


def project_box(points: np.ndarray, T_wc: np.ndarray, K: np.ndarray, W: int, H: int, max_pts: int = 3000):
    P = points[:: max(1, len(points) // max_pts)]
    Tcw = np.linalg.inv(T_wc); pc = P @ Tcw[:3, :3].T + Tcw[:3, 3]; z = pc[:, 2]; f = z > 0.05
    if f.sum() < 5: return None
    zs = np.where(f, z, 1.0)
    u = K[0, 0] * pc[:, 0] / zs + K[0, 2]; v = K[1, 1] * pc[:, 1] / zs + K[1, 2]
    ok = f & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if ok.sum() < 5: return None
    x0, x1 = np.percentile(u[ok], [2, 98]); y0, y1 = np.percentile(v[ok], [2, 98])
    return [float(x0), float(y0), float(x1), float(y1)]


def _area(b): return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
def _inter(a, b): return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def object_consistency(o, geo) -> dict:
    """Per-observation IoU / coverage between the projected 3D object and the box of the mask it came from."""
    H, W = int(geo["hw"][0][0]), int(geo["hw"][0][1]); ious, covs, rat = [], [], []
    for fi, x0, y0, x1, y1 in o.obs:
        if fi >= len(geo["poses_wc"]): continue
        pb = project_box(o.points, geo["poses_wc"][fi], geo["K"][fi], W, H)
        ob = [x0, y0, x1, y1]
        if pb is None:
            ious.append(0.0); covs.append(0.0); rat.append(float("nan")); continue
        inter = _inter(pb, ob)
        ious.append(inter / max(_area(pb) + _area(ob) - inter, 1e-9)); covs.append(inter / max(_area(ob), 1e-9))
        rat.append(_area(pb) / max(_area(ob), 1e-9))
    med = lambda x: float(np.nanmedian(x)) if len(x) else float("nan")
    return {"iou": ious, "cover": covs, "median_iou": med(ious), "median_cover": med(covs), "size_ratio": med(rat)}


def _verdict(r: dict) -> str:
    if r["n_obs"] < 2: return "too few observations"
    if r["median_cover"] < 0.5: return "POSE/TIME or wrong merge: 3D object does not cover where it was seen"
    if r["size_ratio"] > 2.5: return "MERGED: cloud much bigger than its masks"
    if r["tag_prob"] is not None and r["tag_prob"] < 0.35: return "geometry ok, TAG unsure (name may be out of vocabulary)"
    return "ok"


def _sheet(o, cons, frames_dir: Path, geo, out_path: Path, max_tiles: int = 6, tile_w: int = 360):
    from PIL import Image, ImageDraw
    from .annotate import _font
    H, W = int(geo["hw"][0][0]), int(geo["hw"][0][1]); s = tile_w / W
    order = np.argsort([ob[0] for ob in o.obs]); pick = order[np.unique(np.linspace(0, len(order) - 1, min(max_tiles, len(order))).astype(int))]
    tiles = []
    for k in pick:
        fi, x0, y0, x1, y1 = o.obs[k]
        fp = frames_dir / f"{fi:05d}.png"
        if not fp.exists(): continue
        im = Image.fromarray(imread_rgb(fp)).resize((tile_w, int(H * s))); dr = ImageDraw.Draw(im)
        dr.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=(255, 220, 0), width=3)           # where the mask was
        pb = project_box(o.points, geo["poses_wc"][fi], geo["K"][fi], W, H)
        if pb: dr.rectangle([pb[0] * s, pb[1] * s, pb[2] * s, pb[3] * s], outline=(0, 230, 255), width=3)   # where the object projects
        f = _font(15); txt = f"frame {fi}  IoU {cons['iou'][k]:.2f}  cover {cons['cover'][k]:.2f}"
        dr.rectangle([0, 0, tile_w, 20], fill=(0, 0, 0)); dr.text((4, 2), txt, fill=(255, 255, 255), font=f)
        tiles.append(np.asarray(im))
    if not tiles: return False
    cols = min(3, len(tiles)); rows = (len(tiles) + cols - 1) // cols; th = tiles[0].shape[0]; hdr = 64
    canvas = np.full((hdr + rows * th, cols * tile_w, 3), 30, np.uint8)
    for i, t in enumerate(tiles): canvas[hdr + (i // cols) * th: hdr + (i // cols + 1) * th, (i % cols) * tile_w:(i % cols + 1) * tile_w] = t
    im = Image.fromarray(canvas); dr = ImageDraw.Draw(im); f = _font(17)
    alts = ", ".join(f"{w} {p:.2f}" for w, p in o.tag_scores) if o.tag_scores else "-"
    dr.text((6, 4), f"#{o.id} {o.tag}   seen in {len(set(ob[0] for ob in o.obs))} frames   median IoU {cons['median_iou']:.2f}  cover {cons['median_cover']:.2f}", fill=(255, 255, 255), font=f)
    dr.text((6, 34), f"tag candidates: {alts}     yellow = mask box, cyan = projected 3D object", fill=(200, 200, 200), font=_font(14))
    imwrite_rgb(out_path, np.asarray(im)); return True


def inspect_scene(scene_dir, out_dir=None, ids=None, worst: int = 30) -> list[dict]:
    scene_dir = Path(scene_dir); out = Path(out_dir or scene_dir / "inspect"); out.mkdir(parents=True, exist_ok=True)
    graph = SceneGraph.load(scene_dir / "graph"); geo = load_geometry(scene_dir)
    objs = [o for o in graph.objects.values() if o.status != "missing"]
    if not any(o.obs for o in objs):
        raise RuntimeError("this scene has no observation records (made by an older version): run `process` again")
    rows, cache = [], {}
    for o in objs:
        c = object_consistency(o, geo); cache[o.id] = c
        rows.append({"id": o.id, "tag": o.tag, "status": o.status, "tag_guess": o.tag_guess, "tag_prob": o.tag_scores[0][1] if o.tag_scores else None,
                     "alternatives": ", ".join(f"{w} {p:.2f}" for w, p in o.tag_scores[1:]), "n_obs": len(o.obs),
                     "frames": len({ob[0] for ob in o.obs}), "median_iou": round(c["median_iou"], 3),
                     "median_cover": round(c["median_cover"], 3), "size_ratio": round(c["size_ratio"], 2) if c["size_ratio"] == c["size_ratio"] else None})
        rows[-1]["verdict"] = _verdict(rows[-1])
    rows.sort(key=lambda r: (r["median_cover"] if r["n_obs"] >= 2 else 9, r["median_iou"]))
    with open(out / "ranking.csv", "w", newline="", encoding="utf-8-sig") as f:        # utf-8-sig: Excel shows it correctly
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    want = [r["id"] for r in rows[:worst]]
    if ids: want = list(dict.fromkeys(list(ids) + want))
    byid = {o.id: o for o in objs}; made = 0
    for oid in want:
        if oid in byid and byid[oid].obs:
            if _sheet(byid[oid], cache[oid], scene_dir / "frames", geo, out / f"obj_{oid:03d}_{byid[oid].tag.replace(' ', '_')}.png"): made += 1
    for r in rows: r["sheet"] = f"obj_{r['id']:03d}_{r['tag'].replace(' ', '_')}.png" if r["id"] in want else ""
    return rows
