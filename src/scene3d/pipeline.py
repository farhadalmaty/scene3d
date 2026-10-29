"""Frame -> masks -> features -> 3D candidates -> association/fusion -> scene graph (review Sec. 3)."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .datatypes import Frame, Detection
from .geometry import backproject, voxel_downsample, largest_cluster
from .association import associate, fuse
from .graph import SceneGraph


@dataclass
class PipelineConfig:
    delta_nn: float = 0.03       # nn-ratio distance floor (review: 2.5 cm); grows with object size, see association.py
    delta_sim: float = 1.1       # association threshold (review: 1.1)
    voxel: float = 0.025         # voxel downsampling (review: 2.5 cm)
    dbscan_eps: float = 0.06
    min_points: int = 25
    min_mask_area: int = 80      # px, absolute floor
    min_mask_frac: float = 0.0015  # ... and at least this share of the image (small objects stay in, specks go)
    max_mask_frac: float = 0.35  # masks covering more of the image are walls/floor, not objects
    erode_px: int = 3            # shrink masks before lifting: removes boundary 'flying pixels'
    min_obs: int = 3             # keep objects seen in >= this many frames (singletons are mostly noise)
    max_extent: float = 3.0      # drop objects whose 3D bbox diagonal exceeds this (map units ~ metres)
    drop_tags: tuple = ("wall", "floor", "ceiling")   # structure, not interactive objects
    max_depth: float = float("inf")
    max_crops: int = 10          # top-k crops per object for captioning (review: 10)
    crop_mode: str = "masked"    # what the tagger sees: masked | context | both (compare with `scene3d retag`)
    min_tag_prob: float = 0.35   # names with lower confidence are withheld: the object is kept as 'unknown'
    min_cover: float = 0.5       # objects whose 3D cloud does not cover where their masks were seen are 'unreliable'
    max_size_ratio: float = 3.0  # ... and so are objects whose cloud is this many times bigger than their masks
    merge_iou: float = 0.3
    merge_sem: float = 0.8
    up: tuple = (0.0, 0.0, 1.0)


def _crop(rgb, mask, pad=0.1, fill=127):
    """Bounding-box crop of the object with everything OUTSIDE the mask painted neutral grey.
    Without this, a thin object (book on a table) is described by its surroundings, not by itself."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
    ya, yb, xa, xb = max(0, y0 - py), y1 + py, max(0, x0 - px), x1 + px
    out = np.full(rgb[ya:yb, xa:xb].shape, fill, rgb.dtype)
    m = mask[ya:yb, xa:xb]
    out[m] = rgb[ya:yb, xa:xb][m]
    return out


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _context_crop(rgb, mask, pad=0.3):
    """Plain bounding-box crop WITH surroundings (context helps recognise e.g. a microwave in a kitchen)."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
    return rgb[max(0, y0 - py):y1 + py, max(0, x0 - px):x1 + px].copy()


class Pipeline:
    def __init__(self, segmenter, embedder, captioner=None, cfg: PipelineConfig | None = None):
        self.seg, self.emb, self.cap = segmenter, embedder, captioner
        self.cfg = cfg or PipelineConfig()
        self.graph = SceneGraph(up=self.cfg.up)
        self._crops: dict[int, list[tuple[int, np.ndarray]]] = {}
        self.stats = {"masks": 0, "too_big": 0, "too_few_points": 0, "no_overlap": 0, "merged_into_existing": 0,
                      "new_objects": 0, "geo": [], "sem": []}

    def process_frame(self, fr: Frame) -> int:
        """Masks and recognition run on the high-resolution frame when there is one (fr.rgb_hi); the 3D lifting uses the
        depth-resolution copy of each mask, because depth/pose live at that resolution."""
        c, st = self.cfg, self.stats
        H, W = fr.depth.shape
        rgb_seg = fr.rgb_hi if fr.rgb_hi is not None else fr.rgb
        min_area = max(c.min_mask_area, c.min_mask_frac * H * W)
        keep = []
        for m in self.seg.segment(rgb_seg):
            ml = m if m.shape == (H, W) else cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            area = int(ml.sum())
            if area < min_area: continue
            if area > c.max_mask_frac * H * W: st["too_big"] += 1; continue
            keep.append((m, ml))
        if not keep:
            return 0
        feats = self.emb.embed(rgb_seg, [m for m, _ in keep])
        created = 0
        k = np.ones((c.erode_px, c.erode_px), np.uint8) if c.erode_px > 1 else None
        for (m, ml), f in zip(keep, feats):
            st["masks"] += 1
            mm = ml
            if k is not None:                                   # drop the boundary ring; keep the mask if it would vanish
                er = cv2.erode(ml.astype(np.uint8), k).astype(bool)
                if er.sum() >= 0.3 * ml.sum(): mm = er
            pts = backproject(fr.depth, mm, fr.K, fr.pose_wc, c.max_depth)
            pts = largest_cluster(voxel_downsample(pts, c.voxel), c.dbscan_eps)
            if len(pts) < c.min_points:
                st["too_few_points"] += 1; continue
            det = Detection(fr.index, ml, pts, f)
            obj, _, info = associate(det, self.graph.present(), c.delta_nn, c.delta_sim)
            if info["overlapping"] == 0: st["no_overlap"] += 1
            elif info["geo"] is not None: st["geo"].append(info["geo"]); st["sem"].append(info["sem"])
            if obj is None:
                obj = self.graph.add(pts, f, fr.index); created += 1; st["new_objects"] += 1
            else:
                fuse(obj, det, c.voxel); st["merged_into_existing"] += 1
            obj.obs.append([int(fr.index)] + _bbox(ml))           # box in DEPTH-resolution coordinates
            lst = self._crops.setdefault(obj.id, [])
            lst.append((int(m.sum()), _crop(rgb_seg, m), _context_crop(rgb_seg, m)))
            lst.sort(key=lambda t: -t[0]); del lst[c.max_crops:]
        return created

    def _drop(self, ids):
        for i in list(ids):
            self.graph.objects.pop(i, None); self._crops.pop(i, None)

    def tag_crops(self, oid: int, mode: str | None = None) -> list:
        mode = mode or self.cfg.crop_mode
        ent = self._crops.get(oid, [])
        masked, ctx = [e[1] for e in ent], [e[2] for e in ent]
        return masked if mode == "masked" else ctx if mode == "context" else masked + ctx

    def _mark_unreliable(self, geo) -> int:
        """Objects whose 3D cloud does not land where their masks were seen (bad merges, pose errors) get status
        'unreliable': kept for inspection, hidden from users and agents."""
        from .diagnose import object_consistency
        c, n = self.cfg, 0
        for o in self.graph.objects.values():
            if o.status != "present" or len(o.obs) < 2: continue
            r = object_consistency(o, geo)
            if r["median_cover"] < c.min_cover or (r["size_ratio"] == r["size_ratio"] and r["size_ratio"] > c.max_size_ratio):
                o.status = "unreliable"; n += 1
        return n

    def finalize(self, geo=None):
        """geo (optional): {"poses_wc", "K", "hw"} of the keyframes: enables the consistency filter."""
        c, g, st = self.cfg, self.graph, self.stats
        st["candidates"] = len(g.objects)
        before = set(g.objects)
        g.merge_duplicates(c.merge_iou, c.merge_sem, c.voxel)
        self._drop(before - set(g.objects)); st["after_merge"] = len(g.objects)
        self._drop([i for i, o in g.objects.items() if o.n_obs < c.min_obs]); st["after_min_obs"] = len(g.objects)
        self._drop([i for i, o in g.objects.items() if np.linalg.norm(o.extent) > c.max_extent])
        st["after_size_filter"] = len(g.objects)
        st["dropped_structure"] = st["unknown_name"] = st["unreliable"] = 0
        if self.cap is not None:
            for o in g.objects.values():
                crops = self.tag_crops(o.id)
                if crops:
                    if hasattr(self.cap, "describe_scored"):
                        o.tag, o.caption, o.tag_scores = self.cap.describe_scored(crops)
                    else:
                        o.tag, o.caption = self.cap.describe(crops)
            gone = [i for i, o in g.objects.items() if o.tag in c.drop_tags]
            st["dropped_structure"] = len(gone); self._drop(gone)
            for o in g.objects.values():                             # withhold names the tagger is not sure about
                if o.tag_scores and o.tag_scores[0][1] < c.min_tag_prob:
                    o.tag_guess, o.tag, o.caption = o.tag, "unknown", "unknown object"; st["unknown_name"] += 1
        if geo is not None:
            st["unreliable"] = self._mark_unreliable(geo)
        st["final"] = len(g.objects); st["shown"] = len(g.shown())
        g.build_edges()
        return g

    def save_crops(self, out_dir):
        """Write the tagging crops of every remaining object (masked: m<k>.png, with context: c<k>.png) so names can be
        recomputed later (`scene3d retag`) without running the depth model again."""
        from pathlib import Path
        from .imageio_safe import imwrite_rgb
        out = Path(out_dir)
        for oid in self.graph.objects:
            d = out / f"{oid:03d}"; d.mkdir(parents=True, exist_ok=True)
            for k, (_, mc, cc) in enumerate(self._crops.get(oid, [])):
                imwrite_rgb(d / f"m{k}.png", mc); imwrite_rgb(d / f"c{k}.png", cc)

    def diagnostics(self) -> dict:
        """Numbers that tell WHY the map is fragmented (or not). Medians of best geo/sem among overlapping candidates."""
        st = dict(self.stats); geo, sem = st.pop("geo"), st.pop("sem")
        st["median_best_geo"] = round(float(np.median(geo)), 3) if geo else None
        st["median_best_sem"] = round(float(np.median(sem)), 3) if sem else None
        st["frac_geo_above_0.15"] = round(float(np.mean(np.array(geo) > 0.15)), 3) if geo else None
        return st

    def run(self, frames) -> SceneGraph:
        for fr in frames:
            self.process_frame(fr)
        return self.finalize()
