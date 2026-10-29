"""Model-free test backends: a rendered box world with exact ground truth.
Everything goes through the same interfaces as real models (colour encodes identity)."""
from __future__ import annotations
import numpy as np
from ..datatypes import Frame

# name: (centre, size, rgb colour)
DEFAULT_WORLD = {
    "table": ((0.0, 0.0, 0.40), (1.20, 0.80, 0.80), (200, 40, 40)),
    "mug":   ((0.20, 0.10, 0.85), (0.08, 0.08, 0.10), (40, 200, 40)),
    "book":  ((-0.30, 0.0, 0.82), (0.20, 0.15, 0.04), (40, 40, 200)),
    "chair": ((1.30, 0.20, 0.25), (0.40, 0.40, 0.50), (200, 200, 40)),
}


def _box_points(c, s, step):
    c, s = np.asarray(c, float), np.asarray(s, float)
    pts = []
    for ax in range(3):
        o = [i for i in range(3) if i != ax]
        g = [np.arange(-s[i] / 2, s[i] / 2 + 1e-9, step) for i in o]
        A, B = np.meshgrid(*g)
        for sign in (-1, 1):
            p = np.zeros((A.size, 3)); p[:, o[0]] = A.ravel(); p[:, o[1]] = B.ravel(); p[:, ax] = sign * s[ax] / 2
            pts.append(p + c)
    return np.vstack(pts)


class SyntheticWorld:
    def __init__(self, world=None, size=(320, 240), f=300.0, step=0.008, depth_noise=0.003, seed=0, texture=0.0):
        self.world = world or DEFAULT_WORLD
        self.W, self.H = size
        self.K = np.array([[f, 0, self.W / 2], [0, f, self.H / 2], [0, 0, 1.0]])
        self.rng = np.random.default_rng(seed)
        self.noise = depth_noise
        self.names = list(self.world)
        self.pts = [_box_points(c, s, step) for c, s, _ in self.world.values()]
        self.texture = texture
        # surface-attached brightness pattern (a function of the 3D position): gives feature detectors something to track
        self.fac = [1 + texture * (np.sin(40 * P[:, 0] + .3) * np.sin(37 * P[:, 1] + 1) * np.sin(43 * P[:, 2] + 2)
                                   + .6 * np.sin(97 * P[:, 0] + .5) * np.sin(89 * P[:, 1] + 1.5) * np.sin(101 * P[:, 2])) for P in self.pts]

    @staticmethod
    def look_at(pos, target, up=(0, 0, 1.0)):
        pos, target, up = map(lambda v: np.asarray(v, float), (pos, target, up))
        f = target - pos; f /= np.linalg.norm(f)
        r = np.cross(f, up); r /= np.linalg.norm(r)
        d = np.cross(f, r)
        T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, d, f, pos
        return T

    def render(self, pose_wc, index) -> Frame:
        R, t = pose_wc[:3, :3], pose_wc[:3, 3]
        zs, us, vs, ids, fs = [], [], [], [], []
        for k, P in enumerate(self.pts):
            pc = (P - t) @ R
            m = pc[:, 2] > 0.2
            pc = pc[m]
            u = self.K[0, 0] * pc[:, 0] / pc[:, 2] + self.K[0, 2]
            v = self.K[1, 1] * pc[:, 1] / pc[:, 2] + self.K[1, 2]
            zs.append(pc[:, 2]); us.append(u); vs.append(v); ids.append(np.full(len(pc), k)); fs.append(self.fac[k][m])
        z, u, v, idk, fk = map(np.concatenate, (zs, us, vs, ids, fs))
        order = np.argsort(-z)                                  # far -> near, near overwrites
        depth = np.zeros((self.H, self.W), np.float32); lab = -np.ones((self.H, self.W), int); fac = np.ones((self.H, self.W))
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                uu = np.round(u[order]).astype(int) + du; vv = np.round(v[order]).astype(int) + dv
                ok = (uu >= 0) & (uu < self.W) & (vv >= 0) & (vv < self.H)
                depth[vv[ok], uu[ok]] = z[order][ok]; lab[vv[ok], uu[ok]] = idk[order][ok]; fac[vv[ok], uu[ok]] = fk[order][ok]
        rgb = np.full((self.H, self.W, 3), 128, np.uint8)
        for k, (_, _, col) in enumerate(self.world.values()):
            mk = lab == k
            rgb[mk] = np.clip(np.array(col)[None, :] * fac[mk][:, None], 0, 255).astype(np.uint8)
        depth[depth > 0] += self.rng.normal(0, self.noise, (depth > 0).sum()).astype(np.float32)
        fr = Frame(index, rgb, depth, self.K, pose_wc)
        fr._label_img = lab   # ground-truth labels, used only by SyntheticSegmenter
        return fr

    def orbit(self, n=12, radius=2.0, height=1.2, target=(0.3, 0.1, 0.5)):
        for i in range(n):
            a = 2 * np.pi * i / n
            pos = (target[0] + radius * np.cos(a), target[1] + radius * np.sin(a), height)
            yield self.render(self.look_at(pos, target), i)


class SyntheticSegmenter:
    """Finds the known flat colours (tolerant, so it also survives lossy video compression)."""
    def __init__(self, world=None, tol=40.0):
        self.cols = [np.array(c[2], float) for c in (world or DEFAULT_WORLD).values()]; self.tol = tol
    def segment(self, rgb):
        px = rgb.astype(float)
        return [np.linalg.norm(px - c, axis=-1) < self.tol for c in self.cols]


class SyntheticEmbedder:
    def __init__(self, world=None, d=64, seed=1, tol=40.0):
        self.world = world or DEFAULT_WORLD; self.tol = tol
        rng = np.random.default_rng(seed)
        self.base = {n: self._unit(rng.normal(size=d)) for n in self.world}
        self.cols = {n: np.array(c[2], float) for n, c in self.world.items()}
        self.rng = np.random.default_rng(seed + 1); self.d = d

    @staticmethod
    def _unit(v): return v / np.linalg.norm(v)
    def _name_px(self, px):
        px = np.asarray(px, float).reshape(-1, 3)
        near = np.min([np.linalg.norm(px - c, axis=1) for c in self.cols.values()], axis=0) < self.tol
        m = (px[near] if near.any() else px).mean(0)
        return min(self.cols, key=lambda n: np.linalg.norm(self.cols[n] - m))
    def _noisy(self, n): return self._unit(self.base[n] + self.rng.normal(0, 0.05, self.d))
    def embed(self, rgb, masks): return np.stack([self._noisy(self._name_px(rgb[m])) for m in masks])
    def embed_crops(self, crops): return np.stack([self._noisy(self._name_px(c)) for c in crops])
    def embed_text(self, text):
        hit = [self.base[n] for n in self.base if n in text.lower()]
        return self._unit(np.sum(hit, 0)) if hit else self._unit(np.random.default_rng(abs(hash(text)) % 2**32).normal(size=self.d))


class SyntheticCaptioner:
    def __init__(self, embedder): self.e = embedder
    def describe(self, crops):
        votes = [self.e._name_px(c) for c in crops]
        n = max(set(votes), key=votes.count)
        return n, f"a {n}"


class SyntheticGeometry:
    """Stands in for a pose/depth network: returns ground truth for the stored frames, matched to the
    (possibly re-encoded) input by nearest image. Lets the whole video pipeline be tested without models."""
    def __init__(self, gt_frames): self.gt = list(gt_frames)
    def estimate(self, rgbs):
        import cv2
        small = lambda a: cv2.resize(a, (40, 30), interpolation=cv2.INTER_AREA).astype(float)
        G = [small(g.rgb) for g in self.gt]
        out = []
        for i, r in enumerate(rgbs):
            j = int(np.argmin([np.abs(small(np.asarray(r)) - g).mean() for g in G]))
            g = self.gt[j]
            out.append(Frame(i, np.asarray(r), g.depth.copy(), g.K.copy(), g.pose_wc.copy()))
        return out
