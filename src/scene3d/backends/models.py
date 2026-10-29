"""Real model backends for CPU / Apple Silicon. NOT exercised in the dev sandbox (no network, no torch):
verify on your machine. Heavy imports are lazy so the core stays light."""
from __future__ import annotations
import numpy as np


def pick_device() -> str:
    """cuda if an NVIDIA GPU + CUDA build of torch is present, otherwise cpu (Windows laptops: usually cpu)."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class FastSamSegmenter:
    """Class-agnostic masks. FastSAM is a light CPU-friendly stand-in for SAM/eSAM (review Sec. 3.1)."""
    def __init__(self, weights="FastSAM-s.pt", imgsz=640, conf=0.4, iou=0.9, device=None):
        from ultralytics import FastSAM
        self.m, self.imgsz, self.conf, self.iou = FastSAM(weights), imgsz, conf, iou
        self.device = device or "cpu"
    def segment(self, rgb):
        r = self.m(rgb, device=self.device, retina_masks=True, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False)[0]
        if r.masks is None: return []
        import cv2
        out = []
        for m in r.masks.data.cpu().numpy():
            if m.shape != rgb.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            out.append(m.astype(bool))
        return out


class ClipEmbedder:
    """CLIP image/text features via open_clip (review Sec. 3.1)."""
    def __init__(self, model="ViT-B-32", pretrained="laion2b_s34b_b79k", device=None):
        import torch, open_clip
        self.torch = torch
        self.device = device or pick_device()
        self.model, _, self.pre = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model.eval().to(self.device); self.tok = open_clip.get_tokenizer(model)
    def _img(self, crops):
        from PIL import Image
        t = self.torch
        x = t.stack([self.pre(Image.fromarray(c)) for c in crops]).to(self.device)
        with t.no_grad(): f = self.model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()
    def embed_crops(self, crops): return self._img(list(crops))
    def embed(self, rgb, masks):
        crops = []
        for m in masks:
            ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            py, px = int((y1 - y0) * .1), int((x1 - x0) * .1)
            crops.append(rgb[max(0, y0 - py):y1 + py, max(0, x0 - px):x1 + px])
        return self._img(crops)
    def embed_text(self, text):
        t = self.torch
        with t.no_grad(): f = self.model.encode_text(self.tok([text]).to(self.device))
        f = f / f.norm(dim=-1, keepdim=True)
        return f[0].float().cpu().numpy()


TEMPLATES = ["a photo of a {}", "a close-up photo of a {}", "a blurry photo of a {}", "a photo of a {} in a room"]
# words that name (almost) the same thing: their probabilities are added, so 'mug' vs 'cup' vs 'glass' does not look unsure
SYNONYM_GROUPS = [("cup", "mug", "glass"), ("tv", "monitor"), ("table", "desk", "coffee table"), ("chair", "armchair"),
                  ("pot", "pan"), ("lamp", "ceiling light"), ("painting", "poster", "picture frame"),
                  ("jacket", "coat", "clothes"), ("bag", "backpack"), ("shelf", "bookcase"),
                  ("cabinet", "dresser", "drawer", "wardrobe", "nightstand", "tv stand"), ("pillow", "blanket")]


class VocabCaptioner:
    """LVLM-free tagging: pick the closest word from a vocabulary by CLIP similarity.

    Closed vocabulary: an object that is not in the list gets the NEAREST word, so the confidence matters. Confidence =
    probability of the winning word's synonym group (softmax over all words, CLIP logit scale 100). Text prompts are
    averaged over several templates (a standard CLIP trick)."""
    def __init__(self, embedder: ClipEmbedder, vocab: list[str], top_k: int = 3, templates=TEMPLATES, groups=SYNONYM_GROUPS):
        self.e, self.vocab, self.top_k = embedder, vocab, top_k
        T = []
        for w in vocab:
            v = np.mean([embedder.embed_text(t.format(w)) for t in templates], axis=0)
            T.append(v / (np.linalg.norm(v) + 1e-9))
        self.T = np.stack(T)
        gid = {}
        for k, grp in enumerate(groups):
            for w in grp: gid[w] = k
        self.group = np.array([gid.get(w, 1000 + i) for i, w in enumerate(vocab)])     # singletons get their own group

    def describe_scored(self, crops):
        f = self.e.embed_crops(crops).mean(0); f /= np.linalg.norm(f)
        z = 100.0 * (self.T @ f); z -= z.max(); p = np.exp(z); p /= p.sum()
        gp = {g: float(p[self.group == g].sum()) for g in np.unique(self.group)}
        ranked = sorted(gp, key=lambda g: -gp[g])[: self.top_k]
        top = []
        for g in ranked:
            members = np.nonzero(self.group == g)[0]
            top.append([self.vocab[int(members[np.argmax(p[members])])], round(gp[g], 3)])
        return top[0][0], f"a {top[0][0]}", top
    def describe(self, crops):
        w, cap, _ = self.describe_scored(crops); return w, cap


COMMON_VOCAB = [
    # furniture
    "chair", "armchair", "stool", "bench", "table", "desk", "coffee table", "sofa", "bed", "shelf", "bookcase", "cabinet",
    "dresser", "drawer", "nightstand", "wardrobe", "tv stand",
    # appliances and kitchen
    "refrigerator", "microwave", "oven", "stove", "sink", "toaster", "kettle", "coffee machine", "washing machine",
    "air conditioner", "radiator", "fan", "pot", "pan", "plate", "bowl", "cup", "mug", "glass", "bottle", "cutting board",
    # electronics
    "tv", "monitor", "laptop", "keyboard", "mouse", "phone", "remote", "speaker", "printer", "router", "camera", "headphones",
    "charger", "cable", "lamp", "ceiling light",
    # bathroom and textiles
    "toilet", "bathtub", "shower", "mirror", "towel", "curtain", "pillow", "blanket", "rug", "clothes", "jacket", "coat", "hat",
    "shoe", "hanger",
    # misc
    "book", "bag", "backpack", "suitcase", "box", "basket", "bucket", "plant", "vase", "clock", "painting", "poster",
    "picture frame", "trash can", "umbrella", "fire extinguisher", "door", "window", "stairs",
    # structure (dropped from the map)
    "wall", "floor", "ceiling"]   # wall/floor/ceiling are dropped as structure


from dataclasses import dataclass


@dataclass
class Backends:
    geometry: object
    segmenter: object
    embedder: object
    captioner: object | None = None


def default_backends(device: str | None = None, vocab: list[str] | None = None,
                     geometry_model: str = "depth-anything/da3-small", fastsam_weights: str = "FastSAM-s.pt") -> Backends:
    """Real models for a laptop: DA3-small (pose+depth), FastSAM (masks), CLIP ViT-B/32 (features), vocab tags."""
    from .geometry_da3 import DepthAnything3Geometry
    device = device or pick_device()
    emb = ClipEmbedder(device=device)
    return Backends(
        geometry=DepthAnything3Geometry(geometry_model, device=device),
        segmenter=FastSamSegmenter(fastsam_weights, device=device),
        embedder=emb,
        captioner=VocabCaptioner(emb, vocab or COMMON_VOCAB))
