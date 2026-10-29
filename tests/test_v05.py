import os, sys, tempfile, types, unittest, json
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scene3d import Pipeline, PipelineConfig, AgentTools, SceneGraph
from scene3d.annotate import object_overlays
from scene3d.diagnose import inspect_scene
from scene3d.retag import retag_scene, load_crops
from scene3d.video import sample_keyframes
from scene3d.backends.geometry_da3 import DepthAnything3Geometry
from scene3d.backends.models import VocabCaptioner, Backends
from scene3d.backends.synthetic import (SyntheticWorld, SyntheticSegmenter, SyntheticEmbedder, SyntheticCaptioner,
                                        SyntheticGeometry, DEFAULT_WORLD)
from scene3d.service.processing import process_video, ProcessConfig
from test_service import make_video


class ScoredCaptioner:
    """SyntheticCaptioner + a fixed confidence, like the real tagger."""
    def __init__(self, emb, conf=0.9): self.c = SyntheticCaptioner(emb); self.conf = conf
    def describe(self, crops): return self.c.describe(crops)
    def describe_scored(self, crops):
        t, cap = self.c.describe(crops); return t, cap, [[t, self.conf], ["other", 1 - self.conf]]


class SpyEmbedder(SyntheticEmbedder):
    def __init__(self, *a, **k): super().__init__(*a, **k); self.seen = []
    def embed(self, rgb, masks): self.seen.append((rgb.shape[:2], masks[0].shape if masks else None)); return super().embed(rgb, masks)


def hi_frames(world, n=12, scale=2):
    out = []
    for fr in world.orbit(n):
        fr.rgb_hi = cv2.resize(fr.rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST); out.append(fr)
    return out


class TestHighResolution(unittest.TestCase):
    def test_recognition_runs_on_hires_while_3d_uses_depth_resolution(self):
        w = SyntheticWorld(); e = SpyEmbedder()
        seg = SyntheticSegmenter(); p = Pipeline(seg, e, SyntheticCaptioner(e), PipelineConfig())
        g = p.run(hi_frames(w))
        self.assertEqual(sorted(o.tag for o in g.present()), sorted(DEFAULT_WORLD))
        self.assertEqual(e.seen[0][0], (480, 640))                               # embedder saw the 2x frame
        for o in g.present():
            for fi, x0, y0, x1, y1 in o.obs: self.assertTrue(0 <= x0 < x1 <= 320 and 0 <= y0 < y1 <= 240)   # depth-res boxes
        crop = p._crops[next(iter(g.objects))][0][1]
        self.assertGreater(crop.shape[0], 10)

    def test_two_pass_sampling_returns_matching_hires_frames(self):
        d = tempfile.mkdtemp(); vp, gt, b = make_video(d, 36)
        lo, meta = sample_keyframes(vp, 8, max_side=200, return_meta=True, hi_side=320)
        self.assertEqual(len(lo), 8); self.assertEqual(len(meta["hi_frames"]), 8)
        self.assertEqual(max(lo[0].shape[:2]), 200); self.assertEqual(max(meta["hi_frames"][0].shape[:2]), 320)
        self.assertEqual(sorted(meta["indices"]), meta["indices"])

    def test_depth_model_resolution_is_forwarded(self):
        calls = {}
        class FakeModel:
            def inference(self, imgs, process_res=None):
                calls["res"] = process_res; n = len(imgs)
                return types.SimpleNamespace(depth=np.ones((n, 8, 6), np.float32), conf=np.ones((n, 8, 6)),
                                             extrinsics=np.tile(np.eye(4)[:3], (n, 1, 1)), intrinsics=np.tile(np.eye(3), (n, 1, 1)))
        g = DepthAnything3Geometry.__new__(DepthAnything3Geometry); g.model = FakeModel(); g.conf_percentile = 20.0; g.process_res = 504
        g.process_res = 756; g.estimate([np.zeros((16, 12, 3), np.uint8)])
        self.assertEqual(calls["res"], 756)


class TestConfidenceGate(unittest.TestCase):
    def _graph(self, conf):
        w = SyntheticWorld(); e = SyntheticEmbedder()
        p = Pipeline(SyntheticSegmenter(), e, ScoredCaptioner(e, conf), PipelineConfig(min_tag_prob=0.35))
        return p.run(w.orbit(12)), e, p

    def test_unsure_names_are_withheld_and_hidden(self):
        g, e, p = self._graph(0.2)
        self.assertTrue(all(o.tag == "unknown" for o in g.present())); self.assertEqual(p.stats["unknown_name"], 4)
        self.assertTrue(all(o.tag_guess in DEFAULT_WORLD for o in g.present()))      # the guess is kept for inspection
        t = AgentTools(g, e)
        self.assertEqual(t.list_objects()["count"], 0); self.assertEqual(t.list_objects()["hidden"]["unknown_name"], 4)
        self.assertEqual(t.list_objects(include_unknown=True)["count"], 4)
        self.assertEqual(g.shown(), [])
        self.assertEqual(t.where_is("mug")["found"], False)

    def test_confident_names_are_kept(self):
        g, e, p = self._graph(0.9)
        self.assertEqual(sorted(o.tag for o in g.shown()), sorted(DEFAULT_WORLD)); self.assertEqual(p.stats["unknown_name"], 0)
        card = AgentTools(g, e).list_objects()["objects"][0]; self.assertEqual(card["tag_confidence"], 0.9)

    def test_labels_skip_unknown_unless_asked(self):
        g, e, p = self._graph(0.2); geo_pose = np.eye(4); K = np.array([[300., 0, 160], [0, 300., 120], [0, 0, 1]])
        pose = SyntheticWorld.look_at((2.3, 0.1, 1.2), (0.3, 0.1, 0.5))
        self.assertEqual(object_overlays(g, pose, K, 320, 240), [])
        self.assertGreater(len(object_overlays(g, pose, K, 320, 240, show_unknown=True)), 0)


class FakeClip:
    """Embedder for VocabCaptioner tests: each 'word' is a fixed random unit vector; cup/mug/glass are near-identical."""
    def __init__(self):
        rng = np.random.default_rng(0); self.v = {}
        base = rng.normal(size=32); base /= np.linalg.norm(base)
        for w in ("cup", "mug", "glass"):
            x = base + 0.03 * rng.normal(size=32); self.v[w] = x / np.linalg.norm(x)
        for w in ("towel", "bottle", "lamp", "book"):
            x = rng.normal(size=32); self.v[w] = x / np.linalg.norm(x)
        self.text_calls = 0
    def embed_text(self, t):
        self.text_calls += 1
        word = next(w for w in self.v if f" {w}" in t or t.endswith(w)); return self.v[word]
    def embed_crops(self, crops): return np.stack([self.v[c] for c in crops])      # a 'crop' is just the word here


class TestTagger(unittest.TestCase):
    def test_synonyms_do_not_look_unsure_and_templates_are_averaged(self):
        emb = FakeClip(); cap = VocabCaptioner(emb, ["cup", "mug", "glass", "towel", "bottle", "lamp", "book"])
        self.assertEqual(emb.text_calls, 7 * 4)                                           # 4 prompt templates per word
        tag, _, scores = cap.describe_scored(["mug"])
        self.assertIn(tag, ("cup", "mug", "glass")); self.assertGreater(scores[0][1], 0.9)   # group probability adds up
        tag2, _, s2 = cap.describe_scored(["towel"]); self.assertEqual(tag2, "towel"); self.assertGreater(s2[0][1], 0.9)
        # ambiguous crop (average of towel and bottle): low confidence for the winner
        mixed = cap.e.embed_crops
        cap.e.embed_crops = lambda crops: np.stack([(emb.v["towel"] + emb.v["bottle"]) / 2])
        _, _, s3 = cap.describe_scored(["x"]); self.assertLess(s3[0][1], 0.75)
        cap.e.embed_crops = mixed


class TestReliabilityAndRetag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(); vp, cls.gt, b = make_video(cls.d, 36)
        e = b.embedder; cls.b = Backends(b.geometry, b.segmenter, e, ScoredCaptioner(e, 0.9))
        cls.sc = os.path.join(cls.d, "s"); cls.g = process_video(vp, cls.sc, cls.b, ProcessConfig(n_frames=12))

    def test_inconsistent_objects_are_hidden_but_inspectable(self):
        d = tempfile.mkdtemp(); vp, gt, b = make_video(d, 36); e = b.embedder
        p = Pipeline(b.segmenter, e, ScoredCaptioner(e, 0.9), PipelineConfig())
        from scene3d.video import sample_keyframes
        rgbs = sample_keyframes(vp, 12); frames = b.geometry.estimate(rgbs)
        for fr in frames: p.process_frame(fr)
        victim = min(p.graph.objects.values(), key=lambda o: np.linalg.norm(o.extent)); victim.points = victim.points + np.array([0.9, 0.0, 0.0])    # wrong place (the mug)
        geo = {"poses_wc": np.stack([f.pose_wc for f in frames]), "K": np.stack([f.K for f in frames]), "hw": np.array([f.depth.shape for f in frames])}
        g = p.finalize(geo)
        self.assertEqual(victim.status, "unreliable"); self.assertEqual(p.stats["unreliable"], 1)
        self.assertNotIn(victim.id, [o.id for o in g.shown()]); self.assertEqual(len(g.shown()), 3)

    def test_crops_are_saved_and_retag_changes_names_without_the_depth_model(self):
        crops = os.path.join(self.sc, "crops"); self.assertTrue(os.path.isdir(crops))
        oid = next(iter(self.g.present())).id
        m, c, both = (len(load_crops(crops, oid, k)) for k in ("masked", "context", "both"))
        self.assertGreater(m, 0); self.assertEqual(both, m + c)
        e = self.b.embedder
        r = retag_scene(self.sc, ScoredCaptioner(e, 0.9), 0.35, "masked", write=False)
        self.assertEqual(r["unknown"], 0); self.assertEqual(sorted(r["tags"]), sorted(DEFAULT_WORLD))
        r2 = retag_scene(self.sc, ScoredCaptioner(e, 0.2), 0.35, "masked", write=True)            # stricter / unsure tagger
        self.assertEqual(r2["unknown"], 4)
        g2 = SceneGraph.load(os.path.join(self.sc, "graph")); self.assertTrue(all(o.tag == "unknown" for o in g2.present()))
        rows = inspect_scene(self.sc)                                                              # hidden objects stay inspectable
        self.assertEqual(len(rows), 4); self.assertTrue(all(r["tag"] == "unknown" and r["tag_guess"] for r in rows))

    def test_retag_without_crops_gives_clear_error(self):
        d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "graph")); SceneGraph().save(os.path.join(d, "graph"))
        with self.assertRaises(RuntimeError): retag_scene(d, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
