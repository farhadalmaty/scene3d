import sys, unittest
import numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scene3d import Pipeline, PipelineConfig, AgentTools, SceneGraph
from scene3d.backends.synthetic import (SyntheticWorld, SyntheticSegmenter, SyntheticEmbedder,
                                        SyntheticCaptioner, DEFAULT_WORLD)


def build():
    w = SyntheticWorld(); e = SyntheticEmbedder()
    pipe = Pipeline(SyntheticSegmenter(), e, SyntheticCaptioner(e), PipelineConfig())
    g = pipe.run(w.orbit(12))
    return g, e


class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.g, cls.e = build(); cls.t = AgentTools(cls.g, cls.e)

    def test_object_count_and_tags(self):
        tags = sorted(o.tag for o in self.g.present())
        self.assertEqual(tags, sorted(DEFAULT_WORLD), tags)

    def test_positions_accurate(self):
        for o in self.g.present():
            true_c = np.array(DEFAULT_WORLD[o.tag][0])
            err = np.linalg.norm(o.center - true_c)
            # visible-surface bbox is biased vs. true centre; allow 8 cm
            self.assertLess(err, 0.08, (o.tag, err, o.center, true_c))

    def test_relations(self):
        s = " | ".join(self.t.scene_summary()["relations"])
        self.assertIn("mug", s); self.assertIn("on_top_of", s)

    def test_queries(self):
        m = self.t.where_is("the mug")
        self.assertTrue(m["found"]); self.assertEqual(m["object"]["tag"], "mug")
        a = self.t.find_objects("chair")["matches"][0]["id"]
        b = self.t.find_objects("table")["matches"][0]["id"]
        d = self.t.distance_between(a, b)["centre_distance"]
        self.assertAlmostEqual(d, np.linalg.norm(np.array(DEFAULT_WORLD["chair"][0]) - DEFAULT_WORLD["table"][0]), delta=0.1)

    def test_scale_calibration_and_persistence(self):
        import tempfile
        ids = {o.tag: o.id for o in self.g.present()}
        g2 = SceneGraph.load(self._roundtrip())
        self.assertEqual(len(g2.objects), len(self.g.objects))
        t2 = AgentTools(g2, self.e)
        k = t2.calibrate_scale(ids["chair"], ids["table"], 2.0)["scale_factor"]
        self.assertGreater(k, 0); self.assertTrue(g2.metric_scale)

    def _roundtrip(self):
        import tempfile; d = tempfile.mkdtemp(); self.g.save(d); return d


if __name__ == "__main__":
    unittest.main(verbosity=2)
