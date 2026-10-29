import os, sys, tempfile, time, types, unittest, json
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scene3d import AgentTools, SceneGraph
from scene3d.backends.synthetic import (SyntheticWorld, SyntheticSegmenter, SyntheticEmbedder, SyntheticCaptioner,
                                        SyntheticGeometry, DEFAULT_WORLD)
from scene3d.backends.models import Backends
from scene3d.service.processing import process_video, ProcessConfig
from scene3d.service.core import SceneService
from scene3d.agent_chat import ChatAgent
from scene3d.tool_schemas import tool_definitions
from scene3d.alignment import rotation_to_z, up_from_cameras, align_gravity
from scene3d.imageio_safe import open_video, imwrite_rgb, imread_rgb


def make_video(d, n=36):
    w = SyntheticWorld(); gt = list(w.orbit(n)); vp = os.path.join(d, "v.avi")
    vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for f in gt: vw.write(cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR))
    vw.release()
    e = SyntheticEmbedder()
    return vp, gt, Backends(SyntheticGeometry(gt), SyntheticSegmenter(), e, SyntheticCaptioner(e))


class TestVideoPipeline(unittest.TestCase):
    def test_video_to_graph(self):
        d = tempfile.mkdtemp(); vp, gt, b = make_video(d)
        stages = []
        g = process_video(vp, os.path.join(d, "out"), b, ProcessConfig(n_frames=12), lambda s, f, m="": stages.append(s))
        self.assertEqual(sorted(o.tag for o in g.present()), sorted(DEFAULT_WORLD))
        self.assertTrue({"frames", "geometry", "objects", "finalize"} <= set(stages))
        s = " | ".join(AgentTools(g).scene_summary()["relations"])
        self.assertIn("on_top_of", s)
        self.assertTrue(os.path.exists(os.path.join(d, "out", "graph", "scene.json")))
        self.assertTrue(os.path.exists(os.path.join(d, "out", "frames", "00000.png")))
        # map is z-up and heights are non-negative after alignment
        zs = np.concatenate([o.points[:, 2] for o in g.present()])
        self.assertGreater(zs.min(), -0.3)

    def test_gravity_alignment_recovers_tilted_world(self):
        w = SyntheticWorld(); frames = list(w.orbit(12))
        rng = np.random.default_rng(3); ax = np.array([1.0, 0.5, 0.2]); ax /= np.linalg.norm(ax); a = np.deg2rad(40)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K           # tilt the world by 40 deg
        T = np.eye(4); T[:3, :3] = R
        for f in frames: f.pose_wc = T @ f.pose_wc
        align_gravity(frames, refine=True)
        up_after = -np.mean([f.pose_wc[:3, 1] for f in frames], 0); up_after /= np.linalg.norm(up_after)
        self.assertGreater(up_after[2], 0.97)        # camera-down axes now point to -Z (within ~14 deg)

    def test_rotation_to_z(self):
        for up in ([0, -1, 0], [0.3, -0.9, 0.1], [0, 0, 1], [0, 0, -1]):
            u = np.array(up, float); u /= np.linalg.norm(u)
            np.testing.assert_allclose(rotation_to_z(u) @ u, [0, 0, 1], atol=1e-9)


class TestWindowsSafety(unittest.TestCase):
    def test_cyrillic_tags_roundtrip_utf8(self):
        g = SceneGraph(); o = g.add(np.random.rand(40, 3), np.ones(4), 0); o.tag = "чашка"; o.caption = "красная чашка"
        d = tempfile.mkdtemp(); g.save(d)
        raw = open(os.path.join(d, "scene.json"), "rb").read().decode("utf-8")
        self.assertIn("красная чашка", raw)
        self.assertEqual(SceneGraph.load(d).objects[o.id].tag, "чашка")

    def test_unicode_image_path(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, "кадр.png")
        img = (np.random.rand(20, 30, 3) * 255).astype(np.uint8); imwrite_rgb(p, img)
        np.testing.assert_array_equal(imread_rgb(p), img)

    def test_non_ascii_video_path_rejected_clearly(self):
        with self.assertRaises(ValueError) as cm: open_video("C:\\Users\\Фархад\\v.mp4")
        self.assertIn("non-ASCII", str(cm.exception))


class TestService(unittest.TestCase):
    def test_job_lifecycle_and_queries(self):
        d = tempfile.mkdtemp(); vp, gt, b = make_video(d)
        svc = SceneService(os.path.join(d, "data"), backends_factory=lambda: b, config=ProcessConfig(n_frames=12))
        sid = svc.submit_video(vp, "demo")
        meta = svc.wait(sid, 120)
        self.assertEqual(meta["status"], "done", meta); self.assertEqual(meta["n_objects"], 4)
        t = svc.tools(sid)
        self.assertEqual(t.where_is("the mug")["object"]["tag"], "mug")
        self.assertEqual(len(svc.store.list()), 1)
        svc.shutdown()

    def test_failed_job_reports_error(self):
        d = tempfile.mkdtemp(); bad = os.path.join(d, "bad.mp4"); open(bad, "wb").write(b"not a video")
        svc = SceneService(os.path.join(d, "data"), backends_factory=lambda: None)
        sid = svc.submit_video(bad)
        meta = svc.wait(sid, 60)
        self.assertEqual(meta["status"], "error"); self.assertTrue(meta["error"])
        with self.assertRaises(RuntimeError): svc.tools(sid)
        svc.shutdown()


class FakeClient:
    """Scripted stand-in for anthropic.Anthropic(): first asks for a tool, then answers."""
    def __init__(self): self.calls = []; self.messages = self
    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            blk = types.SimpleNamespace(type="tool_use", id="tu_1", name="where_is", input={"query": "mug"})
            return types.SimpleNamespace(stop_reason="tool_use", content=[blk])
        txt = types.SimpleNamespace(type="text", text="The mug is on the table.")
        return types.SimpleNamespace(stop_reason="end_turn", content=[txt])


class TestAgent(unittest.TestCase):
    def test_tool_schemas(self):
        defs = {t["name"]: t for t in tool_definitions()}
        self.assertEqual(len(defs), 11)
        self.assertEqual(defs["update_position"]["input_schema"]["properties"]["new_center"],
                         {"type": "array", "items": {"type": "number"}})
        self.assertEqual(defs["find_objects"]["input_schema"]["required"], ["query"])
        self.assertEqual(defs["calibrate_scale"]["input_schema"]["properties"]["real_distance_m"]["type"], "number")

    def test_chat_tool_loop(self):
        d = tempfile.mkdtemp(); vp, gt, b = make_video(d)
        g = process_video(vp, os.path.join(d, "o"), b, ProcessConfig(n_frames=12))
        fc = FakeClient(); agent = ChatAgent(AgentTools(g, b.embedder), client=fc)
        self.assertEqual(agent.ask("where is the mug?"), "The mug is on the table.")
        msgs = agent.messages                      # user, assistant(tool_use), user(tool_result), assistant(text)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user", "assistant"])
        res = msgs[2]["content"][0]
        self.assertEqual(res["type"], "tool_result"); self.assertEqual(res["tool_use_id"], "tu_1"); self.assertFalse(res["is_error"])
        payload = json.loads(res["content"])
        self.assertTrue(payload["found"]); self.assertEqual(payload["object"]["tag"], "mug")

    def test_tool_error_is_returned_not_raised(self):
        g = SceneGraph(); agent = ChatAgent(AgentTools(g), client=FakeClient())
        out, err = agent._run_tool("get_object", {"object_id": 99})
        self.assertTrue(err); self.assertIn("no object", out)


class JunkSegmenter:
    """Perfect masks PLUS what real FastSAM also returns: a wall-sized mask, specks on background without depth,
    and part-masks (a fragment lying inside a single real object)."""
    def __init__(self, inner): self.inner = inner; self.rng = np.random.default_rng(5)
    def segment(self, rgb):
        masks = self.inner.segment(rgb)
        H, W = rgb.shape[:2]
        bg = np.all(np.abs(rgb.astype(int) - 128) < 3, axis=-1)
        out = list(masks) + [np.ones((H, W), bool)]                           # 'wall'
        for _ in range(3):                                                     # specks over background
            m = np.zeros((H, W), bool); y, x = self.rng.integers(5, H - 40), self.rng.integers(5, W - 40)
            m[y:y + 25, x:x + 25] = True; out.append(m & bg)
        for m in masks:                                                        # fragments inside one object
            if m.sum() > 400:
                ys, xs = np.nonzero(m); cut = np.median(xs); out.append(m & (np.arange(W)[None, :] < cut))
        return out


class TestFilters(unittest.TestCase):
    def test_junk_masks_do_not_pollute_the_map(self):
        from scene3d import Pipeline, PipelineConfig
        w = SyntheticWorld(); e = SyntheticEmbedder()
        p = Pipeline(JunkSegmenter(SyntheticSegmenter()), e, SyntheticCaptioner(e), PipelineConfig())
        for fr in w.orbit(24): p.process_frame(fr)
        g = p.finalize(); d = p.diagnostics()
        self.assertEqual(sorted(o.tag for o in g.present()), sorted(DEFAULT_WORLD))
        self.assertGreater(d["too_big"], 0)                                   # wall masks rejected
        self.assertIn("median_best_geo", d); self.assertEqual(d["final"], 4)
        for o in g.present():                                                 # fragments must not inflate the objects
            true = np.array(DEFAULT_WORLD[o.tag][1]); self.assertLess(np.abs(o.extent - true).max(), 0.12, (o.tag, o.extent))

    def test_singleton_candidates_are_dropped(self):
        from scene3d import Pipeline, PipelineConfig
        e = SyntheticEmbedder(); p = Pipeline(SyntheticSegmenter(), e, None, PipelineConfig(min_obs=3))
        rng = np.random.default_rng(1)
        for i in range(5):
            o = p.graph.add(rng.uniform(0, 1, (60, 3)) + i * 3, e.base["mug"], 0); o.n_obs = 1 if i < 3 else 4
        g = p.finalize()
        self.assertEqual(len(g.objects), 2); self.assertEqual(p.stats["after_min_obs"], 2)

    def test_depth_edge_filter_removes_jumps_only(self):
        from scene3d.alignment import clean_depth_edges
        d = np.full((40, 40), 2.0, np.float32); d[:, 20:] = 3.5                # a depth step (object border)
        out = clean_depth_edges(d, 0.06)
        self.assertEqual(out[10, 5], 2.0); self.assertEqual(out[10, 35], 3.5)  # flat areas untouched
        self.assertEqual(out[10, 20], 0.0)                                     # the jump is invalidated

    def test_small_object_not_swallowed_by_support(self):
        # mug stands on the table: tolerance must follow the SMALLER object, so they stay separate
        from scene3d.association import similarity_parts
        from scene3d.datatypes import Detection, ObjectNode
        rng = np.random.default_rng(0)
        table = ObjectNode(0, np.column_stack([rng.uniform(-.6, .6, 4000), rng.uniform(-.4, .4, 4000), np.full(4000, .8)]), np.array([1., 0]))
        mug_pts = np.column_stack([rng.uniform(.15, .23, 300), rng.uniform(.06, .14, 300), rng.uniform(.8, .9, 300)])
        det = Detection(1, np.zeros((2, 2), bool), mug_pts, np.array([0., 1.]))
        geo, sem = similarity_parts(det, table, 0.03)
        self.assertLess(geo + sem, 1.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
