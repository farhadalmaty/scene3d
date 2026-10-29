import os, sys, tempfile, csv, json, unittest
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scene3d import SceneGraph
from scene3d.compress import compress_video, ffmpeg_command, prepare_source, find_ffmpeg
from scene3d.diagnose import inspect_scene, object_consistency, project_box
from scene3d.service.processing import process_video, ProcessConfig
from scene3d.annotate import load_geometry
from scene3d.backends.synthetic import DEFAULT_WORLD
from test_service import make_video
from test_track import patch_rotation


class TestCompress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(); cls.src = os.path.join(cls.d, "big.mp4")
        rng = np.random.default_rng(0); W, H = 1280, 720
        base = cv2.resize((rng.random((H // 8, W // 8, 3)) * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
        vw = cv2.VideoWriter(cls.src, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        for i in range(60):
            f = np.roll(base, i * 6, axis=1).copy(); f[:60, :200] = 255; vw.write(f)
        vw.release()

    def _probe(self, p):
        cap = cv2.VideoCapture(p); ok, fr = cap.read(); n = 1 if ok else 0
        while cap.read()[0]: n += 1
        return fr, n, cap.get(cv2.CAP_PROP_FPS)

    def test_opencv_engine_shrinks_and_keeps_aspect(self):
        dst = os.path.join(self.d, "o.mp4"); r = compress_video(self.src, dst, max_side=640, fps=15, engine="opencv")
        fr, n, fps = self._probe(dst)
        self.assertEqual(fr.shape[:2], (360, 640)); self.assertEqual(n, 30); self.assertAlmostEqual(fps, 15, delta=0.5)
        self.assertGreater(r["ratio"], 2)

    @unittest.skipUnless(find_ffmpeg(), "ffmpeg not available")
    def test_ffmpeg_engine_shrinks_much_more(self):
        dst = os.path.join(self.d, "f.mp4"); r = compress_video(self.src, dst, max_side=640, fps=15, engine="ffmpeg")
        fr, n, fps = self._probe(dst)
        self.assertEqual(r["engine"], "ffmpeg"); self.assertEqual(fr.shape[:2], (360, 640)); self.assertGreater(r["ratio"], 8)

    def test_rotation_flag_is_honoured_by_both_engines(self):
        src = os.path.join(self.d, "rot.mp4"); import shutil; shutil.copy(self.src, src); patch_rotation(src, 90)
        engines = ["opencv"] + (["ffmpeg"] if find_ffmpeg() else [])
        for eng in engines:
            dst = os.path.join(self.d, f"rot_{eng}.mp4"); compress_video(src, dst, max_side=640, fps=15, engine=eng)
            fr, n, _ = self._probe(dst); h, w = fr.shape[:2]
            self.assertEqual((w, h), (360, 640), eng)                              # portrait after rotation
            self.assertGreater(fr[:int(h * .12), int(w * .9):].mean(), 150, eng)  # white block top-left -> top-right

    def test_ffmpeg_command_shape(self):
        cmd = ffmpeg_command("ffmpeg", "in.mov", "out.mp4", 960, 15, 28, 20)
        s = " ".join(cmd)
        for part in ("-crf 28", "libx264", "fps=15", "min(960,iw)", "-an", "-t 20", "yuv420p"): self.assertIn(part, s)

    def test_auto_compress_only_when_needed(self):
        p, info = prepare_source(self.src, os.path.join(self.d, "a"), True, 5000.0, engine="opencv"); self.assertIsNone(info); self.assertEqual(str(p), self.src)
        p, info = prepare_source(self.src, os.path.join(self.d, "b"), True, 0.01, engine="opencv"); self.assertEqual(info["reason"], "large file"); self.assertTrue(p.exists())
        p, info = prepare_source(self.src, os.path.join(self.d, "c"), False, 0.01); self.assertIsNone(info)

    def test_process_video_compresses_and_records_it(self):
        d = tempfile.mkdtemp(); vp, gt, b = make_video(d, 36)
        cfg = ProcessConfig(n_frames=12, compress_threshold_mb=0.0001, compress_max_side=320, compress_fps=10)
        process_video(vp, os.path.join(d, "o"), b, cfg)
        info = json.load(open(os.path.join(d, "o", "process_info.json"), encoding="utf-8"))
        self.assertTrue(info["video"].endswith("source.mp4")); self.assertIn("engine", info["compression"])


class TestObservationsAndInspect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(); vp, cls.gt, b = make_video(cls.d, 36)
        cls.sc = os.path.join(cls.d, "s"); cls.g = process_video(vp, cls.sc, b, ProcessConfig(n_frames=12))

    def test_observations_are_recorded_and_persisted(self):
        for o in self.g.present():
            self.assertGreaterEqual(len(o.obs), 3); self.assertEqual(len(o.obs[0]), 5)
        g2 = SceneGraph.load(os.path.join(self.sc, "graph")); o = next(iter(g2.present()))
        self.assertEqual(len(o.obs), len(self.g.objects[o.id].obs))

    def test_geometry_is_consistent_with_masks_on_a_good_scene(self):
        geo = load_geometry(self.sc)
        for o in self.g.present():
            c = object_consistency(o, geo)
            self.assertGreater(c["median_cover"], 0.6, (o.tag, c["median_cover"]))

    def test_displaced_object_is_ranked_most_suspicious_and_explained(self):
        g = SceneGraph.load(os.path.join(self.sc, "graph"))
        victim = next(o for o in g.present() if o.tag == "mug"); victim.points = victim.points + np.array([0.9, 0.0, 0.0])   # wrong place
        g.save(os.path.join(self.sc, "graph"))
        rows = inspect_scene(self.sc, ids=[victim.id], worst=2)
        self.assertEqual(rows[0]["id"], victim.id); self.assertIn("POSE/TIME", rows[0]["verdict"])
        self.assertTrue(all(r["verdict"] == "ok" for r in rows[1:]))
        out = os.path.join(self.sc, "inspect")
        self.assertTrue(os.path.exists(os.path.join(out, f"obj_{victim.id:03d}_mug.png")))
        with open(os.path.join(out, "ranking.csv"), encoding="utf-8-sig") as f: self.assertEqual(len(list(csv.DictReader(f))), len(rows))

    def test_old_scene_without_observations_gives_clear_error(self):
        sc2 = os.path.join(self.d, "old"); import shutil; shutil.copytree(self.sc, sc2)
        p = os.path.join(sc2, "graph", "scene.json"); m = json.load(open(p, encoding="utf-8"))
        for o in m["objects"]: o["obs"] = []
        json.dump(m, open(p, "w", encoding="utf-8"))
        with self.assertRaises(RuntimeError): inspect_scene(sc2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
