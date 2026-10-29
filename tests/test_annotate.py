import os, sys, tempfile, unittest
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scene3d import SceneGraph
from scene3d.annotate import (interp_pose, object_overlays, draw_overlays, annotate_video, annotate_keyframes,
                              load_geometry)
from scene3d.service.processing import process_video, ProcessConfig
from scene3d.backends.synthetic import DEFAULT_WORLD
from test_service import make_video


def patch(cx, cy, z, half, n=40):
    g = np.linspace(-half, half, n)
    X, Y = np.meshgrid(g, g)
    return np.column_stack([cx + X.ravel(), cy + Y.ravel(), np.full(X.size, z)])


class TestProjection(unittest.TestCase):
    K = np.array([[300., 0, 160], [0, 300., 120], [0, 0, 1]])

    def test_occlusion_between_objects(self):
        g = SceneGraph()
        near = g.add(patch(0, 0, 2.0, 0.5), np.ones(3), 0); near.tag = "near"; near.n_obs = 5
        far = g.add(patch(0, 0, 4.0, 0.15), np.ones(3), 0); far.tag = "far"; far.n_obs = 5     # directly behind
        side = g.add(patch(1.6, 0, 4.0, 0.15), np.ones(3), 0); side.tag = "side"; side.n_obs = 5  # behind, but off to the side
        ov = {d["tag"] for d in object_overlays(g, np.eye(4), self.K, 320, 240)}
        self.assertIn("near", ov); self.assertIn("side", ov); self.assertNotIn("far", ov)

    def test_object_behind_camera_and_only_filter(self):
        g = SceneGraph(); a = g.add(patch(0, 0, -3.0, 0.3), np.ones(3), 0); a.tag = "back"
        b = g.add(patch(0, 0, 2.0, 0.3), np.ones(3), 0); b.tag = "front"
        self.assertEqual([d["tag"] for d in object_overlays(g, np.eye(4), self.K, 320, 240)], ["front"])
        self.assertEqual(object_overlays(g, np.eye(4), self.K, 320, 240, only={"nothing"}), [])

    def test_interp_pose(self):
        a = np.eye(4); b = np.eye(4); b[:3, 3] = [2, 0, 0]
        c, s = np.cos(np.pi / 2), np.sin(np.pi / 2); b[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        P = np.stack([a, b]); k = np.array([0, 10])
        np.testing.assert_allclose(interp_pose(P, k, 0), a); np.testing.assert_allclose(interp_pose(P, k, 10), b)
        m = interp_pose(P, k, 5)
        np.testing.assert_allclose(m[:3, 3], [1, 0, 0]); np.testing.assert_allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-9)
        ang = np.degrees(np.arccos((np.trace(m[:3, :3]) - 1) / 2)); self.assertAlmostEqual(ang, 45, delta=0.5)
        np.testing.assert_allclose(interp_pose(P, k, 99), b)                 # holds last keyframe outside the range

    def test_draw_keeps_shape_and_changes_pixels(self):
        img = np.full((240, 320, 3), 100, np.uint8)
        out = draw_overlays(img, [{"id": 1, "tag": "чашка", "box": (50, 60, 150, 140), "vis": 1.0, "depth": 2.0}])
        self.assertEqual(out.shape, img.shape); self.assertGreater(np.abs(out.astype(int) - img).sum(), 0)


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(); cls.vp, cls.gt, b = make_video(cls.d, 36)
        cls.scene = os.path.join(cls.d, "scene")
        cls.g = process_video(cls.vp, cls.scene, b, ProcessConfig(n_frames=12))

    def test_labels_land_on_the_right_pixels_in_keyframes(self):
        geo = load_geometry(self.scene); col = np.array(DEFAULT_WORLD["mug"][2], float)
        checked = 0
        for i in range(len(geo["poses_wc"])):
            rgb = cv2.cvtColor(cv2.imread(os.path.join(self.scene, "frames", f"{i:05d}.png")), cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            ys, xs = np.nonzero(np.linalg.norm(rgb.astype(float) - col, axis=-1) < 40)
            if len(xs) < 40: continue                                          # mug not visible in this frame
            true_box = (xs.min(), ys.min(), xs.max(), ys.max())
            mug = [d for d in object_overlays(self.g, geo["poses_wc"][i], geo["K"][i], W, H) if d["tag"] == "mug"]
            self.assertTrue(mug, f"mug visible in keyframe {i} but not labelled")
            x0, y0, x1, y1 = mug[0]["box"]; a, b, c, e = true_box
            inter = max(0, min(x1, c) - max(x0, a)) * max(0, min(y1, e) - max(y0, b))
            union = (x1 - x0) * (y1 - y0) + (c - a) * (e - b) - inter
            self.assertGreater(inter / union, 0.4, (i, mug[0]["box"], true_box)); checked += 1
        self.assertGreaterEqual(checked, 3)

    def test_annotate_video_writes_labelled_copy(self):
        out = os.path.join(self.d, "annotated.mp4")
        r = annotate_video(self.vp, self.scene, out)
        self.assertEqual(r["frames"], 36); self.assertGreater(r["labels_drawn"], 36)
        cap = cv2.VideoCapture(out); n = 0; changed = 0
        src = cv2.VideoCapture(self.vp)
        while True:
            ok, f = cap.read(); ok2, s = src.read()
            if not ok or not ok2: break
            n += 1; changed += int(np.abs(f.astype(int) - s.astype(int)).mean() > 0.5)
        self.assertEqual(n, 36); self.assertGreater(changed, 30)               # almost every frame carries labels

    def test_annotate_keyframes_and_only_filter(self):
        paths = annotate_keyframes(self.scene, os.path.join(self.d, "kf"), only="mug")
        self.assertEqual(len(paths), 12); self.assertTrue(all(p.exists() for p in paths))


if __name__ == "__main__":
    unittest.main(verbosity=2)
