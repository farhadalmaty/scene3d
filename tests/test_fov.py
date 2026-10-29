import os, sys, copy, tempfile, types, unittest, warnings, json
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scene3d.backends.geometry_da3 import DepthAnything3Geometry, intrinsics_from_fov
from scene3d.calib import epipolar_consistency, sweep_fov, sampson
from scene3d.backends.synthetic import SyntheticWorld, SyntheticSegmenter, SyntheticEmbedder, SyntheticCaptioner, SyntheticGeometry
from scene3d.backends.models import Backends
from scene3d.service.processing import process_video, ProcessConfig
from scene3d.cli import CAMERA_FOV
from scene3d.service.core import config_from_request
from test_service import make_video


class FakeDA3:
    """Stand-in for the model. mode: 'scales' (library rescales given K to the processed size), 'passthrough' (it takes K
    as already being at the processed size), 'ignores' (returns its own guess)."""
    def __init__(self, mode, proc=(28, 18)):
        self.mode, self.calls, self.proc = mode, [], proc
    def inference(self, imgs, process_res=None, intrinsics=None):
        n = len(imgs); hp, wp = self.proc; h, w = imgs[0].shape[:2]
        self.calls.append({"process_res": process_res, "K": None if intrinsics is None else np.array(intrinsics)})
        if intrinsics is None or self.mode == "ignores":
            K = np.tile(np.array([[50.0, 0, wp / 2], [0, 60.0, hp / 2], [0, 0, 1]]), (n, 1, 1))     # its own (anisotropic) guess
        elif self.mode == "scales":
            K = np.array(intrinsics).copy(); K[:, 0] *= wp / w; K[:, 1] *= hp / h
        else:
            K = np.array(intrinsics).copy()
        return types.SimpleNamespace(depth=np.ones((n, hp, wp), np.float32), conf=np.ones((n, hp, wp)),
                                     extrinsics=np.tile(np.eye(4)[:3], (n, 1, 1)), intrinsics=K)


def geo(mode):
    g = DepthAnything3Geometry.__new__(DepthAnything3Geometry)
    g.model = FakeDA3(mode); g.conf_percentile = 20.0; g.process_res = 504; g.fov_long = None; g.known_K = None; g.notes = []; g._scale_input_k = True
    return g


IMGS = [np.zeros((64, 36, 3), np.uint8)] * 3            # portrait 36x64 input


class TestGivenIntrinsics(unittest.TestCase):
    def test_focal_from_fov(self):
        K = intrinsics_from_fov(100.0, 64, 36, 2)
        self.assertEqual(K.shape, (2, 3, 3)); self.assertAlmostEqual(K[0, 0, 0], 32 / np.tan(np.radians(50)), places=6)
        self.assertEqual(K[0, 0, 0], K[0, 1, 1]); self.assertEqual((K[0, 0, 2], K[0, 1, 2]), (18, 32))

    def test_no_fov_means_model_guess(self):
        g = geo("scales"); g.estimate(IMGS); self.assertIsNone(g.model.calls[0]["K"])

    def test_library_that_rescales_is_used_once(self):
        g = geo("scales"); g.fov_long = 100.0; fr = g.estimate(IMGS)
        self.assertEqual(len(g.model.calls), 1); self.assertEqual(g.notes, [])
        self.assertAlmostEqual(fr[0].K[0, 0] / fr[0].K[1, 1], (18 / 36) / (28 / 64), places=3)      # K at processed size

    def test_library_that_does_not_rescale_triggers_one_corrected_rerun(self):
        g = geo("passthrough"); g.fov_long = 100.0; fr = g.estimate(IMGS)
        self.assertEqual(len(g.model.calls), 2); self.assertIn("did not rescale", g.notes[0])
        self.assertAlmostEqual(g.model.calls[1]["K"][0, 0, 0], g.model.calls[0]["K"][0, 0, 0] * 18 / 36, places=6)

    def test_ignored_intrinsics_are_reported(self):
        g = geo("ignores"); g.fov_long = 100.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always"); g.estimate(IMGS)
        self.assertTrue(any("ignoring" in str(x.message) for x in w)); self.assertEqual(len(g.model.calls), 1)


class TestFovSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = list(SyntheticWorld(texture=0.3).orbit(24))

    def _scaled(self, s, aspect=1.0):
        fr = [copy.copy(f) for f in self.frames]
        for f in fr: f.K = f.K.copy(); f.K[0, 0] *= s; f.K[1, 1] *= s * aspect
        return fr

    def test_true_intrinsics_score_best(self):
        e_true = epipolar_consistency(self.frames)["median_px"]
        for s in (0.6, 0.8, 1.5, 2.0): self.assertGreater(epipolar_consistency(self._scaled(s))["median_px"], e_true, s)
        self.assertGreater(epipolar_consistency(self._scaled(1.0, 1.21))["median_px"], 3 * e_true)        # the fy/fx=1.21 case

    def test_sampson_is_zero_for_a_perfect_match(self):
        F = np.array([[0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]); p1 = np.array([[10., 20.], [30., 5.]]); p2 = np.array([[7., 20.], [9., 5.]])
        np.testing.assert_allclose(sampson(F, p1, p2), 0, atol=1e-9)

    def test_sweep_picks_the_right_field_of_view(self):
        true_f = self.frames[0].K[0, 0]; h, w = self.frames[0].depth.shape
        class Fake:                         # geometry whose K follows the fov it is given (like DA3 with intrinsics)
            fov_long = None
            def estimate(s, rgbs):
                fov = s.fov_long
                f = true_f if fov is None else (max(h, w) / 2) / np.tan(np.radians(fov) / 2)
                out = [copy.copy(x) for x in self_.frames]
                for x in out: x.K = x.K.copy(); x.K[0, 0] = x.K[1, 1] = f * (1.0 if fov is not None else 1.0)
                return out
        self_ = self
        true_fov = 2 * np.degrees(np.arctan(max(h, w) / 2 / true_f))
        rows = sweep_fov([None], Fake(), [None, true_fov * 0.7, true_fov, true_fov * 1.4])
        best = min((r for r in rows if r["fov_long"] is not None), key=lambda r: r["median_px"])
        self.assertAlmostEqual(best["fov_long"], true_fov, places=3)
        self.assertEqual(len(rows), 4); self.assertIn("fy_over_fx", rows[0])


class TestProcessingReportsCameraModel(unittest.TestCase):
    def test_info_records_fov_and_warns_about_anisotropy(self):
        d = tempfile.mkdtemp(); vp, gt, b = make_video(d, 36)
        msgs = []
        class Aniso(SyntheticGeometry):
            def estimate(s, rgbs):
                fr = super().estimate(rgbs)
                for f in fr: f.K = f.K.copy(); f.K[1, 1] *= 1.21
                return fr
        b2 = Backends(Aniso(gt), b.segmenter, b.embedder, b.captioner)
        process_video(vp, os.path.join(d, "o"), b2, ProcessConfig(n_frames=12, fov_long=100.0), lambda s, f, m="": msgs.append(m))
        info = json.load(open(os.path.join(d, "o", "process_info.json"), encoding="utf-8"))
        self.assertEqual(info["fov_long_given"], 100.0); self.assertAlmostEqual(info["model_fy_over_fx"], 1.21, places=2)
        self.assertTrue(any("fy/fx" in m for m in msgs)); self.assertIn("model_fov_long_deg", info)

    def test_camera_presets(self):
        self.assertEqual(sorted(CAMERA_FOV), ["main", "tele", "ultrawide"]); self.assertGreater(CAMERA_FOV["ultrawide"], CAMERA_FOV["main"])


class TestCameraInfoFromTheApp(unittest.TestCase):
    def test_exact_intrinsics_are_scaled_to_the_frame_size_and_used(self):
        g = geo("scales"); g.known_K = {"fx": 1000.0, "fy": 1010.0, "cx": 540.0, "cy": 960.0, "width": 1080, "height": 1920}
        g.estimate(IMGS)                                         # frames are 36x64: scale 1/30
        K = g.model.calls[0]["K"][0]
        self.assertAlmostEqual(K[0, 0], 1000 * 36 / 1080, places=6); self.assertAlmostEqual(K[1, 1], 1010 * 64 / 1920, places=6)
        self.assertAlmostEqual(K[0, 2], 18.0, places=6); self.assertAlmostEqual(K[1, 2], 32.0, places=6)

    def test_exact_intrinsics_win_over_fov(self):
        g = geo("scales"); g.fov_long = 100.0; g.known_K = {"fx": 50.0, "fy": 50.0, "cx": 18.0, "cy": 32.0, "width": 36, "height": 64}
        g.estimate(IMGS); self.assertAlmostEqual(g.model.calls[0]["K"][0, 0, 0], 50.0)

    def test_request_to_config_priority_and_validation(self):
        base = ProcessConfig()
        self.assertIs(config_from_request(base), base)
        self.assertEqual(config_from_request(base, camera="ultrawide").fov_long, CAMERA_FOV["ultrawide"])
        self.assertEqual(config_from_request(base, camera="ultrawide", fov_long=95.0).fov_long, 95.0)
        j = json.dumps({"fx": 1, "fy": 1, "cx": 1, "cy": 1, "width": 2, "height": 2})
        c = config_from_request(base, camera="main", fov_long=95.0, intrinsics_json=j)
        self.assertEqual(c.camera_intrinsics["width"], 2.0); self.assertIsNone(c.fov_long)
        with self.assertRaises(ValueError): config_from_request(base, camera="fisheye")
        with self.assertRaises(ValueError): config_from_request(base, intrinsics_json=json.dumps({"fx": 1}))
        self.assertIsNone(base.camera_intrinsics)                 # the shared base config is never mutated


if __name__ == "__main__":
    unittest.main(verbosity=2)
