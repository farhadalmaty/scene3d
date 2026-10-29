import os, sys, struct, tempfile, unittest
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scene3d.videometa import mp4_rotation
from scene3d.imageio_safe import open_video
from scene3d.backends.synthetic import SyntheticWorld, SyntheticSegmenter, SyntheticEmbedder, SyntheticCaptioner, SyntheticGeometry
from scene3d.backends.models import Backends
from scene3d.service.processing import process_video, ProcessConfig
from scene3d.annotate import load_geometry, interp_pose, annotate_video
from scene3d.track import PoseTracker, track_video_poses
from scene3d.imageio_safe import imread_rgb


def patch_rotation(path, deg):
    """Rewrite the tkhd matrix of an MP4 written by OpenCV so it carries a display-rotation flag."""
    buf = bytearray(open(path, "rb").read())
    i = buf.find(b"tkhd"); ver = buf[i + 4]
    off = i + 4 + 4 + (20 if ver == 0 else 32) + 16
    a, b = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}[deg]
    one = 0x10000
    vals = [a * one, b * one, 0, -b * one, a * one, 0]
    for k, v in zip((0, 4, 8, 12, 16, 20), vals):
        buf[off + k:off + k + 4] = struct.pack(">i", v)
    open(path, "wb").write(bytes(buf))


def write_mp4(path, n=6, size=(320, 240)):                  # asymmetric content: a bright block in the top-left corner
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for k in range(n):
        f = np.full((size[1], size[0], 3), 60, np.uint8); f[:60, :100] = (255, 255, 255); vw.write(f)
    vw.release()


class TestRotation(unittest.TestCase):
    def test_parse_all_angles_and_non_mp4(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, "v.mp4"); write_mp4(p)
        self.assertEqual(mp4_rotation(p), 0)
        for deg in (90, 180, 270):
            write_mp4(p); patch_rotation(p, deg); self.assertEqual(mp4_rotation(p), deg)
        avi = os.path.join(d, "v.avi"); vw = cv2.VideoWriter(avi, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48)); vw.write(np.zeros((48, 64, 3), np.uint8)); vw.release()
        self.assertEqual(mp4_rotation(avi), 0); self.assertEqual(mp4_rotation(os.path.join(d, "missing.mp4")), 0)

    def test_frames_are_rotated_exactly_once_whatever_the_opencv_build_does(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, "v.mp4"); write_mp4(p, size=(320, 240)); patch_rotation(p, 90)
        cap = open_video(p); ok, fr = cap.read(); cap.release()
        self.assertEqual(fr.shape[:2], (320, 240))                            # portrait now: H=320, W=240
        # block that was top-left in the stored frame must be TOP-RIGHT after a clockwise 90-degree rotation
        self.assertGreater(fr[:100, -60:].mean(), 200); self.assertLess(fr[:100, :60].mean(), 100)
        cap = open_video(p, rotate=0); ok, fr0 = cap.read(); cap.release()     # manual override
        self.assertEqual(fr0.shape[:2], (240, 320))

    def test_unrotated_file_untouched(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, "v.mp4"); write_mp4(p)
        cap = open_video(p); ok, fr = cap.read(); cap.release()
        self.assertEqual(fr.shape[:2], (240, 320)); self.assertGreater(fr[:60, :100].mean(), 200)


def textured_video(N=48, tex=0.3):
    w = SyntheticWorld(texture=tex); t = np.linspace(0, 1, N); ang = 2 * np.pi * (t ** 1.6); gt = []   # uneven camera speed
    for i, a in enumerate(ang):
        pos = (0.3 + 2.0 * np.cos(a), 0.1 + 2.0 * np.sin(a), 1.2 + 0.25 * np.sin(3 * a))
        gt.append(w.render(w.look_at(pos, (0.3, 0.1, 0.5)), i))
    d = tempfile.mkdtemp(); vp = os.path.join(d, "v.avi")
    vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for f in gt: vw.write(cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR))
    vw.release()
    e = SyntheticEmbedder(tol=100)
    return d, vp, gt, Backends(SyntheticGeometry(gt), SyntheticSegmenter(tol=100), e, SyntheticCaptioner(e))


def umeyama_align(gt, geo):
    ki = geo["frame_index"].astype(int); A = np.stack([gt[k].pose_wc[:3, 3] for k in ki]); B = geo["poses_wc"][:, :3, 3]
    ma, mb = A.mean(0), B.mean(0); U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma) / len(A))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt
    sc = np.trace(np.diag(S) @ D) / ((A - ma) ** 2).sum() * len(A); t = mb - sc * R @ ma
    out = []
    for f in gt:
        T = np.eye(4); T[:3, :3] = R @ f.pose_wc[:3, :3]; T[:3, 3] = sc * R @ f.pose_wc[:3, 3] + t; out.append(T)
    return np.stack(out)


class TestTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d, cls.vp, cls.gt, b = textured_video()
        cls.sc = os.path.join(cls.d, "s"); process_video(cls.vp, cls.sc, b, ProcessConfig(n_frames=8))
        cls.geo = load_geometry(cls.sc); cls.gta = umeyama_align(cls.gt, cls.geo)

    def _frames(self, tr):
        cap = cv2.VideoCapture(self.vp)
        while True:
            ok, bgr = cap.read()
            if not ok: return
            yield cv2.cvtColor(cv2.resize(bgr, (tr.hw[1], tr.hw[0])), cv2.COLOR_BGR2RGB)

    def test_tracking_beats_interpolation_on_uneven_motion(self):
        kf = [imread_rgb(os.path.join(self.sc, "frames", f"{i:05d}.png")) for i in range(8)]
        tr = PoseTracker(self.geo, kf)
        poses, info = track_video_poses(self._frames(tr), 48, tr)
        self.assertGreaterEqual(info["tracked"], 6, info)
        kidx = self.geo["frame_index"].astype(float); keys = set(kidx.astype(int))
        et, ei = [], []
        for i in range(48):
            if i in keys: continue
            et.append(np.linalg.norm(poses[i][:3, 3] - self.gta[i][:3, 3]))
            ei.append(np.linalg.norm(interp_pose(self.geo["poses_wc"], kidx, i)[:3, 3] - self.gta[i][:3, 3]))
        # frames localised from the IMAGE are accurate to a few cm, several times better than interpolation on the same frames
        m = np.array(info["tracked_mask"]); idx = np.nonzero(m)[0]
        e_tr = np.array([np.linalg.norm(poses[i][:3, 3] - self.gta[i][:3, 3]) for i in idx])
        e_ip = np.array([np.linalg.norm(interp_pose(self.geo["poses_wc"], kidx, i)[:3, 3] - self.gta[i][:3, 3]) for i in idx])
        self.assertLess(np.median(e_tr), 0.04, e_tr)
        self.assertLess(np.median(e_tr) * 2.5, np.median(e_ip), (np.median(e_tr), np.median(e_ip)))
        # overall (gaps filled by interpolation between tracked frames) is never worse than plain interpolation
        self.assertLessEqual(np.median(et), np.median(ei) + 1e-6)

    def test_annotate_video_uses_tracking_and_reports_it(self):
        r = annotate_video(self.vp, self.sc, os.path.join(self.d, "a.mp4"))
        self.assertEqual(r["pose_mode"], "tracked"); self.assertGreaterEqual(r["tracking"]["tracked"], 6)
        r2 = annotate_video(self.vp, self.sc, os.path.join(self.d, "b.mp4"), track=False)
        self.assertEqual(r2["pose_mode"], "interpolated")

    def test_old_scene_without_depth_falls_back_to_interpolation(self):
        sc2 = os.path.join(self.d, "old"); import shutil; shutil.copytree(self.sc, sc2)
        g = dict(np.load(os.path.join(sc2, "geometry.npz"))); g.pop("depth"); np.savez(os.path.join(sc2, "geometry.npz"), **g)
        r = annotate_video(self.vp, sc2, os.path.join(self.d, "c.mp4"))
        self.assertEqual(r["pose_mode"], "interpolated"); self.assertIn("fallback_reason", r["tracking"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
