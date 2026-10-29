"""Make `import depth_anything_3` work: repeatedly try it, pip-install whichever module is missing, retry.
(DA3's declared dependency list is incomplete, so reading pyproject.toml is not enough.)"""
import re, subprocess, sys

PIP_NAME = {"cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml", "sklearn": "scikit-learn",
            "skimage": "scikit-image", "OpenGL": "PyOpenGL", "attr": "attrs", "dotenv": "python-dotenv"}
NEVER = {"xformers"}          # real xformers is not needed on CPU; do not pip it
PROBE = ("import warnings; warnings.filterwarnings('ignore'); "
         "from scene3d.backends.da3_compat import import_da3; import_da3(); print('DA3 import OK')")

installed = []
for _ in range(60):
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
    if r.returncode == 0:
        print("\n" + r.stdout.strip()); print("installed along the way:", installed or "nothing"); sys.exit(0)
    m = re.search(r"No module named '([^']+)'", r.stderr)
    if not m:
        print("\nImport failed for another reason. Last lines:\n" + "\n".join(r.stderr.strip().splitlines()[-15:])); sys.exit(1)
    mod = m.group(1).split(".")[0]
    pkg = PIP_NAME.get(mod, mod)
    if mod in NEVER or pkg in installed:
        print(f"\nStuck on module '{mod}'. Last lines:\n" + "\n".join(r.stderr.strip().splitlines()[-15:])); sys.exit(1)
    print(f"missing '{mod}' -> pip install {pkg}", flush=True)
    if subprocess.call([sys.executable, "-m", "pip", "install", pkg]) != 0:
        print(f"pip could not install '{pkg}'. Send me this line."); sys.exit(1)
    installed.append(pkg)
print("too many iterations"); sys.exit(1)
