"""Import Depth Anything 3 on machines where some of its *optional* heavy modules cannot be installed.

DA3's package imports its Gaussian-splat exporter at import time, which needs `gsplat` (CUDA build, painful on
Windows CPU). We never use that exporter, so if such a module is missing we register an inert stub for it.
Anything else that is missing raises ModuleNotFoundError (install it with pip)."""
from __future__ import annotations
import importlib, importlib.abc, importlib.machinery, sys, types

STUBBABLE = ("gsplat", "pytorch3d", "nerfview", "viser")


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        full = f"{self.__name__}.{name}"
        def _missing(*a, **k):
            raise RuntimeError(f"{full} is a stub: the real module is not installed (not needed for pose/depth)")
        return _missing


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self): self.roots: set[str] = set()
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.roots:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = _Stub(spec.name); m.__path__ = []; return m
    def exec_module(self, module): pass


_finder = _StubFinder()


def import_da3():
    """Return the DepthAnything3 class, stubbing optional heavy modules when needed."""
    for _ in range(len(STUBBABLE) + 1):
        try:
            from depth_anything_3.api import DepthAnything3
            return DepthAnything3
        except ModuleNotFoundError as e:
            root = (e.name or "").split(".")[0]
            if root in STUBBABLE and root not in _finder.roots:
                _finder.roots.add(root)
                if _finder not in sys.meta_path: sys.meta_path.append(_finder)
                for k in [k for k in sys.modules if k == "depth_anything_3" or k.startswith("depth_anything_3.")]:
                    del sys.modules[k]                  # drop the half-imported package, retry cleanly
                continue
            raise
    raise ImportError("could not import depth_anything_3")
