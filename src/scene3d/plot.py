"""Quick visual check of a scene graph (no 3D viewer needed): top view + side view with labelled boxes."""
from __future__ import annotations
import numpy as np
from .graph import SceneGraph


def plot_scene(graph: SceneGraph, out_path, max_pts_per_obj: int = 1500, max_objects: int = 40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    allobjs = graph.shown()
    objs = sorted(allobjs, key=lambda o: -o.n_obs)[:max_objects]      # most-observed first: the most trustworthy
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=130)
    views = [("Вид сверху (x, y)", 0, 1), ("Вид сбоку (x, z), z вверх", 0, 2)]
    cmap = plt.get_cmap("tab10")
    for ax, (title, i, j) in zip(axes, views):
        for k, o in enumerate(objs):
            c = cmap(k % 10)
            P = o.points[:: max(1, len(o.points) // max_pts_per_obj)]
            ax.scatter(P[:, i], P[:, j], s=1.5, color=c, alpha=.5)
            lo, hi = o.bbox_min, o.bbox_max
            ax.add_patch(Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j], fill=False, ec=c, lw=1.6))
            ax.annotate(f"#{o.id} {o.tag}", (lo[i], hi[j]), fontsize=8.5, color=c, va="bottom")
        ax.set_title(title); ax.set_aspect("equal"); ax.grid(alpha=.25)
        ax.set_xlabel("x"); ax.set_ylabel("y" if j == 1 else "z")
    unit = "м" if graph.metric_scale else "условные единицы"
    shown = f"показаны {len(objs)} самых надёжных из {len(allobjs)}" if len(allobjs) > len(objs) else f"{len(objs)}"
    fig.suptitle(f"Объектов: {shown}  |  единицы: {unit}")
    fig.tight_layout(); fig.savefig(str(out_path)); plt.close(fig)
