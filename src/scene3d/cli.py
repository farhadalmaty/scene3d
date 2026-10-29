from __future__ import annotations
import argparse, json, os, sys


from .presets import CAMERA_FOV


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")           # Windows consoles default to cp866/cp1251
        except Exception: pass


def main(argv=None):
    _utf8()
    ap = argparse.ArgumentParser("scene3d", description="3D object map from video + AI agent")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("process", help="video -> object map (runs the neural models)")
    p.add_argument("video"); p.add_argument("-o", "--out", default="data/scene1")
    p.add_argument("--frames", type=int, default=24, help="keyframes for reconstruction (CPU: 16-32)")
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--geometry-model", default="depth-anything/da3-small", help="e.g. depth-anything/da3-base")
    p.add_argument("--vocab", default="", help="comma-separated tag vocabulary (default: built-in 30 words)")
    p.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    p.add_argument("--plot", action="store_true", help="also write plan.png")
    p.add_argument("--min-obs", type=int, default=None, help="keep objects seen in at least N frames (default 3)")
    p.add_argument("--delta-nn", type=float, default=None, help="association distance floor (default 0.03)")
    p.add_argument("--delta-sim", type=float, default=None, help="association threshold (default 1.1)")
    p.add_argument("--max-objects-plot", type=int, default=40)
    p.add_argument("--fov-long", type=float, default=None, help="camera field of view in degrees along the long side of the frame (default: model guesses)")
    p.add_argument("--camera", choices=sorted(CAMERA_FOV), default=None, help="phone camera preset: main (1x), ultrawide (0.5x), tele (2x)")
    p.add_argument("--min-tag-prob", type=float, default=None, help="withhold names with lower confidence (default 0.35)")
    p.add_argument("--crop-mode", choices=["masked", "context", "both"], default=None, help="what the tagger sees (default masked)")
    p.add_argument("--hi-side", type=int, default=960, help="long side of frames used for segmentation/recognition; 0 = depth resolution")
    p.add_argument("--da3-res", type=int, default=None, help="long side for the depth model (default 504; larger = slower, more detail)")
    p.add_argument("--no-compress", action="store_true", help="do not shrink big videos before processing")
    p.add_argument("--compress-side", type=int, default=960, help="long side after compression (default 960)")
    p.add_argument("--compress-crf", type=int, default=23, help="H.264 quality of the working copy (default 23; higher = smaller, blurrier)")
    p.add_argument("--depth-conf-cut", type=float, default=None, help="%% of least-confident depth pixels to drop per frame (default 20)")
    p.add_argument("--compress-fps", type=float, default=15.0, help="frame rate after compression (default 15)")
    p.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270],
                   help="force clockwise rotation of the frames (default: read the rotation flag of the file)")

    s = sp.add_parser("serve", help="start the REST API (for the phone app / web client)")
    s.add_argument("--data", default="data"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)

    c = sp.add_parser("chat", help="talk to the AI agent about a processed scene (needs ANTHROPIC_API_KEY)")
    c.add_argument("graph", help="path to <scene>/graph")

    a = sp.add_parser("ask", help="plain (no-LLM) lookup in a graph")
    a.add_argument("graph"); a.add_argument("query")
    pl = sp.add_parser("plot", help="draw top/side view of a graph")
    pl.add_argument("graph"); pl.add_argument("-o", "--out", default="plan.png")
    an = sp.add_parser("annotate", help="draw object labels onto the video (needs a scene made by `process`)")
    an.add_argument("scene", help="scene folder, e.g. data\\scene1")
    an.add_argument("--video", default=None, help="source video (default: the one used by `process`)")
    an.add_argument("-o", "--out", default=None, help="output .mp4 (default: <scene>/annotated.mp4)")
    an.add_argument("--only", default="", help="label only these tags, e.g. mug,book")
    an.add_argument("--keyframes", action="store_true", help="label only the keyframes (exact poses) and save PNGs")
    an.add_argument("--depth", action="store_true", help="append distance from the camera to each label")
    an.add_argument("--show-unknown", action="store_true", help="also label objects whose name is unsure")
    an.add_argument("--no-track", action="store_true", help="skip per-frame camera tracking, interpolate between keyframes (fast, approximate)")
    ins = sp.add_parser("inspect", help="find out why objects have wrong names/places: ranking.csv + contact sheets")
    ins.add_argument("scene"); ins.add_argument("--ids", default="", help="object ids to always draw, e.g. 5,112,140")
    ins.add_argument("--worst", type=int, default=30, help="draw sheets for the N most suspicious objects (default 30)")
    cf = sp.add_parser("calibrate-fov", help="find the camera field of view that fits the footage (e.g. for 0.5x ultra-wide)")
    cf.add_argument("video"); cf.add_argument("--fovs", default="auto", help="comma-separated candidates in degrees (long side), or auto")
    cf.add_argument("--frames", type=int, default=12, help="keyframes per trial (default 12: faster)")
    cf.add_argument("--device", default=None); cf.add_argument("--geometry-model", default="depth-anything/da3-small")
    rt = sp.add_parser("retag", help="rename objects from the saved crops without re-running the depth model")
    rt.add_argument("scene"); rt.add_argument("--min-tag-prob", type=float, default=0.35)
    rt.add_argument("--mode", choices=["masked", "context", "both"], default="masked")
    rt.add_argument("--compare", action="store_true", help="try all three modes and only print statistics")
    rt.add_argument("--vocab", default="", help="comma-separated vocabulary (default: built-in)")
    cp = sp.add_parser("compress", help="shrink a video (downscale + lower fps + H.264) before processing or sharing")
    cp.add_argument("video"); cp.add_argument("-o", "--out", default=None, help="output .mp4 (default: <name>_small.mp4 next to the input)")
    cp.add_argument("--side", type=int, default=960, help="long side in px (default 960)")
    cp.add_argument("--fps", type=float, default=15.0)
    cp.add_argument("--crf", type=int, default=28, help="H.264 quality, higher = smaller (ffmpeg only; default 28)")
    cp.add_argument("--seconds", type=float, default=None, help="keep only the first N seconds")
    cp.add_argument("--engine", choices=["auto", "ffmpeg", "opencv"], default="auto")
    d = sp.add_parser("demo", help="full pipeline on a rendered synthetic scene, no models needed")
    d.add_argument("-o", "--out", default="data/demo")
    fr = sp.add_parser("frames", help="dump every N-th frame of a video as PNG")
    fr.add_argument("video"); fr.add_argument("out"); fr.add_argument("--every", type=int, default=15)
    args = ap.parse_args(argv)

    if args.cmd == "process":
        from .service.processing import ProcessConfig, process_video
        from .backends.models import default_backends
        vocab = [w.strip() for w in args.vocab.split(",") if w.strip()] or None
        print("loading models (first run downloads weights)...", flush=True)
        b = default_backends(args.device, vocab, args.geometry_model)
        def cb(stage, frac, msg=""): print(f"[{stage:9s}] {frac * 100:5.1f}%  {msg}", flush=True)
        from .pipeline import PipelineConfig
        pc = PipelineConfig()
        if args.min_obs is not None: pc.min_obs = args.min_obs
        if args.min_tag_prob is not None: pc.min_tag_prob = args.min_tag_prob
        if args.crop_mode: pc.crop_mode = args.crop_mode
        if args.delta_nn is not None: pc.delta_nn = args.delta_nn
        if args.delta_sim is not None: pc.delta_sim = args.delta_sim
        g = process_video(args.video, args.out, b, ProcessConfig(n_frames=args.frames, max_side=args.max_side, pipeline=pc, rotate=args.rotate,
                                              compress=not args.no_compress, compress_max_side=args.compress_side,
                                              hi_side=args.hi_side or None, da3_res=args.da3_res,
                                              fov_long=args.fov_long or (CAMERA_FOV[args.camera] if args.camera else None),
                                              compress_crf=args.compress_crf, depth_conf_cut=args.depth_conf_cut,
                                              compress_fps=args.compress_fps), cb)
        info = json.load(open(os.path.join(args.out, "process_info.json"), encoding="utf-8"))
        d = info["diagnostics"]
        print("\n--- diagnostics ---")
        print(f"masks lifted to 3D: {d['masks']}  (dropped: too big {d['too_big']}, too few points {d['too_few_points']})")
        print(f"association: merged into existing {d['merged_into_existing']}, new {d['new_objects']}, no spatial overlap with any object {d['no_overlap']}")
        print(f"median best geo {d['median_best_geo']}, median best sem {d['median_best_sem']}, share with geo>0.15: {d['frac_geo_above_0.15']}")
        print(f"objects: candidates {d['candidates']} -> after merge {d['after_merge']} -> seen>=min_obs {d['after_min_obs']} -> size filter {d['after_size_filter']} -> final {d['final']}")
        print(f"names: {d['shown']} shown, {d['unknown_name']} withheld as unsure (tag confidence < threshold), {d['dropped_structure']} walls/floors dropped")
        print(f"reliability: {d['unreliable']} objects hidden because their 3D cloud does not match where they were seen (see `scene3d inspect`)")
        print(f"\n{d['shown']} objects shown, saved to {os.path.join(args.out, 'graph')}")
        if args.plot:
            from .plot import plot_scene
            plot_scene(g, os.path.join(args.out, "plan.png"), max_objects=args.max_objects_plot); print("plan.png written")
    elif args.cmd == "serve":
        import uvicorn
        from .service.api import create_app
        from .service.core import SceneService
        uvicorn.run(create_app(SceneService(args.data)), host=args.host, port=args.port)
    elif args.cmd == "chat":
        from .graph import SceneGraph
        from .query import AgentTools
        from .agent_chat import ChatAgent
        agent = ChatAgent(AgentTools(SceneGraph.load(args.graph), None, args.graph))
        print("Agent ready. Empty line to exit.")
        while (q := input("> ").strip()):
            print(agent.ask(q))
    elif args.cmd == "ask":
        from .graph import SceneGraph
        from .query import AgentTools
        print(json.dumps(AgentTools(SceneGraph.load(args.graph)).where_is(args.query), indent=2, ensure_ascii=False))
    elif args.cmd == "plot":
        from .graph import SceneGraph
        from .plot import plot_scene
        plot_scene(SceneGraph.load(args.graph), args.out); print("saved", args.out)
    elif args.cmd == "demo":
        from .pipeline import Pipeline
        from .query import AgentTools
        from .backends.synthetic import SyntheticWorld, SyntheticSegmenter, SyntheticEmbedder, SyntheticCaptioner
        e = SyntheticEmbedder()
        g = Pipeline(SyntheticSegmenter(), e, SyntheticCaptioner(e)).run(SyntheticWorld().orbit(12))
        g.save(os.path.join(args.out, "graph"))
        print(json.dumps(AgentTools(g, e).list_objects(), indent=2, ensure_ascii=False))
    elif args.cmd == "annotate":
        from .annotate import annotate_video, annotate_keyframes
        if args.keyframes:
            ps = annotate_keyframes(args.scene, os.path.join(args.scene, "annotated"), args.only, args.depth, args.show_unknown)
            print(f"{len(ps)} labelled keyframes in {os.path.join(args.scene, 'annotated')}")
        else:
            video = args.video
            if video is None:
                video = json.load(open(os.path.join(args.scene, "process_info.json"), encoding="utf-8")).get("video")
            if not video: sys.exit("source video unknown: pass --video PATH")
            out = args.out or os.path.join(args.scene, "annotated.mp4")
            def cb(frac, msg=""): print(f"\r[annotate] {frac * 100:5.1f}%  {msg}", end="", flush=True)
            r = annotate_video(video, args.scene, out, args.only, cb, args.depth, track=not args.no_track, show_unknown=args.show_unknown)
            print(f"\n{r['frames']} frames, {r['labels_drawn']} labels -> {r['out']}")
            t = r.get('tracking') or {}
            if r['pose_mode'] == 'tracked':
                print(f"camera pose per frame: {t['keyframes']} keyframes, {t['tracked']} tracked from the image, {t['interpolated']} interpolated, {t['rejected']} rejected")
            else:
                print(f"camera pose per frame: interpolated between keyframes ({t.get('fallback_reason', 'tracking disabled')})")
            print(f"video rotation applied: {r['rotation']} deg")
    elif args.cmd == "inspect":
        from .diagnose import inspect_scene
        ids = [int(x) for x in args.ids.replace(" ", "").split(",") if x]
        rows = inspect_scene(args.scene, None, ids, args.worst)
        print(f"{'id':>4} {'tag':16s} {'tag_p':>5} {'obs':>3} {'IoU':>5} {'cover':>5} {'size':>5}  verdict")
        for r in rows[:25]:
            tp = '-' if r['tag_prob'] is None else f"{r['tag_prob']:.2f}"
            print(f"{r['id']:>4} {r['tag'][:16]:16s} {tp:>5} {r['n_obs']:>3} {r['median_iou']:5.2f} {r['median_cover']:5.2f} {str(r['size_ratio']):>5}  {r['verdict']}")
        print(f"\n{len(rows)} objects. Full table: {os.path.join(args.scene, 'inspect', 'ranking.csv')}; sheets: {os.path.join(args.scene, 'inspect')}")
    elif args.cmd == "calibrate-fov":
        from .calib import sweep_fov
        from .video import sample_keyframes
        from .backends.geometry_da3 import DepthAnything3Geometry
        from .backends.models import pick_device
        fovs = [None] + ([60.0, 70.0, 80.0, 90.0, 100.0, 110.0] if args.fovs == "auto" else [float(x) for x in args.fovs.split(",")])
        rgbs = sample_keyframes(args.video, args.frames, 640)
        geo = DepthAnything3Geometry(args.geometry_model, device=args.device or pick_device())
        print(f"{len(fovs)} trials x {len(rgbs)} keyframes (each trial runs the depth model once)...", flush=True)
        rows = sweep_fov(rgbs, geo, fovs, lambda fr, m: print(f"  {fr * 100:3.0f}%  {m}", flush=True))
        rows.sort(key=lambda r: (r["median_px"] != r["median_px"], r["median_px"]))
        print(f"\n{'given FOV':>10} {'error px':>9} {'model fy/fx':>11} {'model FOV':>10}")
        for r in rows:
            print(f"{'(model guess)' if r['fov_long'] is None else str(r['fov_long']) + ' deg':>10} {r['median_px']:9.2f} {r['fy_over_fx']:11.2f} {r['model_fov_long']:9.0f}")
        best = next((r for r in rows if r["fov_long"] is not None), None)
        if best: print(f"\nbest given value: --fov-long {best['fov_long']:g}   (lower error = more consistent with the images)")
    elif args.cmd == "retag":
        from .retag import retag_scene
        from .backends.models import ClipEmbedder, VocabCaptioner, COMMON_VOCAB
        vocab = [w.strip() for w in args.vocab.split(",") if w.strip()] or COMMON_VOCAB
        emb = ClipEmbedder(); cap = VocabCaptioner(emb, vocab)
        for mode in (["masked", "context", "both"] if args.compare else [args.mode]):
            r = retag_scene(args.scene, cap, args.min_tag_prob, mode, write=not args.compare)
            print(f"[{mode:7s}] objects {r['objects']}, unsure {r['unknown']}, median confidence {r['conf_median']}, "
                  f"confidence>=0.5: {r['conf_ge_0.5']}, structure {r['structure']}, names changed {r['changed']}")
            if not args.compare: print("tags:", r["tags"])
    elif args.cmd == "compress":
        from .compress import compress_video, find_ffmpeg
        out = args.out or os.path.splitext(args.video)[0] + "_small.mp4"
        if args.engine == "auto" and not find_ffmpeg():
            print("note: ffmpeg not found, using the OpenCV fallback (bigger files). Better: pip install imageio-ffmpeg")
        r = compress_video(args.video, out, args.side, args.fps, args.crf, args.seconds, args.engine)
        print(f"{r['in_mb']} MB -> {r['out_mb']} MB (x{r['ratio']} smaller, engine: {r['engine']}) -> {out}")
    elif args.cmd == "frames":
        from .video import extract_keyframes
        print(f"saved {len(extract_keyframes(args.video, args.out, args.every))} frames")


if __name__ == "__main__":
    main()
