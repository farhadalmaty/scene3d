"""REST API (FastAPI) — the contract a future phone app talks to.

  POST /api/scenes                     multipart: video, name   -> {scene_id}
  GET  /api/scenes                     list
  GET  /api/scenes/{id}                status / progress / counts
  GET  /api/scenes/{id}/objects        all objects
  GET  /api/scenes/{id}/objects/{oid}  one object + relations
  GET  /api/scenes/{id}/summary
  POST /api/scenes/{id}/find           {query, top_k}
  POST /api/scenes/{id}/calibrate      {id_a, id_b, real_distance_m}
  POST /api/scenes/{id}/chat           {message}  -> {reply}      (needs ANTHROPIC_API_KEY)
  GET  /api/scenes/{id}/plot.png       top/side view for quick inspection
  POST /api/scenes/{id}/annotate?only=mug   render the labelled video (blocking), -> {url}
  GET  /api/scenes/{id}/annotated.mp4  the labelled video
  DELETE /api/scenes/{id}
"""
from __future__ import annotations
import tempfile, uuid
from pathlib import Path
from .core import SceneService


def create_app(service: SceneService):
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel

    app = FastAPI(title="scene3d", version="0.2")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    chats: dict = {}

    class FindBody(BaseModel):
        query: str
        top_k: int = 5

    class CalBody(BaseModel):
        id_a: int
        id_b: int
        real_distance_m: float

    class ChatBody(BaseModel):
        message: str

    def tools(sid):
        try:
            return service.tools(sid)
        except KeyError:
            raise HTTPException(404, f"unknown scene {sid}")
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.get("/api/health")
    def health(): return {"ok": True}

    @app.post("/api/scenes")
    async def upload(video: UploadFile = File(...), name: str = Form("")):
        suffix = Path(video.filename or "v.mp4").suffix or ".mp4"
        tmp = Path(service.store.root) / f"upload_{uuid.uuid4().hex}{suffix}"
        with open(tmp, "wb") as f:
            while chunk := await video.read(1 << 20):
                f.write(chunk)
        return {"scene_id": service.submit_video(tmp, name, copy=False)}

    @app.get("/api/scenes")
    def scenes(): return service.store.list()

    @app.get("/api/scenes/{sid}")
    def scene(sid: str):
        try: return service.store.meta(sid)
        except KeyError: raise HTTPException(404, f"unknown scene {sid}")

    @app.delete("/api/scenes/{sid}")
    def delete(sid: str):
        try: service.store.delete(sid); chats.pop(sid, None); return {"deleted": sid}
        except KeyError: raise HTTPException(404, f"unknown scene {sid}")

    @app.get("/api/scenes/{sid}/objects")
    def objects(sid: str, include_missing: bool = False): return tools(sid).list_objects(include_missing)

    @app.get("/api/scenes/{sid}/objects/{oid}")
    def obj(sid: str, oid: int):
        try: return tools(sid).get_object(oid)
        except KeyError as e: raise HTTPException(404, str(e))

    @app.get("/api/scenes/{sid}/summary")
    def summary(sid: str): return tools(sid).scene_summary()

    @app.post("/api/scenes/{sid}/find")
    def find(sid: str, body: FindBody): return tools(sid).find_objects(body.query, body.top_k)

    @app.post("/api/scenes/{sid}/calibrate")
    def calibrate(sid: str, body: CalBody):
        try:
            r = tools(sid).calibrate_scale(body.id_a, body.id_b, body.real_distance_m)
            service.store.update(sid, metric_scale=True); return r
        except (KeyError, ValueError) as e: raise HTTPException(400, str(e))

    @app.post("/api/scenes/{sid}/chat")
    def chat(sid: str, body: ChatBody):
        from ..agent_chat import ChatAgent
        t = tools(sid)
        if sid not in chats:
            try: chats[sid] = ChatAgent(t)
            except Exception as e: raise HTTPException(503, f"agent unavailable: {e}")
        return {"reply": chats[sid].ask(body.message)}

    @app.get("/api/scenes/{sid}/plot.png")
    def plot(sid: str):
        from ..plot import plot_scene
        t = tools(sid)
        p = Path(tempfile.gettempdir()) / f"scene3d_{sid}.png"
        plot_scene(t.g, p)
        return FileResponse(p, media_type="image/png")

    @app.post("/api/scenes/{sid}/annotate")
    def annotate(sid: str, only: str = ""):
        try: service.annotate(sid, only)
        except KeyError: raise HTTPException(404, f"unknown scene {sid}")
        except (RuntimeError, FileNotFoundError, StopIteration) as e: raise HTTPException(409, str(e))
        return {"url": f"/api/scenes/{sid}/annotated.mp4"}

    @app.get("/api/scenes/{sid}/annotated.mp4")
    def annotated(sid: str):
        try: p = service.store.dir(sid) / "annotated.mp4"
        except KeyError: raise HTTPException(404, f"unknown scene {sid}")
        if not p.exists(): raise HTTPException(404, "not rendered yet: POST /annotate first")
        return FileResponse(p, media_type="video/mp4")

    return app
