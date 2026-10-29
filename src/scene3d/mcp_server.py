"""MCP server exposing the scene graph to an AI agent (Claude Desktop, Claude Code, any MCP client).

    set SCENE3D_GRAPH=C:\\scene3d\\data\\<scene_id>\\graph      (Windows cmd)
    scene3d-mcp
Needs `pip install mcp`. Optional text-query embedder: SCENE3D_CLIP=1 (loads CLIP)."""
from __future__ import annotations
import os
from .graph import SceneGraph
from .query import AgentTools
from .tool_schemas import TOOL_NAMES


def build_tools() -> AgentTools:
    path = os.environ.get("SCENE3D_GRAPH", "./data/graph")
    emb = None
    if os.environ.get("SCENE3D_CLIP") == "1":
        from .backends.models import ClipEmbedder
        emb = ClipEmbedder()
    return AgentTools(SceneGraph.load(path), emb, path)


def main():
    from mcp.server.fastmcp import FastMCP
    t = build_tools()
    mcp = FastMCP("scene3d")
    for name in TOOL_NAMES:
        mcp.tool()(getattr(t, name))
    mcp.run()


if __name__ == "__main__":
    main()
