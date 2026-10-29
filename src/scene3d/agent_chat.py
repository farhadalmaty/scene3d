"""In-app chat agent: Claude (Anthropic Messages API) calling AgentTools in a tool-use loop.
Needs `pip install anthropic` and ANTHROPIC_API_KEY. The client is injectable (tests use a fake)."""
from __future__ import annotations
import json, os
from .query import AgentTools
from .tool_schemas import tool_definitions

DEFAULT_MODEL = os.environ.get("SCENE3D_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You help a user find and reason about objects detected in a video of a room.
You have tools over a 3D object map (positions, sizes, relations). Rules:
- At the start of a conversation call scene_summary to see what exists and whether the scale is calibrated.
- Object tags and captions are English. Translate the user's description to English before calling
  find_objects / where_is, but answer in the user's language.
- Every tool result has a `units` field. If it says arbitrary units, do NOT state distances in metres;
  report them as map units and offer calibration (ask for one real distance, then call calibrate_scale).
- Coordinates: z is up. Say 'on the table' etc. using get_relations rather than guessing.
- If several objects match about equally, list the candidates instead of picking one silently.
- mark_missing / update_position / calibrate_scale change the saved map: only call them after the user
  has asked for it or confirmed."""


class ChatAgent:
    def __init__(self, tools: AgentTools, client=None, model: str | None = None, system: str = SYSTEM_PROMPT,
                 max_steps: int = 8, max_tokens: int = 1024):
        self.tools, self.model, self.system = tools, model or DEFAULT_MODEL, system
        self.max_steps, self.max_tokens = max_steps, max_tokens
        self.defs = tool_definitions(); self.messages: list[dict] = []
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client

    def _run_tool(self, name: str, args: dict) -> tuple[str, bool]:
        try:
            return json.dumps(getattr(self.tools, name)(**args), ensure_ascii=False), False
        except Exception as e:
            return f"{type(e).__name__}: {e}", True

    def ask(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        for _ in range(self.max_steps):
            resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=self.system,
                                               tools=self.defs, messages=self.messages)
            self.messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            results = []
            for b in resp.content:
                if getattr(b, "type", "") == "tool_use":
                    out, err = self._run_tool(b.name, dict(b.input))
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
            self.messages.append({"role": "user", "content": results})
        return "(stopped: too many tool steps)"
