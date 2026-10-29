"""Build JSON-schema tool definitions (Anthropic Messages API format) from AgentTools signatures."""
from __future__ import annotations
import inspect, typing
from .query import AgentTools

TOOL_NAMES = ("list_objects", "get_object", "find_objects", "where_is", "distance_between", "objects_near",
              "get_relations", "scene_summary", "mark_missing", "update_position", "calibrate_scale")
_MAP = {int: "integer", float: "number", str: "string", bool: "boolean"}


def _schema(tp) -> dict:
    if tp in _MAP: return {"type": _MAP[tp]}
    origin = typing.get_origin(tp)
    if origin in (list, typing.List):
        (inner,) = typing.get_args(tp) or (float,)
        return {"type": "array", "items": _schema(inner)}
    return {"type": "string"}


def tool_definitions() -> list[dict]:
    out = []
    for name in TOOL_NAMES:
        fn = getattr(AgentTools, name)
        hints = typing.get_type_hints(fn)
        props, req = {}, []
        for pname, par in inspect.signature(fn).parameters.items():
            if pname == "self": continue
            props[pname] = _schema(hints.get(pname, str))
            if par.default is inspect._empty: req.append(pname)
            else: props[pname]["default"] = par.default
        out.append({"name": name, "description": (fn.__doc__ or "").strip(),
                    "input_schema": {"type": "object", "properties": props, "required": req}})
    return out
