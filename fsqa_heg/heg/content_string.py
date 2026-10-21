"""Extract text from chat-style message dicts (subset of mini-swe-agent)."""

from __future__ import annotations

import json


def _format_tool_call(args_str: str) -> str:
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
        if isinstance(args, dict) and "command" in args:
            return f"```\n{args['command']}\n```"
    except Exception:
        pass
    return f"```\n{args_str}\n```"


def _format_observation(content: str) -> str | None:
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "returncode" in data:
            lines = []
            for key, value in data.items():
                lines.append(f"<{key}>")
                lines.append(str(value))
            return "\n".join(lines)
        return content
    except Exception:
        return content


def get_content_string(message: dict) -> str:
    texts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        fo = _format_observation(content)
        if fo:
            texts.append(fo)
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool_use":
                texts.append(_format_tool_call(json.dumps(item.get("input", {}))))
            elif item.get("type") == "tool_result":
                rc = item.get("content", "")
                if isinstance(rc, str):
                    fo = _format_observation(rc)
                    if fo:
                        texts.append(fo)
            elif text := item.get("text"):
                texts.append(text)

    if tool_calls := message.get("tool_calls"):
        for tc in tool_calls:
            func = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
            if func:
                args = func.get("arguments", "{}") if isinstance(func, dict) else getattr(func, "arguments", "{}")
                texts.append(_format_tool_call(args))

    if output := message.get("output"):
        if isinstance(output, str):
            fo = _format_observation(output)
            if fo:
                texts.append(fo)
        elif isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if isinstance(c, dict) and (text := c.get("text")):
                            texts.append(text)
                elif item.get("type") == "function_call":
                    texts.append(_format_tool_call(item.get("arguments", "{}")))

    return "\n\n".join(t for t in texts if t)
