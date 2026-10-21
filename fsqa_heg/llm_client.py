"""DeepSeek chat client (extracted for the FSQA HE-G pipeline)."""

from __future__ import annotations

import os
import time

try:
    import requests
except ImportError:
    requests = None  # type: ignore


def get_deepseek_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY")
    return key


def run_deepseek_chat(
    messages: list[dict],
    *,
    model: str = "deepseek-chat",
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 4096,
    temperature: float | None = 0.3,
    timeout: int = 180,
) -> str:
    """OpenAI-compatible DeepSeek chat completions (https://api.deepseek.com)."""
    if requests is None:
        raise RuntimeError("需要 requests 包: pip install requests")
    api_key = api_key or get_deepseek_api_key()
    base_url = (base_url or os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")).rstrip("/")
    url = base_url + "/v1/chat/completions"
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if "reasoner" not in model and temperature is not None:
        payload["temperature"] = temperature

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    wait = 2.0
    last_err: Exception | None = None
    for _ in range(8):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code in (429,) or resp.status_code >= 500:
                time.sleep(wait)
                wait = min(wait * 2, 120.0)
                continue
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            text = (msg.get("content") or "").strip()
            if not text and msg.get("reasoning_content"):
                text = str(msg.get("reasoning_content", "")).strip()
            return text
        except Exception as e:
            last_err = e
            time.sleep(wait)
            wait = min(wait * 2, 120.0)
    raise RuntimeError(f"DeepSeek 请求失败: {last_err}")


def run_deepseek_with_examples(
    system_prompt: str,
    examples: list,
    user_input: str,
    *,
    model: str = "deepseek-chat",
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 4096,
    temperature: float | None = 0.3,
    timeout: int = 180,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    for inp, out in examples:
        messages.append({"role": "user", "content": inp})
        messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": user_input})
    return run_deepseek_chat(
        messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
