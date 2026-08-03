"""Cloud VLM backends via OpenAI-compatible chat completions APIs.

Works with OpenRouter, OpenAI, Groq, Together, Fireworks, and any other
endpoint that accepts Bearer auth + /chat/completions (vision image_url).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from .vlm import (
    apply_no_think_to_messages,
    pil_to_data_url,
    strip_thinking_text,
    tensor_to_pil,
)

logger = logging.getLogger("ProductionFlow.VLM.API")

# Presets: base URL for chat completions root (…/v1, not including /chat/completions).
API_PROVIDERS = {
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "qwen/qwen2.5-vl-72b-instruct",
        "env_key": "OPENROUTER_API_KEY",
        "openrouter_headers": True,
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "openrouter_headers": False,
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "env_key": "GROQ_API_KEY",
        "openrouter_headers": False,
    },
    "Together": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "env_key": "TOGETHER_API_KEY",
        "openrouter_headers": False,
    },
    "Fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "default_model": "accounts/fireworks/models/qwen2p5-vl-32b-instruct",
        "env_key": "FIREWORKS_API_KEY",
        "openrouter_headers": False,
    },
    "Custom (OpenAI-compatible)": {
        "base_url": "",
        "default_model": "",
        "env_key": "OPENAI_API_KEY",
        "openrouter_headers": False,
    },
}

API_PROVIDER_NAMES = list(API_PROVIDERS.keys())


def provider_default_model(provider: str) -> str:
    return API_PROVIDERS.get(provider, {}).get("default_model") or ""


def provider_default_base_url(provider: str) -> str:
    return API_PROVIDERS.get(provider, {}).get("base_url") or ""


def _resolve_api_key(provider: str, api_key: str) -> str:
    key = (api_key or "").strip()
    if key:
        return key
    env_name = API_PROVIDERS.get(provider, {}).get("env_key") or "OPENAI_API_KEY"
    env_val = os.environ.get(env_name, "").strip()
    if env_val:
        return env_val
    # Generic fallbacks
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return ""


def _normalize_base_url(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return url
    # Allow pasting full chat/completions URL by mistake
    for suffix in ("/chat/completions", "/completions"):
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    return url


def _chat_completions_url(base_url: str) -> str:
    base = _normalize_base_url(base_url)
    if not base:
        raise ValueError("API base URL is empty. Set base_url or pick a provider preset.")
    return base + "/chat/completions"


def _http_json(
    url: str,
    payload: dict,
    headers: dict,
    timeout: float = 180.0,
) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        if v is None or v == "":
            continue
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(
            f"VLM API HTTP {e.code} from {url}: {err_body[:2000] or e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"VLM API connection failed ({url}): {e.reason}") from e

    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"VLM API returned non-JSON: {body[:500]}") from e


def _extract_text(result: dict) -> str:
    # OpenAI-compatible
    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        # Some providers return content parts
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text") or "")
                elif isinstance(p, str):
                    parts.append(p)
            return "".join(parts)
        # Older text field
        text = choices[0].get("text")
        if isinstance(text, str):
            return text
    # Rare alternate shapes
    if isinstance(result.get("output_text"), str):
        return result["output_text"]
    raise RuntimeError(f"VLM API response missing choices[].message.content: {str(result)[:800]}")


class ApiVlmSession:
    """Drop-in session for ProductionFlowVLMGenerate (same .generate interface)."""

    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        app_url: str = "",
        app_name: str = "ComfyUI-ProductionFlow",
        timeout: float = 180.0,
    ):
        self.provider = provider
        self.base_url = _normalize_base_url(base_url) or provider_default_base_url(provider)
        self.api_key = api_key
        self.model = (model or "").strip()
        self.app_url = (app_url or "").strip()
        self.app_name = (app_name or "").strip() or "ComfyUI-ProductionFlow"
        self.timeout = float(timeout)
        self.info = type("Info", (), {"path": f"api:{provider}:{self.model}", "backend": "api"})()

    def unload(self):
        return None

    def _headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        meta = API_PROVIDERS.get(self.provider, {})
        if meta.get("openrouter_headers") or "openrouter" in self.base_url.lower():
            # Optional OpenRouter ranking headers
            headers["HTTP-Referer"] = self.app_url or "https://github.com/marvinlossa/ComfyUI-ProductionFlow"
            headers["X-OpenRouter-Title"] = self.app_name
            headers["X-Title"] = self.app_name  # older docs
        return headers

    def generate(
        self,
        prompt,
        image=None,
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        seed=0,
        enable_thinking=False,
    ):
        if not self.api_key:
            raise RuntimeError(
                f"No API key for {self.provider}. Paste a key in the loader, or set "
                f"{API_PROVIDERS.get(self.provider, {}).get('env_key', 'OPENAI_API_KEY')}."
            )
        if not self.model:
            raise RuntimeError("Model id is empty. Set the cloud model name (e.g. qwen/qwen2.5-vl-72b-instruct).")
        if not self.base_url:
            raise RuntimeError("Base URL is empty. Pick a provider or set a custom endpoint.")

        # Multimodal when image connected
        if image is not None:
            pil = tensor_to_pil(image)
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": pil_to_data_url(pil, max_side=1280)},
                },
                {"type": "text", "text": prompt},
            ]
            messages = [{"role": "user", "content": content}]
            mode = "vision+text"
        else:
            messages = [{"role": "user", "content": prompt}]
            mode = "text-only"

        messages = apply_no_think_to_messages(
            messages,
            enable_thinking=enable_thinking,
            model_path=self.model,
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(max(temperature, 0.0)),
            "top_p": float(top_p),
        }
        # Optional fields — ignore if provider rejects (we only send safe common ones)
        if top_k and int(top_k) > 0:
            # OpenAI rejects top_k; OpenRouter/Groq/Together often accept it
            if self.provider not in ("OpenAI",):
                payload["top_k"] = int(top_k)
        if seed:
            payload["seed"] = int(seed)

        url = _chat_completions_url(self.base_url)
        logger.info(
            "API VLM generate (%s, provider=%s, model=%s, max_tokens=%s)...",
            mode,
            self.provider,
            self.model,
            max_tokens,
        )
        try:
            result = _http_json(url, payload, self._headers(), timeout=self.timeout)
        except RuntimeError as e:
            # OpenAI newer models sometimes want max_completion_tokens
            msg = str(e).lower()
            if "max_tokens" in msg and "max_completion_tokens" in msg:
                payload.pop("max_tokens", None)
                payload["max_completion_tokens"] = int(max_tokens)
                result = _http_json(url, payload, self._headers(), timeout=self.timeout)
            elif "top_k" in msg or "unknown" in msg and "top_k" in str(payload):
                payload.pop("top_k", None)
                result = _http_json(url, payload, self._headers(), timeout=self.timeout)
            else:
                raise

        text = _extract_text(result)
        if not enable_thinking:
            text = strip_thinking_text(text)
        logger.info("API VLM finished (%s chars)", len(text or ""))
        return text or ""


def load_api_vlm_session(
    provider: str,
    api_key: str,
    model: str,
    base_url: str = "",
    app_url: str = "",
    app_name: str = "ComfyUI-ProductionFlow",
    timeout: float = 180.0,
) -> ApiVlmSession:
    if provider not in API_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {', '.join(API_PROVIDER_NAMES)}")

    resolved_url = _normalize_base_url(base_url) or provider_default_base_url(provider)
    resolved_key = _resolve_api_key(provider, api_key)
    resolved_model = (model or "").strip() or provider_default_model(provider)

    if provider.startswith("Custom") and not resolved_url:
        raise RuntimeError(
            "Custom provider needs a base_url (e.g. https://api.example.com/v1)."
        )

    return ApiVlmSession(
        provider=provider,
        base_url=resolved_url,
        api_key=resolved_key,
        model=resolved_model,
        app_url=app_url,
        app_name=app_name,
        timeout=timeout,
    )
