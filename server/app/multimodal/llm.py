"""Provider-agnostic LLM caller with Gemini support and deterministic mock fallback.

Environment variables:
- GEMINI_API_KEY: Google Gemini API key
- MM_MODE: 'live' or 'mock' (default: 'live' if GEMINI_API_KEY present, else 'mock')
- MM_LLM_MODEL: Model name (default: 'gemini-2.5-flash')
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
import httpx

log = logging.getLogger("relearn.multimodal.llm")

ENV_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".env"

def _load_env_config():
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    mode = os.environ.get("MM_MODE", "live" if key else "mock").strip()
    model = os.environ.get("MM_LLM_MODEL", "gemini-2.5-flash").strip()
    # Normalize model if unknown version specified
    if "3.5" in model:
        model = "gemini-2.5-flash"
    return key, mode, model

# In-memory cache for repeated inputs
_CACHE: Dict[str, str] = {}


def is_live_available() -> bool:
    """Returns True if live mode is enabled and an API key is present."""
    key, mode, _ = _load_env_config()
    return bool(key and mode == "live")


def call_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    mock_default: Optional[str] = None,
) -> str:
    """Calls Gemini REST API or falls back to mock response."""
    key, mode, model = _load_env_config()
    cache_key = f"{model}:{hash(prompt)}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if not (key and mode == "live"):
        log.info("LLM running in mock mode (no GEMINI_API_KEY or MM_MODE=mock)")
        return mock_default or "{}"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json" if "JSON" in prompt else "text/plain",
        },
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                _CACHE[cache_key] = text
                return text
            else:
                log.warning("Gemini API returned status %d: %s; using mock", resp.status_code, resp.text)
                return mock_default or "{}"
    except Exception as exc:
        log.warning("Gemini API call failed (%s); using mock fallback", exc)
        return mock_default or "{}"
