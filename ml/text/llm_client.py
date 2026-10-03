"""DeepSeek client for offline data generation (plans/05 §7).

Reads DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL and DEEPSEEK_MODEL from the root .env.
Every response is saved under ml/data/llm_raw/<job>/<hash>.json, so a rerun reads from
disk instead of paying again. The key is never printed and never written to disk.

The running app never imports this module.
"""
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "ml" / "data" / "llm_raw"

_client = None


def _config():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY")
    model = os.environ.get("DEEPSEEK_MODEL")
    if not key or not model:
        raise RuntimeError("Set DEEPSEEK_API_KEY and DEEPSEEK_MODEL in the root .env (see .env.example).")
    return key, os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), model


def model_name():
    return _config()[2]


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        key, base_url, _ = _config()
        _client = OpenAI(api_key=key, base_url=base_url, timeout=90, max_retries=0)
    return _client


def _cache_path(job, model, messages, temperature, thinking):
    key = {"model": model, "messages": messages, "temperature": temperature}
    if thinking:
        key["thinking"] = True
    payload = json.dumps(key, sort_keys=True)
    return RAW / job / (hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20] + ".json")


def chat_json(job, messages, temperature=1.0, retries=4, thinking=False):
    """One chat call that must return a JSON object. Returns (parsed, from_cache).

    thinking=False switches off the model's hidden reasoning. Measured on one sentence call:
    about 150 output tokens instead of about 2,000, with no visible loss for this kind of writing.
    """
    model = model_name()
    path = _cache_path(job, model, messages, temperature, thinking)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["parsed"], True

    last_error = None
    for attempt in range(retries):
        try:
            response = _get_client().chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}})
            content = response.choices[0].message.content
            parsed = json.loads(content)
            usage = response.usage
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "job": job, "model": model, "temperature": temperature, "messages": messages,
                "content": content, "parsed": parsed,
                "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            return parsed, False
        except Exception as exc:  # network errors, rate limits and bad JSON are all retried
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"DeepSeek call failed after {retries} tries: {type(last_error).__name__}: {last_error}")


def chat_json_many(job, message_lists, temperature=1.0, workers=8):
    """Run many calls side by side. Returns a list of (parsed or None, from_cache, error or None)."""
    def one(messages):
        try:
            parsed, cached = chat_json(job, messages, temperature)
            return parsed, cached, None
        except RuntimeError as exc:
            return None, False, str(exc)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, message_lists))


def usage_summary(job):
    """Tokens used by a job so far, read from the saved responses."""
    prompt = completion = calls = 0
    for path in (RAW / job).glob("*.json"):
        usage = json.loads(path.read_text(encoding="utf-8"))["usage"]
        prompt, completion, calls = prompt + usage["prompt_tokens"], completion + usage["completion_tokens"], calls + 1
    return {"calls": calls, "prompt_tokens": prompt, "completion_tokens": completion}
