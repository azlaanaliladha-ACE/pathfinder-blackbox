"""Nebius Token Factory access: one OpenAI-compatible client and model-tier resolution.

Model ids differ between docs pages (casing, prefixes), so each tier is resolved against the
account's live /v1/models list: an exact env/config id wins, otherwise the first NVIDIA model
whose id contains the tier's keywords.
"""
import os
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def client():
    from openai import OpenAI
    return OpenAI(base_url=config.NEBIUS_BASE_URL, api_key=os.environ["NEBIUS_API_KEY"], timeout=60)


@lru_cache(maxsize=1)
def available() -> tuple[str, ...]:
    try:
        return tuple(m.id for m in client().models.list())
    except Exception:
        return ()


@lru_cache(maxsize=None)
def model(tier: str) -> str:
    """tier: 'fast' (in-flight), 'reason' (planning + debrief)."""
    wanted, keywords = config.MODEL_TIERS[tier]
    ids = available()
    if not ids or wanted in ids:
        return wanted
    for kws in keywords:                          # keyword sets in order of preference
        hits = [i for i in ids if "nemotron" in i.lower() and all(k in i.lower() for k in kws)]
        if hits:
            return sorted(hits, key=len)[0]
    return wanted


def chat(model_id: str, messages: list, max_tokens: int, **kw):
    return client().chat.completions.create(model=model_id, messages=messages, max_tokens=max_tokens,
                                            temperature=kw.pop("temperature", 0.2), **kw)


def strip_think(text: str) -> str:
    """Reasoning models may prepend <think>...</think>."""
    return (text or "").split("</think>")[-1].strip()
