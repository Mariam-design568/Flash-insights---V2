from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request


SYSTEM_PROMPT = (
    "Tu extrais uniquement des faits présents dans le texte. "
    "Tu réponds en JSON conforme au schéma fourni. Si une information est absente, "
    "tu mets null. Tu n'inventes aucun chiffre, aucune tendance, aucune conclusion."
)
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


def _pace_request() -> None:
    global _LAST_REQUEST_AT
    try:
        interval = float(os.environ.get("LLM_MIN_REQUEST_INTERVAL_SECONDS", "15"))
    except ValueError as exc:
        raise ValueError("LLM_MIN_REQUEST_INTERVAL_SECONDS doit être un nombre positif.") from exc
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("LLM_MIN_REQUEST_INTERVAL_SECONDS doit être un nombre positif.")
    with _REQUEST_LOCK:
        wait = interval - (time.monotonic() - _LAST_REQUEST_AT)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_AT = time.monotonic()


def _validate(value: object, schema: dict) -> dict | None:
    if not isinstance(value, dict):
        return None
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return None
    if any(key not in value for key in required):
        return None
    result = {}
    for key, definition in properties.items():
        item = value.get(key)
        kind = definition.get("type")
        nullable = definition.get("nullable", False)
        if item is None and nullable:
            result[key] = None
            continue
        if kind == "number":
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return None
            item = float(item)
            if not math.isfinite(item):
                return None
            minimum, maximum = definition.get("minimum"), definition.get("maximum")
            if minimum is not None and item < minimum or maximum is not None and item > maximum:
                return None
        elif kind == "string":
            if not isinstance(item, str) or len(item) > definition.get("maxLength", 500):
                return None
            if definition.get("enum") and item not in definition["enum"]:
                return None
            if key == "language" and item and not re.fullmatch(r"[a-z]{2,3}", item):
                return None
        elif kind == "array":
            if not isinstance(item, list) or len(item) > definition.get("maxItems", 10):
                return None
            item_definition = definition.get("items", {"type": "string"})
            if item_definition.get("type") == "string" and any(
                not isinstance(entry, str) or len(entry) > item_definition.get("maxLength", 120)
                for entry in item
            ):
                return None
        else:
            return None
        result[key] = item
    return result


def analyze(text: str, schema: dict) -> dict | None:
    """Return schema-validated JSON, or None when disabled or anything is invalid."""
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key or not isinstance(text, str) or not text.strip():
        return None
    endpoint = os.environ.get(
        "LLM_API_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    ).strip()
    model = os.environ.get("LLM_MODEL", "gemini-3.8-flash").strip()
    if not endpoint or not model:
        return None
    _pace_request()
    payload = json.dumps({
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                {"text": text[:5000], "schema": schema}, ensure_ascii=False
            )},
        ],
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = json.loads(response.read().decode("utf-8"))
        content = response_body["choices"][0]["message"]["content"]
        return _validate(json.loads(content), schema)
    except (
        urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError,
        json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError, AttributeError,
    ):
        return None
