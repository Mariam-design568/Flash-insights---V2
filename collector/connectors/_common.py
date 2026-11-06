from __future__ import annotations

import json
import time
import urllib.request


class TokenBucket:
    def __init__(self, rate_per_second: float, capacity: int = 1):
        self.rate = rate_per_second
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.updated = time.monotonic()

    def acquire(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens < 1:
            time.sleep((1 - self.tokens) / self.rate)
            self.tokens = 0
            self.updated = time.monotonic()
        else:
            self.tokens -= 1


def fetch_json(url: str, headers: dict | None = None, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"User-Agent": "FlashInsights/1.0", **(headers or {})},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read(5_000_001)
    if len(raw) > 5_000_000:
        raise ValueError("La réponse du connecteur dépasse 5 Mo.")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("La réponse du connecteur n'est pas un objet JSON.")
    return value
