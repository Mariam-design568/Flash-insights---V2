from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def canonical_url(url: str) -> str:
    if not isinstance(url, str):
        raise ValueError("L'URL doit être une chaîne de caractères.")
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise ValueError("L'URL source doit utiliser HTTP ou HTTPS.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("L'URL source ne doit pas contenir d'identifiants.")
    host = parsed.hostname.lower()
    if ":" in host:
        host = "[" + host + "]"
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Le port de l'URL est invalide.") from exc
    if port and port not in (80, 443):
        host = f"{host}:{port}"
    query = sorted(
        (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    )
    return urlunsplit(("https", host, parsed.path or "/", urlencode(query), ""))


def _author_hash(author: object) -> str | None:
    if not isinstance(author, str) or not author.strip():
        return None
    normalized = " ".join(author.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _published_date(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.date().isoformat()
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def normalize_mention(raw: dict, connector: str) -> dict:
    """Map a connector item to the shared, provenance-preserving mention shape."""
    content = raw.get("content")
    source_url = raw.get("source_url") or raw.get("url")
    platform = raw.get("platform") or connector
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Mention sans contenu exploitable.")
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("Mention sans URL source.")
    canonical = canonical_url(source_url)
    if not isinstance(platform, str) or not platform.strip():
        raise ValueError("Mention sans plateforme.")
    if len(source_url.strip()) > 1000:
        raise ValueError("L'URL source dépasse 1 000 caractères.")
    collected_at = raw.get("collected_at")
    if isinstance(collected_at, datetime):
        collected = collected_at.isoformat(timespec="seconds")
    elif isinstance(collected_at, str) and collected_at.strip():
        collected = collected_at.strip()
    else:
        collected = datetime.now(timezone.utc).isoformat(timespec="seconds")
    language = raw.get("language")
    if isinstance(language, str):
        language = language.strip().lower() or None
        if language and not re.fullmatch(r"[a-z]{2,3}", language):
            language = None
    else:
        language = None
    raw_hash = hashlib.sha256(
        (platform.strip() + "|" + canonical + "|" + content[:512]).encode("utf-8")
    ).hexdigest()
    product_id = raw.get("product_id")
    if product_id is not None:
        if isinstance(product_id, bool):
            raise ValueError("product_id doit être un entier ou null.")
        try:
            product_id = int(product_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("product_id doit être un entier ou null.") from exc
        if product_id <= 0:
            raise ValueError("product_id doit être strictement positif.")
    source_name = raw.get("source_name") or connector
    return {
        "product_id": product_id,
        "platform": platform.strip()[:80],
        "source_name": str(source_name).strip()[:160] or connector,
        "source_url": source_url.strip()[:1000],
        "content": content.strip()[:5000],
        "published_on": _published_date(raw.get("published_on")),
        "collected_at": collected,
        "author_hash": _author_hash(raw.get("author")),
        "language": language,
        "connector": connector,
        "raw_hash": raw_hash,
    }
