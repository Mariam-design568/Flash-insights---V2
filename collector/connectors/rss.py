from __future__ import annotations

import html
import logging
import os
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

from collector.connectors._common import TokenBucket
from collector.normalize import normalize_mention

LOGGER = logging.getLogger("collector.rss")
_LIMITER = TokenBucket(1.0, capacity=2)  # Maximum déclaré : 60 requêtes RSS par minute.


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    return " ".join(" ".join(parser.parts).split())


def _child_text(element: ET.Element, local_name: str) -> str:
    for child in element.iter():
        if child.tag.rsplit("}", 1)[-1].lower() == local_name.lower() and child.text:
            return child.text.strip()
    return ""


def _source_url(element: ET.Element) -> str:
    for child in element.iter():
        if child.tag.rsplit("}", 1)[-1].lower() == "link":
            href = child.attrib.get("href")
            if href:
                return href.strip()
            if child.text and child.text.strip():
                return child.text.strip()
    return ""


def _feeds(config: dict) -> list[dict]:
    configured = config.get("feeds")
    if configured is not None:
        return configured
    feeds = []
    for entry in os.environ.get("RSS_FEEDS", "").split(";"):
        parts = entry.strip().split("|", 2)
        if len(parts) >= 2 and parts[0].strip():
            try:
                product_id = int(parts[1])
            except ValueError:
                LOGGER.warning("RSS feed ignoré : product_id invalide")
                continue
            feeds.append({
                "url": parts[0].strip(),
                "product_id": product_id,
                "source_name": parts[2].strip() if len(parts) > 2 else parts[0].strip(),
            })
    return feeds


def fetch(config: dict):
    """Read configured public RSS/Atom feeds; no feed is fetched unless explicitly configured."""
    for feed in _feeds(config):
        if not feed.get("product_id"):
            LOGGER.warning("RSS feed ignoré : association de produit requise")
            continue
        _LIMITER.acquire()
        request = urllib.request.Request(
            feed["url"], headers={"User-Agent": "FlashInsights/1.0 (RSS reader)"}
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read(5_000_001)
        if len(body) > 5_000_000:
            raise ValueError("Le flux RSS dépasse 5 Mo.")
        document = ET.fromstring(body)
        for item in document.iter():
            kind = item.tag.rsplit("}", 1)[-1].lower()
            if kind not in ("item", "entry"):
                continue
            title = _child_text(item, "title")
            description = _child_text(item, "description") or _child_text(item, "summary")
            content = _child_text(item, "encoded") or description
            if title and title.casefold() not in content.casefold():
                content = title + (". " + content if content else "")
            url = _source_url(item)
            if not url:
                continue
            raw = {
                "product_id": feed["product_id"],
                "platform": "RSS",
                "source_name": feed.get("source_name") or feed["url"],
                "source_url": url,
                "content": _plain_text(html.unescape(content))[:5000],
                "published_on": _child_text(item, "pubDate") or
                _child_text(item, "published") or _child_text(item, "updated"),
                "author": _child_text(item, "author"),
            }
            if raw["content"]:
                yield normalize_mention(raw, "rss")
