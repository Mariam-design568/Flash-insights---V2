from __future__ import annotations

import json
import sqlite3
from datetime import date


def mention_exists(db: sqlite3.Connection, mention: dict) -> bool:
    row = db.execute(
        "SELECT 1 FROM product_mentions WHERE product_id = ? AND raw_hash = ? LIMIT 1",
        (mention["product_id"], mention["raw_hash"]),
    ).fetchone()
    return row is not None


def insert_mention(
    db: sqlite3.Connection,
    mention: dict,
    analysis: dict,
) -> bool:
    """Insert a mention once by product and content hash; return False on duplicates."""
    product_id = mention.get("product_id")
    if product_id is None:
        raise ValueError("La mention doit être explicitement associée à un produit.")
    if not mention.get("published_on"):
        raise ValueError("La mention n'a pas de date de publication vérifiable.")
    try:
        date.fromisoformat(mention["published_on"])
    except (TypeError, ValueError) as exc:
        raise ValueError("La date de publication de la mention est invalide.") from exc
    cursor = db.execute(
        "INSERT OR IGNORE INTO product_mentions("
        "product_id, platform, source_name, source_url, content, published_on, collected_at, "
        "sentiment_score, sentiment_label, sentiment_strengths, sentiment_weaknesses, "
        "connector, raw_hash, author_hash, language, analysis_source, model_version, llm_payload"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            product_id, mention["platform"], mention["source_name"], mention["source_url"],
            mention["content"], mention["published_on"], mention["collected_at"],
            analysis["score"], analysis["label"],
            json.dumps(analysis.get("strengths", []), ensure_ascii=False),
            json.dumps(analysis.get("weaknesses", []), ensure_ascii=False),
            mention["connector"], mention["raw_hash"], mention.get("author_hash"),
            analysis.get("language") or mention.get("language"),
            analysis["analysis_source"], analysis.get("model_version"),
            json.dumps(analysis.get("llm_payload"), ensure_ascii=False)
            if analysis.get("llm_payload") is not None else None,
        ),
    )
    return cursor.rowcount == 1
