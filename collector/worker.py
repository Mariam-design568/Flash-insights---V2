from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app import (
    DATABASE_PATH,
    SENTIMENT_TERMS,
    analyze_sentiment,
    initialize_database,
    normalize_text,
)
from collector.dedupe import insert_mention, mention_exists
from collector.enrich_llm import analyze
from collector.normalize import normalize_mention

APP_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = APP_DIR / "collector.log"
LOCK_PATH = APP_DIR / "collector.lock"
TRIGGER_PATH = APP_DIR / "collector.trigger"
MAX_BACKOFF_SECONDS = 3600
CONNECTORS = {"rss"}
SCHEMA = {
    "type": "object",
    "required": ["score", "label", "strengths", "weaknesses", "language", "themes"],
    "properties": {
        "score": {"type": "number", "minimum": -1, "maximum": 1},
        "label": {"type": "string", "enum": ["POSITIF", "NEUTRE", "NEGATIF"]},
        "strengths": {"type": "array", "maxItems": 5, "items": {"type": "string", "maxLength": 120}},
        "weaknesses": {"type": "array", "maxItems": 5, "items": {"type": "string", "maxLength": 120}},
        "language": {"type": "string", "nullable": True, "maxLength": 3},
        "themes": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 80}},
    },
}
LOGGER = logging.getLogger("collector")


class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        try:
            event = json.loads(message)
            if isinstance(event, dict):
                return message
        except json.JSONDecodeError:
            pass
        return json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "connector": record.name.rsplit(".", 1)[-1],
            "items_seen": 0,
            "items_inserted": 0,
            "duplicates_skipped": 0,
            "errors": 0,
            "status": "skipped" if message.startswith("skipped:") else "info",
            "message": message,
        }, ensure_ascii=False)


def _configure_logging() -> None:
    if LOGGER.handlers:
        return
    formatter = _StructuredFormatter()
    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    LOGGER.propagate = False
    for name in CONNECTORS:
        logging.getLogger("collector." + name).setLevel(logging.INFO)


def _log(event: dict) -> None:
    _configure_logging()
    LOGGER.info(json.dumps(
        {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event},
        ensure_ascii=False,
    ))


def _safe_error(exc: Exception) -> str:
    message = str(exc)
    for key in ("LLM_API_KEY",):
        secret = os.environ.get(key, "").strip()
        if secret:
            message = message.replace(secret, "[redacted]")
    return type(exc).__name__ + ": " + message[:300]


def _config() -> dict:
    try:
        interval = int(os.environ.get("COLLECT_INTERVAL_SECONDS", "300"))
        max_items = int(os.environ.get("COLLECTOR_MAX_ITEMS_PER_RUN", "10"))
    except ValueError as exc:
        raise ValueError("COLLECT_INTERVAL_SECONDS et COLLECTOR_MAX_ITEMS_PER_RUN doivent être des entiers.") from exc
    if interval < 1 or max_items < 1:
        raise ValueError("L'intervalle et le maximum d'éléments doivent être positifs.")
    configured = os.environ.get("ENABLED_CONNECTORS", "rss").strip()
    names = [] if configured.lower() in ("none", "off") else [
        name.strip().lower() for name in configured.split(",") if name.strip()
    ]
    unknown = set(names) - CONNECTORS
    if unknown:
        raise ValueError("Connecteurs inconnus : " + ", ".join(sorted(unknown)))
    return {"interval": interval, "max_items": max_items, "connectors": names}


def _fallback_analysis(content: str, language: str | None) -> dict:
    score, label = analyze_sentiment(content)
    normalized = normalize_text(content)
    strengths = [
        term for term in SENTIMENT_TERMS["positive"]
        if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", normalized)
    ][:5]
    weaknesses = [
        term for term in SENTIMENT_TERMS["negative"]
        if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", normalized)
    ][:5]
    return {
        "score": score,
        "label": label,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "language": language,
        "themes": [],
        "analysis_source": "fallback_lexicon",
        "model_version": None,
    }


def _analysis(mention: dict) -> dict:
    result = analyze(mention["content"], SCHEMA)
    if result is None:
        return _fallback_analysis(mention["content"], mention.get("language"))
    model = os.environ.get("LLM_MODEL", "gemini-3.8-flash").strip()
    return {
        **result,
        "analysis_source": "llm:" + model,
        "model_version": model,
        "llm_payload": result,
    }


def _connector_config(db: sqlite3.Connection) -> dict:
    products = [
        {"id": row["id"], "name": row["name"]}
        for row in db.execute("SELECT id, name FROM products ORDER BY id")
    ]
    return {"products": products}


def _run_row(
    db: sqlite3.Connection, connector: str, started: str, seen: int,
    inserted: int, duplicates: int, errors: list[str], status: str,
) -> None:
    db.execute(
        "INSERT INTO collection_runs(connector, started_at, finished_at, items_seen, "
        "items_inserted, duplicates_skipped, errors, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            connector, started, datetime.now(timezone.utc).isoformat(timespec="seconds"),
            seen, inserted, duplicates, json.dumps(errors, ensure_ascii=False), status,
        ),
    )


def run_once(
    database_path: Path | str | None = None,
    connectors: list[str] | None = None,
) -> dict:
    """Run enabled connectors once, recording a run row for each connector."""
    _configure_logging()
    config = _config()
    selected = config["connectors"] if connectors is None else connectors
    if any(name not in CONNECTORS for name in selected):
        raise ValueError("Un connecteur demandé est inconnu.")
    initialize_database(database_path or DATABASE_PATH)
    db = sqlite3.connect(database_path or DATABASE_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA journal_mode = WAL")
    totals = {"items_seen": 0, "items_inserted": 0, "duplicates_skipped": 0, "errors": 0}
    try:
        if not selected:
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with db:
                _run_row(db, "worker", started, 0, 0, 0, [], "ok")
            _log({
                "connector": "worker", "items_seen": 0, "items_inserted": 0,
                "duplicates_skipped": 0, "errors": 0, "status": "ok",
            })
            return {**totals, "status": "ok"}
        connector_config = _connector_config(db)
        remaining = config["max_items"]
        for name in selected:
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            seen = inserted = duplicates = 0
            errors = []
            status = "ok"
            try:
                module = importlib.import_module("collector.connectors." + name)
                for mention in module.fetch(connector_config):
                    if remaining <= 0:
                        break
                    remaining -= 1
                    seen += 1
                    try:
                        if mention.get("connector") != name:
                            mention = normalize_mention(mention, name)
                        if mention_exists(db, mention):
                            duplicates += 1
                            continue
                        analysis = _analysis(mention)
                        with db:
                            if insert_mention(db, mention, analysis):
                                inserted += 1
                            else:
                                duplicates += 1
                    except (ValueError, sqlite3.Error, KeyError, TypeError) as exc:
                        errors.append(_safe_error(exc))
            except Exception as exc:
                status = "error"
                errors.append(_safe_error(exc))
            if errors and status == "ok":
                status = "partial"
            with db:
                _run_row(db, name, started, seen, inserted, duplicates, errors, status)
            totals["items_seen"] += seen
            totals["items_inserted"] += inserted
            totals["duplicates_skipped"] += duplicates
            totals["errors"] += len(errors)
            _log({
                "connector": name, "items_seen": seen, "items_inserted": inserted,
                "duplicates_skipped": duplicates, "errors": len(errors),
                "status": status, "error_details": errors,
            })
        statuses = [
            row["status"] for row in db.execute(
                "SELECT status FROM collection_runs ORDER BY id DESC LIMIT ?", (len(selected),)
            )
        ]
        totals["status"] = (
            "error" if "error" in statuses
            else "partial" if "partial" in statuses
            else "ok"
        )
        return totals
    finally:
        db.close()


def _acquire_lock():
    try:
        descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if not worker_is_running():
            return _acquire_lock()
        raise RuntimeError(f"Un worker semble déjà actif (verrou : {LOCK_PATH}).") from None
    with os.fdopen(descriptor, "w", encoding="ascii") as lock_file:
        lock_file.write(str(os.getpid()))
    return True


def _release_lock() -> None:
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def worker_is_running() -> bool:
    try:
        pid = int(LOCK_PATH.read_text(encoding="ascii").strip())
    except FileNotFoundError:
        return False
    except ValueError:
        try:
            age = time.time() - LOCK_PATH.stat().st_mtime
        except OSError:
            return False
        if age < 5:
            return True
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass
        return False
    except OSError:
        return True
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass
        return False
    return True


def serve_forever() -> None:
    config = _config()
    _acquire_lock()
    stopping = threading.Event()
    previous_handlers = {}
    for signum in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        try:
            previous_handlers[signum] = signal.signal(signum, lambda _sig, _frame: stopping.set())
        except ValueError:
            pass
    delays = {name: config["interval"] for name in config["connectors"]}
    due = {name: 0.0 for name in config["connectors"]}
    try:
        if not config["connectors"]:
            run_once(connectors=[])
        while not stopping.is_set() and config["connectors"]:
            if TRIGGER_PATH.exists():
                try:
                    TRIGGER_PATH.unlink()
                except FileNotFoundError:
                    pass
                due = {name: 0.0 for name in config["connectors"]}
            now = time.monotonic()
            ready = [name for name in config["connectors"] if due[name] <= now]
            for name in ready:
                result = run_once(connectors=[name])
                if result["errors"]:
                    delays[name] = min(
                        MAX_BACKOFF_SECONDS,
                        max(config["interval"], delays[name] * 2),
                    )
                else:
                    delays[name] = config["interval"]
                due[name] = time.monotonic() + delays[name]
            wait = max(0.1, min(due.values()) - time.monotonic())
            stopping.wait(min(wait, 1.0))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        _release_lock()


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker de collecte Flash Insights")
    parser.add_argument("--once", action="store_true", help="Exécuter une collecte puis quitter")
    args = parser.parse_args()
    try:
        if args.once:
            _acquire_lock()
            try:
                try:
                    TRIGGER_PATH.unlink()
                except FileNotFoundError:
                    pass
                run_once()
            finally:
                _release_lock()
        else:
            serve_forever()
    except (RuntimeError, ValueError, sqlite3.Error) as exc:
        _log({"connector": "worker", "items_seen": 0, "errors": 1, "status": "error",
              "error_details": [str(exc)[:300]]})
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
