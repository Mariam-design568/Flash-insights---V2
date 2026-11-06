import io
import json
import os
import tempfile
import threading
import unittest
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from app import connect_db, initialize_database, render_data, render_reputation
from collector.dedupe import insert_mention
from collector.enrich_llm import _pace_request, analyze
from collector.normalize import canonical_url, normalize_mention
from collector.worker import SCHEMA, _analysis, run_once


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "collector-test.db"
        initialize_database(self.database)
        self.db = connect_db(self.database)
        self.product_id = self.db.execute(
            "SELECT id FROM products ORDER BY id LIMIT 1"
        ).fetchone()["id"]
        self.previous_env = {
            key: os.environ.get(key)
            for key in (
                "LLM_API_KEY", "LLM_MODEL", "LLM_API_BASE_URL",
                "ENABLED_CONNECTORS", "RSS_FEEDS", "COLLECTOR_MAX_ITEMS_PER_RUN",
                "FLASH_INSIGHTS_DB", "LLM_MIN_REQUEST_INTERVAL_SECONDS",
            )
        }
        os.environ.pop("LLM_API_KEY", None)
        os.environ["LLM_MODEL"] = "mock-model"
        os.environ["LLM_MIN_REQUEST_INTERVAL_SECONDS"] = "0"
        os.environ["FLASH_INSIGHTS_DB"] = str(self.database)

    def tearDown(self):
        self.db.close()
        for key, value in self.previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp_dir.cleanup()

    def test_canonical_url_removes_tracking_normalizes_host_and_sorts_query(self):
        first = canonical_url(
            "HTTP://Example.COM/path?z=2&utm_source=news&a=1&utm_campaign=x#fragment"
        )
        second = canonical_url("https://example.com/path?a=1&z=2")
        self.assertEqual(first, second)
        self.assertEqual(first, "https://example.com/path?a=1&z=2")

    def test_raw_hash_is_stable_and_sensitive_to_content(self):
        raw = {
            "product_id": self.product_id,
            "platform": "RSS",
            "source_url": "http://example.org/a?utm_source=x",
            "content": "Un avis fiable.",
            "published_on": "2026-10-20",
        }
        first = normalize_mention(raw, "rss")
        again = normalize_mention({**raw, "source_url": "https://EXAMPLE.org/a"}, "rss")
        changed = normalize_mention({**raw, "content": "Un avis fragile."}, "rss")
        self.assertEqual(first["raw_hash"], again["raw_hash"])
        self.assertNotEqual(first["raw_hash"], changed["raw_hash"])

    def test_normalize_maps_fields_and_hashes_author_without_retaining_it(self):
        mention = normalize_mention({
            "product_id": self.product_id,
            "platform": "Forum",
            "source_name": "Forum public",
            "source_url": "https://example.org/post/1",
            "content": "Très bon produit.",
            "published_on": "2026-10-20T15:00:00Z",
            "author": "Auteur à ne pas conserver",
            "language": "fr",
        }, "rss")
        self.assertEqual(mention["product_id"], self.product_id)
        self.assertEqual(mention["published_on"], "2026-10-20")
        self.assertEqual(mention["language"], "fr")
        self.assertTrue(mention["author_hash"])
        self.assertNotIn("author", mention)
        self.assertEqual(mention["connector"], "rss")

    def test_dedupe_ignores_second_insert_and_counts_it(self):
        mention = normalize_mention({
            "product_id": self.product_id,
            "platform": "RSS",
            "source_name": "Site",
            "source_url": "https://example.org/duplicate",
            "content": "Excellent et fiable.",
            "published_on": date.today().isoformat(),
        }, "rss")
        analysis = {
            "score": 1.0, "label": "POSITIF", "strengths": ["excellent"],
            "weaknesses": [], "analysis_source": "fallback_lexicon",
        }
        self.assertTrue(insert_mention(self.db, mention, analysis))
        duplicates_skipped = 0
        if not insert_mention(self.db, mention, analysis):
            duplicates_skipped += 1
        count = self.db.execute(
            "SELECT COUNT(*) FROM product_mentions WHERE product_id = ? AND raw_hash = ?",
            (self.product_id, mention["raw_hash"]),
        ).fetchone()[0]
        self.assertEqual(count, 1)
        self.assertEqual(duplicates_skipped, 1)

    def test_collection_and_mention_provenance_are_visible_in_both_ui_areas(self):
        mention = normalize_mention({
            "product_id": self.product_id,
            "platform": "RSS",
            "source_name": "Source publique",
            "source_url": "https://example.org/ui-provenance",
            "content": "Produit excellent et fiable.",
            "published_on": date.today().isoformat(),
        }, "rss")
        insert_mention(self.db, mention, {
            "score": 1.0, "label": "POSITIF", "strengths": ["excellent"],
            "weaknesses": [], "analysis_source": "fallback_lexicon",
        })
        self.db.execute(
            "INSERT INTO collection_runs(connector, started_at, finished_at, items_seen, "
            "items_inserted, duplicates_skipped, errors, status) "
            "VALUES ('rss', '2026-10-29T10:00:00', '2026-10-29T10:00:01', 1, 1, 0, '[]', 'ok')"
        )
        self.db.commit()
        for page in (render_data(self.db), render_reputation(self.db, [self.product_id])):
            self.assertIn("Dernière collecte", page)
            self.assertIn("Source publique", page)
            self.assertIn("https://example.org/ui-provenance", page)
            self.assertIn("fallback_lexicon", page)

    def test_enrich_llm_fallback_uses_local_lexicon_and_marks_source(self):
        mention = {"content": "Excellent, je recommande ce produit.", "language": "fr"}
        analysis = _analysis(mention)
        self.assertEqual(analysis["label"], "POSITIF")
        self.assertEqual(analysis["analysis_source"], "fallback_lexicon")

    @staticmethod
    def _llm_response(value):
        payload = {"choices": [{"message": {"content": json.dumps(value)}}]}
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    def test_enrich_llm_mock_validates_and_falls_back_on_invalid_response(self):
        os.environ["LLM_API_KEY"] = "test-key-not-real"
        valid = {
            "score": 0.75,
            "label": "POSITIF",
            "strengths": ["autonomie"],
            "weaknesses": [],
            "language": "fr",
            "themes": ["autonomie"],
        }
        with patch("urllib.request.urlopen", return_value=self._llm_response(valid)) as mocked:
            self.assertEqual(analyze("Bonne autonomie.", SCHEMA)["score"], 0.75)
            self.assertEqual(
                mocked.call_args.args[0].get_header("Authorization"),
                "Bearer test-key-not-real",
            )
        invalid = {**valid, "score": 1.5}
        with patch("urllib.request.urlopen", return_value=self._llm_response(invalid)):
            self.assertIsNone(analyze("Bonne autonomie.", SCHEMA))
            fallback = _analysis({
                "content": "Excellent et fiable.",
                "language": "fr",
            })
        self.assertEqual(fallback["analysis_source"], "fallback_lexicon")
        self.assertEqual(fallback["label"], "POSITIF")

    def test_llm_requests_are_spaced_by_configured_interval(self):
        os.environ["LLM_MIN_REQUEST_INTERVAL_SECONDS"] = "15"
        with patch("collector.enrich_llm._LAST_REQUEST_AT", 100.0), \
                patch("collector.enrich_llm.time.monotonic", side_effect=(105.0, 115.0)), \
                patch("collector.enrich_llm.time.sleep") as sleep:
            _pace_request()
        sleep.assert_called_once_with(10.0)

    def test_worker_dry_run_records_ok_without_connectors(self):
        os.environ["ENABLED_CONNECTORS"] = ""
        result = run_once(self.database)
        run = self.db.execute(
            "SELECT connector, items_seen, items_inserted, status "
            "FROM collection_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertEqual(result["items_seen"], 0)
        self.assertEqual(run["connector"], "worker")
        self.assertEqual(run["items_inserted"], 0)
        self.assertEqual(run["status"], "ok")

    def test_worker_rejects_removed_api_connectors(self):
        os.environ["ENABLED_CONNECTORS"] = "youtube"
        with self.assertRaisesRegex(ValueError, "Connecteurs inconnus"):
            run_once(self.database)

    def test_worker_rss_collects_one_local_feed_item(self):
        body = """<?xml version="1.0"?>
          <rss version="2.0"><channel><title>Test</title><item>
            <title>Excellent produit</title>
            <description>Très bon et fiable.</description>
            <link>https://example.org/reviews/rss-1</link>
            <pubDate>Tue, 20 Oct 2026 10:00:00 GMT</pubDate>
            <author>Public author</author>
          </item></channel></rss>""".encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/rss+xml")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            os.environ["ENABLED_CONNECTORS"] = "rss"
            os.environ["RSS_FEEDS"] = (
                f"http://127.0.0.1:{server.server_port}/feed.xml|"
                f"{self.product_id}|Test RSS"
            )
            result = run_once(self.database)
            with patch(
                "collector.worker._analysis",
                side_effect=AssertionError("Duplicate mentions should not use Gemini"),
            ):
                duplicate_result = run_once(self.database)
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()
        mention = self.db.execute(
            "SELECT connector, source_url, analysis_source, raw_hash "
            "FROM product_mentions WHERE source_url = ?",
            ("https://example.org/reviews/rss-1",),
        ).fetchone()
        run = self.db.execute(
            "SELECT items_inserted, duplicates_skipped, status FROM collection_runs "
            "WHERE connector = 'rss' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertEqual(result["items_inserted"], 1)
        self.assertEqual(duplicate_result["items_inserted"], 0)
        self.assertEqual(duplicate_result["duplicates_skipped"], 1)
        self.assertEqual(mention["connector"], "rss")
        self.assertEqual(mention["analysis_source"], "fallback_lexicon")
        self.assertTrue(mention["raw_hash"])
        self.assertEqual(run["items_inserted"], 0)
        self.assertEqual(run["duplicates_skipped"], 1)
        self.assertEqual(run["status"], "ok")


if __name__ == "__main__":
    unittest.main()
