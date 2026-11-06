import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import (
    LLMError,
    analyze_sentiment,
    analyze_sentiment_with_llm,
    call_llm_json,
    connect_db,
    create_product_mention,
    create_signal,
    create_study,
    extract_data_with_llm,
    initialize_database,
    publish_study,
    render_dashboard,
    render_reputation,
    render_signal_detail,
    refresh_study,
    review_signal,
    review_study,
    sentiment_summary,
    submit_study,
)


class FlashInsightsWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.previous_llm_api_key = os.environ.pop("LLM_API_KEY", None)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "test.db"
        initialize_database(self.database)
        self.db = connect_db(self.database)
        self.sector_id = self.db.execute(
            "SELECT id FROM sectors WHERE name = 'E-commerce'"
        ).fetchone()["id"]

    def tearDown(self):
        self.db.close()
        self.temp_dir.cleanup()
        if self.previous_llm_api_key is None:
            os.environ.pop("LLM_API_KEY", None)
        else:
            os.environ["LLM_API_KEY"] = self.previous_llm_api_key

    def test_demo_data_initialization_does_not_call_llm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "seed.db"
            with patch(
                "app.analyze_sentiment_with_llm",
                side_effect=AssertionError("Demo seeding must not call the LLM"),
            ) as mocked:
                initialize_database(database)
            mocked.assert_not_called()
            db = connect_db(database)
            try:
                mention_count = db.execute(
                    "SELECT COUNT(*) FROM product_mentions"
                ).fetchone()[0]
                self.assertEqual(mention_count, 8)
                self.assertEqual(
                    db.execute(
                        "SELECT COUNT(*) FROM product_mentions "
                        "WHERE analysis_source = 'fallback_lexicon'"
                    ).fetchone()[0],
                    mention_count,
                )
            finally:
                db.close()

    def test_signal_alert_needs_distinct_sources_and_threshold(self):
        data_ids = [
            row["id"]
            for row in self.db.execute(
                "SELECT id FROM collected_data WHERE sector_id = ? ORDER BY id LIMIT 2",
                (self.sector_id,),
            )
        ]
        signal_id, alerted = create_signal(
            self.db, self.sector_id, "Test de recoupement", "Signal de test", data_ids, "Test"
        )
        self.assertTrue(alerted)
        signal = self.db.execute(
            "SELECT confidence, independent_sources, status FROM signals WHERE id = ?",
            (signal_id,),
        ).fetchone()
        self.assertEqual(signal["independent_sources"], 2)
        self.assertEqual(signal["status"], "ALERTE")
        self.assertGreaterEqual(signal["confidence"], 0.60)
        self.assertIsNotNone(
            self.db.execute("SELECT id FROM alerts WHERE signal_id = ?", (signal_id,)).fetchone()
        )

    def test_duplicate_observations_from_one_source_do_not_create_alert(self):
        data_id = self.db.execute(
            "SELECT id FROM collected_data WHERE sector_id = ? ORDER BY id LIMIT 1",
            (self.sector_id,),
        ).fetchone()["id"]
        self.db.execute(
            "INSERT INTO collected_data(source_id, sector_id, label, value, unit, published_on, "
            "collected_at, origin_url) SELECT source_id, sector_id, label, value, unit, "
            "published_on, collected_at, origin_url FROM collected_data WHERE id = ?",
            (data_id,),
        )
        second_data_id = self.db.execute("SELECT last_insert_rowid()").fetchone()[0]
        signal_id, alerted = create_signal(
            self.db,
            self.sector_id,
            "Source dupliquée",
            "Deux lignes mais une seule source",
            [data_id, second_data_id],
            "Test",
        )
        self.assertFalse(alerted)
        signal = self.db.execute(
            "SELECT independent_sources, status FROM signals WHERE id = ?", (signal_id,)
        ).fetchone()
        self.assertEqual(signal["independent_sources"], 1)
        self.assertEqual(signal["status"], "DETECTE")
        self.assertIsNone(
            self.db.execute("SELECT id FROM alerts WHERE signal_id = ?", (signal_id,)).fetchone()
        )

    def test_dashboard_counts_database_records_and_shows_collection_freshness(self):
        content = render_dashboard(self.db)
        signal_count = self.db.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        data_count = self.db.execute("SELECT COUNT(*) FROM collected_data").fetchone()[0]
        self.assertIn("Signaux enregistrés", content)
        self.assertIn(f'<strong class="stat-value">{signal_count}</strong>', content)
        self.assertIn(f"Périmètre : {data_count} données enregistrées", content)
        self.assertIn("Dernière collecte :", content)
        self.assertIn("ne sont pas actualisées en continu", content)

    def test_signal_detail_explains_score_sources_and_measurement_evolution(self):
        evidence = self.db.execute(
            "SELECT id, source_id, sector_id, label FROM collected_data "
            "WHERE sector_id = ? ORDER BY id LIMIT 2",
            (self.sector_id,),
        ).fetchall()
        self.db.execute(
            "INSERT INTO collected_data(source_id, sector_id, label, value, unit, published_on, "
            "collected_at, origin_url) VALUES (?, ?, ?, ?, 'indice', '2026-03-01', "
            "'2026-03-02T10:00:00', 'https://example.org/previous')",
            (evidence[0]["source_id"], self.sector_id, evidence[0]["label"], 100.0),
        )
        signal_id, _ = create_signal(
            self.db,
            self.sector_id,
            "Hausse de l'indice",
            "Une hausse est observée dans les données récentes.",
            [row["id"] for row in evidence],
            "Test",
        )
        content = render_signal_detail(self.db, signal_id)
        self.assertIn("Pourquoi ce signal compte", content)
        self.assertIn("Seuil du secteur", content)
        self.assertIn("fiabilité", content)
        self.assertIn("+12.4 indice", content)
        self.assertIn("Consulter la source", content)
        self.assertIn("Confirmer le signal", content)
        self.assertIn("Écarter le signal", content)

    def test_signal_review_requires_explicit_decision_and_closes_its_alert(self):
        data_ids = [
            row["id"]
            for row in self.db.execute(
                "SELECT id FROM collected_data WHERE sector_id = ? ORDER BY id LIMIT 2",
                (self.sector_id,),
            )
        ]
        signal_id, alerted = create_signal(
            self.db, self.sector_id, "Décision analyste", "À examiner", data_ids, "Test"
        )
        self.assertTrue(alerted)
        review_signal(self.db, signal_id, "CONFIRME", "Analyste")
        self.assertEqual(
            self.db.execute("SELECT status FROM signals WHERE id = ?", (signal_id,)).fetchone()[0],
            "CONFIRME",
        )
        self.assertEqual(
            self.db.execute("SELECT status FROM alerts WHERE signal_id = ?", (signal_id,)).fetchone()[0],
            "TRAITEE",
        )
        with self.assertRaisesRegex(ValueError, "déjà été examiné"):
            review_signal(self.db, signal_id, "REJETE", "Analyste")

    def test_study_requires_review_before_distribution(self):
        client_id = self.db.execute("SELECT id FROM clients LIMIT 1").fetchone()["id"]
        study_id = create_study(
            self.db,
            self.sector_id,
            client_id,
            "SECTORIELLE",
            "Étude de test",
            None,
            "Test",
        )
        figure_count = self.db.execute(
            "SELECT COUNT(*) FROM study_figures WHERE study_id = ?", (study_id,)
        ).fetchone()[0]
        self.assertGreater(figure_count, 0)
        with self.assertRaises(ValueError):
            publish_study(self.db, study_id, "Test")

        submit_study(self.db, study_id, "Test")
        validator_id = self.db.execute(
            "SELECT id FROM users WHERE role = 'VALIDATEUR' LIMIT 1"
        ).fetchone()["id"]
        review_study(self.db, study_id, validator_id, "REJETEE", "Préciser la période.")
        study = self.db.execute(
            "SELECT status, version FROM studies WHERE id = ?", (study_id,)
        ).fetchone()
        self.assertEqual(study["status"], "A_CORRIGER")
        refresh_study(self.db, study_id, "Test")
        self.assertEqual(
            self.db.execute("SELECT status FROM studies WHERE id = ?", (study_id,)).fetchone()["status"],
            "BROUILLON",
        )
        submit_study(self.db, study_id, "Test")
        review_study(self.db, study_id, validator_id, "VALIDEE", "Conforme.")
        publish_study(self.db, study_id, "Test")
        self.assertEqual(
            self.db.execute("SELECT status FROM studies WHERE id = ?", (study_id,)).fetchone()["status"],
            "DIFFUSEE",
        )
        self.assertEqual(
            self.db.execute(
                "SELECT COUNT(*) FROM distributions WHERE study_id = ?", (study_id,)
            ).fetchone()[0],
            1,
        )

    def test_sentiment_lexicon_classifies_positive_negative_and_neutral_text(self):
        self.assertEqual(analyze_sentiment("Bonne autonomie et produit fiable."), (1.0, "POSITIF"))
        self.assertEqual(analyze_sentiment("Mauvaise qualité et panne."), (-1.0, "NEGATIF"))
        self.assertEqual(analyze_sentiment("Le colis est arrivé mardi."), (0.0, "NEUTRE"))
        product_id = self.db.execute("SELECT id FROM products LIMIT 1").fetchone()["id"]
        mention_id = create_product_mention(
            self.db, product_id, "Blog", "Avis neutre", "https://example.org/neutral",
            "Le colis est arrivé mardi.", "2026-10-20",
        )
        mentions = self.db.execute(
            "SELECT * FROM product_mentions WHERE id = ?", (mention_id,)
        ).fetchall()
        self.assertEqual(sentiment_summary(mentions)["label"], "Perception neutre")

    def test_llm_request_uses_configured_bearer_key(self):
        response = io.BytesIO(
            json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"}}]}).encode()
        )
        with patch.dict(os.environ, {
            "LLM_API_KEY": "test-key-not-real",
            "LLM_API_BASE_URL": (
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            ),
            "LLM_MODEL": "gemini-3.8-flash",
        }):
            with patch("urllib.request.urlopen", return_value=response) as mocked:
                self.assertEqual(call_llm_json("system", "user"), {"ok": True})
        request = mocked.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        )
        self.assertEqual(json.loads(request.data)["model"], "gemini-3.8-flash")
        self.assertEqual(
            mocked.call_args.args[0].get_header("Authorization"),
            "Bearer test-key-not-real",
        )

    @patch(
        "app.call_llm_json",
        return_value={
            "score": 0.72,
            "label": "POSITIF",
            "strengths": ["autonomie", "robustesse"],
            "weaknesses": ["prix élevé"],
        },
    )
    def test_llm_sentiment_is_validated_and_strengths_are_saved(self, _mock_llm):
        analysis = analyze_sentiment_with_llm("Avis détaillé sur le produit.")
        self.assertEqual(analysis["source"], "LLM")
        product_id = self.db.execute("SELECT id FROM products LIMIT 1").fetchone()["id"]
        mention_id = create_product_mention(
            self.db,
            product_id,
            "Blog",
            "Avis détaillé",
            "https://example.org/review/llm",
            "Avis détaillé sur le produit.",
            "2026-10-20",
            analysis,
        )
        mention = self.db.execute(
            "SELECT sentiment_score, sentiment_label, sentiment_strengths, "
            "sentiment_weaknesses FROM product_mentions WHERE id = ?",
            (mention_id,),
        ).fetchone()
        self.assertEqual(mention["sentiment_score"], 0.72)
        self.assertEqual(mention["sentiment_label"], "POSITIF")
        self.assertIn("autonomie", mention["sentiment_strengths"])
        self.assertIn("prix élevé", mention["sentiment_weaknesses"])

    @patch("app.call_llm_json", side_effect=LLMError("API indisponible"))
    def test_llm_sentiment_falls_back_to_lexicon(self, _mock_llm):
        analysis = analyze_sentiment_with_llm("Très bon produit et fiable.")
        self.assertEqual(analysis["source"], "lexique local")
        self.assertEqual(analysis["label"], "POSITIF")
        self.assertTrue(analysis["error"])
        self.assertTrue(analysis["strengths"])

    @patch(
        "app.call_llm_json",
        return_value={"label": "Taux d'adoption", "value": 42.5, "unit": "%", "sector_id": 1},
    )
    def test_article_extraction_returns_validated_observation(self, _mock_llm):
        sectors = self.db.execute("SELECT id, name FROM sectors").fetchall()
        result = extract_data_with_llm("Le taux d'adoption atteint 42,5 %.", sectors)
        self.assertEqual(result["label"], "Taux d'adoption")
        self.assertEqual(result["value"], 42.5)
        self.assertEqual(result["unit"], "%")
        self.assertIn(result["sector_id"], {sector["id"] for sector in sectors})

    @patch(
        "app.call_llm_json",
        return_value={"label": "Mesure", "value": 2, "unit": "x", "sector_id": 999999},
    )
    def test_article_extraction_rejects_unknown_sector(self, _mock_llm):
        with self.assertRaisesRegex(LLMError, "secteur est inconnu"):
            extract_data_with_llm("Mesure 2 x.", self.db.execute(
                "SELECT id, name FROM sectors"
            ).fetchall())

    def test_product_mention_keeps_source_and_collection_provenance(self):
        product = self.db.execute("SELECT id FROM products LIMIT 1").fetchone()
        mention_id = create_product_mention(
            self.db,
            product["id"],
            "Forum",
            "Avis vérifié",
            "https://example.org/review/123",
            "Bonne autonomie et produit fiable.",
            "2026-10-20",
        )
        mention = self.db.execute(
            "SELECT platform, source_name, source_url, published_on, collected_at, "
            "sentiment_label FROM product_mentions WHERE id = ?",
            (mention_id,),
        ).fetchone()
        self.assertEqual(mention["platform"], "Forum")
        self.assertEqual(mention["source_name"], "Avis vérifié")
        self.assertEqual(mention["source_url"], "https://example.org/review/123")
        self.assertEqual(mention["published_on"], "2026-10-20")
        self.assertTrue(mention["collected_at"])
        self.assertEqual(mention["sentiment_label"], "POSITIF")

    def test_product_mention_rejects_duplicate_source_for_same_product(self):
        product_id = self.db.execute("SELECT id FROM products LIMIT 1").fetchone()["id"]
        mention = (
            product_id, "Blog", "Site d'avis", "https://example.org/review/duplicate",
            "Très bon produit.", "2026-10-20",
        )
        create_product_mention(self.db, *mention)
        with self.assertRaisesRegex(ValueError, "déjà été enregistrée"):
            create_product_mention(self.db, *mention)

    def test_reputation_view_compares_selected_product_and_shows_source(self):
        product = self.db.execute(
            "SELECT id, name FROM products ORDER BY id LIMIT 1"
        ).fetchone()
        create_product_mention(
            self.db,
            product["id"],
            "YouTube",
            "Chaîne de test",
            "https://example.org/video/1",
            "Excellent et fiable, je recommande.",
            "2026-10-20",
        )
        content = render_reputation(self.db, [product["id"]])
        self.assertIn(product["name"], content)
        self.assertIn("Chaîne de test", content)
        self.assertIn("https://example.org/video/1", content)
        other_products = self.db.execute(
            "SELECT name FROM products WHERE id != ? LIMIT 1", (product["id"],)
        ).fetchone()
        self.assertNotIn(f"<h3>{other_products['name']}</h3>", content)

    def test_database_initialization_is_idempotent_for_reputation_tables(self):
        initialize_database(self.database)
        initialize_database(self.database)
        product_count = self.db.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        self.assertEqual(product_count, 3)


if __name__ == "__main__":
    unittest.main()
