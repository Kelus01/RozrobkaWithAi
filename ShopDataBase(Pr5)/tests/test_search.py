import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app import documents, embeddings, index, keyword
from app.documents import Chunk
from app.main import app


class DocumentTests(unittest.TestCase):
    def test_loads_all_documents_and_metadata(self):
        loaded = documents.load_documents()

        self.assertEqual(len(loaded), 16)
        operator_document = next(
            item for item in loaded if item[0] == "внутрішня-інструкція-оператора.md"
        )
        self.assertEqual(operator_document[1]["audience"], "персонал")
        self.assertEqual(operator_document[1]["status"], "чинний")

    def test_split_keeps_section_context_and_metadata(self):
        chunks = documents.split(
            "Вступний текст.\n\n## Заряджання\n\nПеревірте зарядний кабель.",
            "device.md",
            {"title": "Пристрій", "category": "інструкція"},
        )

        self.assertEqual(len(chunks), 2)
        self.assertIn("Пристрій\nПристрій", chunks[0].text)
        self.assertIn("Заряджання", chunks[1].text)
        self.assertEqual(chunks[1].metadata["heading"], "Заряджання")
        self.assertEqual(chunks[1].source, "device.md")


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("Навушники не заряджаються", "headphones.md", {
                "category": "інструкція", "product": "Оріон X2",
                "audience": "клієнти", "status": "чинний",
            }),
            Chunk("Повернення коштів за товар", "returns.md", {
                "category": "правила", "product": "",
                "audience": "клієнти", "status": "чинний",
            }),
            Chunk("Внутрішні ліміти бонусів", "operator.md", {
                "category": "внутрішнє", "product": "",
                "audience": "персонал", "status": "чинний",
            }),
        ]
        self.vectors = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
        self.index = index.build(self.chunks, self.vectors, "test-model")

    def test_search_ranks_by_cosine_and_applies_metadata_filters(self):
        results = index.search(
            self.index,
            np.array([1, 0], dtype=np.float32),
            top_k=2,
            filters={"audience": "клієнти"},
        )

        self.assertEqual([hit.chunk.source for hit in results], ["headphones.md", "returns.md"])
        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertAlmostEqual(results[1].score, 0.0)

    def test_threshold_filters_weak_matches_and_unknown_filter_is_rejected(self):
        results = index.search(
            self.index,
            np.array([1, 0], dtype=np.float32),
            threshold=0.1,
        )
        self.assertEqual([hit.chunk.source for hit in results], ["headphones.md"])
        with self.assertRaises(ValueError):
            index.search(self.index, np.array([1, 0]), filters={"unknown": "value"})

    def test_save_and_load_preserves_vectors_chunks_and_model_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index"
            index.save(self.index, path)
            loaded = index.load(path)

        self.assertEqual(loaded.model_name, "test-model")
        self.assertEqual([chunk.source for chunk in loaded.chunks], [
            chunk.source for chunk in self.chunks
        ])
        np.testing.assert_allclose(loaded.vectors, self.index.vectors)

    def test_load_missing_index_has_actionable_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(index.IndexNotFoundError):
                index.load(Path(directory) / "missing")


class EmbeddingTests(unittest.TestCase):
    def test_e5_queries_and_passages_use_distinct_prefixes(self):
        model = Mock()
        model.encode.side_effect = [
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 1.0]], dtype=np.float32),
        ]

        with patch.object(embeddings, "get_model", return_value=model):
            embeddings.embed_passages(["текст документа"])
            embeddings.embed_query("пошукове питання")

        calls = model.encode.call_args_list
        self.assertEqual(calls[0].args[0], ["passage: текст документа"])
        self.assertEqual(calls[1].args[0], ["query: пошукове питання"])
        self.assertTrue(calls[0].kwargs["normalize_embeddings"])


class KeywordTests(unittest.TestCase):
    def test_tokenizer_preserves_article_codes_and_is_case_insensitive(self):
        self.assertEqual(keyword.tokenize("OR-X2-BLK or-x2-blk"), ["or-x2-blk", "or-x2-blk"])

    def test_bm25_returns_shared_hit_format_and_respects_filters(self):
        chunks = [
            Chunk("Артикул навушників OR-X2-BLK", "headphones.md", {"product": "Оріон X2"}),
            Chunk("Інструкція з повернення товару", "returns.md", {"product": ""}),
        ]
        built = keyword.build(chunks)
        hits = keyword.search(built, "OR-X2-BLK", filters={"product": "Оріон X2"})

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.source, "headphones.md")
        self.assertGreaterEqual(hits[0].score, 0)


class APITests(unittest.TestCase):
    def test_status_and_missing_index_response(self):
        with patch.object(
            index,
            "load",
            side_effect=index.IndexNotFoundError("Індекс відсутній; виконайте python ingest.py."),
        ):
            with TestClient(app) as client:
                status = client.get("/api/status")
                response = client.post("/api/search", json={"query": "доставка"})

        self.assertFalse(status.json()["ready"])
        self.assertEqual(response.status_code, 503)
        self.assertIn("ingest.py", response.json()["detail"])

    def test_search_endpoint_returns_results_and_separate_timings(self):
        chunks = [
            Chunk("Налаштування роутера", "router.md", {
                "title": "Роутер", "category": "інструкція",
                "product": "Альтаїр AX3", "audience": "клієнти",
                "updated": "2026-01-01", "status": "чинний",
            })
        ]

        with TestClient(app) as client:
            app.state.search_index = index.build(
                chunks, np.array([[1.0, 0.0]]), embeddings.MODEL_NAME
            )
            app.state.keyword_index = keyword.build(chunks)
            with patch.object(
                embeddings, "embed_query", return_value=np.array([1.0, 0.0])
            ):
                response = client.post("/api/search", json={
                    "query": "налаштування", "mode": "both",
                    "filters": {"product": "Альтаїр AX3"},
                })

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["semantic"][0]["source"], "router.md")
        self.assertEqual(result["keyword"][0]["source"], "router.md")
        self.assertIn("semantic", result["elapsed"])
        self.assertIn("keyword", result["elapsed"])

    def test_search_endpoint_validates_mode_and_empty_query(self):
        with TestClient(app) as client:
            invalid_mode = client.post("/api/search", json={
                "query": "test", "mode": "unknown",
            })
            empty_query = client.post("/api/search", json={
                "query": "   ",
            })

        self.assertEqual(invalid_mode.status_code, 422)
        self.assertEqual(empty_query.status_code, 422)


if __name__ == "__main__":
    unittest.main()
