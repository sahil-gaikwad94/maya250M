"""WP-04 tests: hybrid retrieval fusion and non-starving repository packing."""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.codebase import (
    HybridRepoRetriever,
    RepoGraphIndexer,
    TopologicalRepoPacker,
    bm25_scores,
    reciprocal_rank_fusion,
    tokenize,
)

REPO_FILES = {
    "lib/__init__.py": "",
    "lib/util.py": textwrap.dedent(
        '''
        """Path and digest helpers."""


        def normalize_path(raw: str) -> str:
            """Drop duplicate separators and any trailing slash."""
            return raw.replace("//", "/").rstrip("/")


        def checksum(data: bytes) -> str:
            """Stable digest used by the cache layer."""
            return str(len(data))
        '''
    ),
    "lib/registry.py": textwrap.dedent(
        '''
        """Name registry."""


        class Registry:
            """Registers callables by name."""

            def register(self, name: str, value) -> None:
                self.names = getattr(self, "names", {})
                self.names[name] = value
        '''
    ),
    "app/main.py": textwrap.dedent(
        '''
        from lib.util import normalize_path


        def handle_request(path: str) -> str:
            return normalize_path(path)
        '''
    ),
}


class RetrievalFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = cls._tmp.name
        for rel, content in REPO_FILES.items():
            full = os.path.join(cls.root, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as handle:
                handle.write(content)
        cls.indexer = RepoGraphIndexer(cls.root)
        cls.indexer.index()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class TestTokenizationAndBM25(unittest.TestCase):
    def test_identifier_sub_tokens(self):
        self.assertEqual(tokenize("normalize_path"), ["normalize", "path"])
        self.assertEqual(tokenize("HTTPServer"), ["http", "server"])
        self.assertIn("handle", tokenize("handle_request(path)"))

    def test_bm25_prefers_the_document_with_the_term(self):
        docs = ["normalize path helper", "checksum digest", "unrelated words here"]
        scores = bm25_scores(tokenize("checksum"), docs)
        self.assertEqual(scores.index(max(scores)), 1)
        self.assertEqual(scores[0], 0.0)


class TestReciprocalRankFusion(unittest.TestCase):
    def test_agreement_beats_a_single_first_place(self):
        first_only = reciprocal_rank_fusion([["a", "b"], ["c"]], k=60)
        consistent = reciprocal_rank_fusion([["c", "b"], ["b", "c"]], k=60)
        # "b" is 2nd in both lists; "a" is 1st in one list and absent from the other.
        self.assertGreater(consistent["b"], first_only["a"])
        self.assertGreater(first_only["a"], 0.0)

    def test_duplicate_entries_count_once(self):
        fused = reciprocal_rank_fusion([["a", "a", "b"]], k=60)
        self.assertAlmostEqual(fused["a"], 1.0 / 61)
        self.assertAlmostEqual(fused["b"], 1.0 / 62)


class TestHybridRepoRetriever(RetrievalFixture):
    def test_lexical_signal_finds_the_named_symbol(self):
        retriever = HybridRepoRetriever(self.indexer)
        ranking = retriever.lexical_ranking("normalize_path trailing slash")
        self.assertTrue(ranking)
        names = [retriever._symbols[i].qualname for i in ranking[:3]]
        self.assertIn("normalize_path", names)

    def test_structural_signal_follows_the_query_identifiers(self):
        retriever = HybridRepoRetriever(self.indexer)
        ranking = retriever.structural_ranking("checksum is returning the wrong digest")
        names = [retriever._symbols[i].qualname for i in ranking[:5]]
        self.assertIn("checksum", names)

    def test_dense_signal_absent_means_not_run(self):
        retriever = HybridRepoRetriever(self.indexer)
        result = retriever.search("normalize_path", k=5)
        self.assertFalse(result.dense_ranked)
        self.assertEqual(set(result.signals), {"lexical", "structural"})
        self.assertEqual(retriever.dense_ranking("normalize_path"), None)

    def test_dense_ranker_is_fused_when_supplied(self):
        def dense(query, documents):
            # Pretend an embedder put every document last except the registry one.
            for index, document in enumerate(documents):
                if "Registry" in document:
                    return [index]
            return []

        retriever = HybridRepoRetriever(self.indexer, dense_ranker=dense)
        result = retriever.search("registry of names", k=6)
        self.assertTrue(result.dense_ranked)
        self.assertIn("dense", result.signals)
        self.assertIn("Registry", [c.qualname for c in result.candidates])
        registry = next(c for c in result.candidates if c.qualname == "Registry")
        self.assertEqual(registry.ranks.get("dense"), 1)

    def test_candidates_report_per_signal_ranks(self):
        retriever = HybridRepoRetriever(self.indexer)
        result = retriever.search("normalize_path", k=4)
        self.assertTrue(result.candidates)
        for candidate in result.candidates:
            self.assertTrue(candidate.ranks)
            self.assertGreater(candidate.score, 0.0)
            self.assertEqual(candidate.symbol_id, f"{candidate.file_path}::{candidate.qualname}")
        scores = [c.score for c in result.candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rendered_context_respects_budget(self):
        retriever = HybridRepoRetriever(self.indexer)
        context = retriever.render_context("normalize_path trailing slash", token_budget=60)
        self.assertLessEqual(context.tokens_used, 60)
        self.assertIn("lib/util.py", context.text)
        self.assertIn("normalize_path", context.text)

        tiny = retriever.render_context("normalize_path", token_budget=5)
        self.assertLessEqual(tiny.tokens_used, 5)
        self.assertTrue(tiny.truncated)

    def test_context_reports_dense_absence(self):
        retriever = HybridRepoRetriever(self.indexer)
        context = retriever.render_context("checksum", token_budget=200)
        self.assertFalse(context.dense_ranked)


class TestPackerBudgetAccounting(RetrievalFixture):
    def _packer(self) -> TopologicalRepoPacker:
        return TopologicalRepoPacker(self.indexer)

    def test_one_huge_file_does_not_starve_smaller_ones(self):
        packer = self._packer()
        huge = "\n".join(f"value_{i} = {i}" for i in range(2000)) + "\n"
        contents = {
            "huge.py": huge,
            "tiny_a.py": "A = 1\n",
            "tiny_b.py": "B = 2\n",
        }
        packed = packer.pack(
            contents=contents,
            order=["huge.py", "tiny_a.py", "tiny_b.py"],
            token_budget=80,
        )
        self.assertIn("tiny_a.py", packed.order)
        self.assertIn("tiny_b.py", packed.order)
        self.assertLessEqual(packed.tokens_used, 80)
        self.assertTrue(packed.truncated)

    def test_single_oversized_file_is_trimmed_not_dropped(self):
        packer = self._packer()
        huge = "\n".join(f"value_{i} = {i}" for i in range(2000)) + "\n"
        packed = packer.pack(contents={"huge.py": huge}, order=["huge.py"], token_budget=120)
        self.assertEqual(packed.order, ["huge.py"])
        self.assertIn("# <|file_sep|> huge.py", packed.text)
        self.assertIn("[truncated]", packed.text)
        self.assertLessEqual(packed.tokens_used, 120)
        self.assertTrue(packed.truncated)
        self.assertEqual(packed.skipped, [])

    def test_unusable_file_is_reported_as_skipped(self):
        packer = self._packer()
        huge = "\n".join(f"value_{i} = {i}" for i in range(500)) + "\n"
        packed = packer.pack(contents={"huge.py": huge}, order=["huge.py"], token_budget=4)
        self.assertEqual(packed.order, [])
        self.assertEqual(packed.skipped, ["huge.py"])
        self.assertTrue(packed.truncated)
        self.assertLessEqual(packed.tokens_used, 4)

    def test_no_budget_still_packs_everything_in_order(self):
        packer = self._packer()
        packed = packer.pack(order=["lib/util.py", "app/main.py"])
        self.assertEqual(packed.order, ["lib/util.py", "app/main.py"])
        self.assertFalse(packed.truncated)
        self.assertEqual(packed.skipped, [])
        self.assertEqual(len(packed.order), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
