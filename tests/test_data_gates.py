"""WP-06 tests: the runner must not train a flagship preset on fixture data."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_maya250m import SYNTHETIC_CORPUS_ID, resolve_corpus


class TestCorpusGate(unittest.TestCase):
    def test_synthetic_fixture_is_labelled_for_small_presets(self):
        documents, provenance = resolve_corpus(None, "MAYA_TINY_TEST")
        self.assertEqual(provenance["source"], "synthetic")
        self.assertTrue(provenance["synthetic"])
        self.assertEqual(provenance["id"], SYNTHETIC_CORPUS_ID)
        self.assertEqual(provenance["documents"], len(documents))
        self.assertEqual(len(str(provenance["sha256"])), 64)

    def test_flagship_preset_refuses_the_fixture(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_corpus(None, "MAYA_250M")
        message = str(ctx.exception)
        self.assertIn("--corpus", message)
        self.assertIn("--allow-synthetic-at-scale", message)

    def test_flagship_can_opt_in_to_a_plumbing_run(self):
        documents, provenance = resolve_corpus(
            None, "MAYA_250M", allow_synthetic_at_scale=True
        )
        self.assertTrue(provenance["synthetic"])
        self.assertTrue(documents)

    def test_real_corpus_is_hashed_and_marked_non_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "corpus.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("alpha\n\nbeta\n")
            documents, provenance = resolve_corpus(path, "MAYA_250M")
        self.assertEqual(documents, ["alpha", "beta"])
        self.assertFalse(provenance["synthetic"])
        self.assertEqual(provenance["source"], "file")
        self.assertEqual(provenance["documents"], 2)
        self.assertEqual(len(str(provenance["sha256"])), 64)

    def test_empty_corpus_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n\n   \n")
            with self.assertRaises(ValueError):
                resolve_corpus(path, "MAYA_TINY_TEST")

    def test_provenance_is_json_serialisable(self):
        import json

        _, provenance = resolve_corpus(None, "MAYA_TINY_TEST")
        self.assertEqual(json.loads(json.dumps(provenance))["source"], "synthetic")


if __name__ == "__main__":
    unittest.main(verbosity=2)
