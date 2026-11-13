"""Phase 6 tests: provenance, dedup, mixtures, contamination (v3 §8)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.training import (
    DataRecord,
    Provenance,
    ProvenanceError,
    build_record,
    dedup_records,
    near_duplicate_groups,
    leakage_report,
    assert_no_group_leakage,
    LeakageError,
    corpus_hash,
    DataMixer,
    MixtureSpec,
    MIXTURES,
)
from maya250m.training.dedup import normalize_text, code_clone_key
from maya250m.evaluation import (
    ContaminationReport,
    contamination_scan,
    assert_no_contamination,
    ContaminationError,
    protect_heldout,
    ngram_overlap,
    exact_containment,
)


def _prov(**kw):
    base = dict(
        source_id="src1",
        license="mit",
        domain="code",
        split_group="repoA",
        quality_score=0.5,
    )
    base.update(kw)
    return base


def _rec(text, **kw):
    return build_record(text, **_prov(**kw))


class TestProvenance(unittest.TestCase):
    def test_missing_split_group_raises(self):
        with self.assertRaises(ProvenanceError):
            build_record("hello world", source_id="s", license="mit", domain="code", split_group="")

    def test_unknown_domain_raises(self):
        with self.assertRaises(ProvenanceError):
            build_record("hello world", source_id="s", license="mit", domain="bogus", split_group="g")

    def test_unknown_license_raises(self):
        with self.assertRaises(ProvenanceError):
            build_record("hello world", source_id="s", license="nope", domain="code", split_group="g")

    def test_synthetic_requires_generator(self):
        with self.assertRaises(ProvenanceError):
            build_record(
                "hello world",
                source_id="s",
                license="mit",
                domain="code",
                split_group="g",
                is_synthetic=True,
            )

    def test_synthetic_ok_with_generator(self):
        r = build_record(
            "hello world",
            source_id="s",
            license="mit",
            domain="code",
            split_group="g",
            is_synthetic=True,
            generator_version="v1",
        )
        self.assertTrue(r.provenance.is_synthetic)


class TestDedup(unittest.TestCase):
    def test_exact_duplicate_dropped(self):
        r1 = _rec("def f(): return 1", record_id="a")
        r2 = _rec("def f(): return 1", record_id="b")
        report = dedup_records([r1, r2])
        self.assertEqual(len(report.kept), 1)
        self.assertEqual(len(report.exact_duplicates), 1)

    def test_normalized_duplicate_dropped(self):
        r1 = _rec("Hello, World!", record_id="a", domain="general")
        r2 = _rec("  hello   world  ", record_id="b", domain="general")
        report = dedup_records([r1, r2])
        self.assertEqual(len(report.kept), 1)

    def test_code_clone_key_ignores_comments(self):
        a = "def f():\n    # comment\n    return 1\n"
        b = "def f():\n    return 1\n"
        self.assertEqual(code_clone_key(a), code_clone_key(b))

    def test_code_clone_keeps_identifiers(self):
        a = "def f():\n    x = 1\n    return x\n"
        b = "def f():\n    y = 1\n    return y\n"
        self.assertNotEqual(code_clone_key(a), code_clone_key(b))

    def test_normalize_text(self):
        self.assertEqual(normalize_text("A,  B!!"), "a b")

    def test_near_duplicate_groups(self):
        base = "the quick brown fox jumps over the lazy dog " * 3
        r1 = _rec(base, record_id="a", domain="general", split_group="g1")
        r2 = _rec(base.replace("quick", "fast"), record_id="b", domain="general", split_group="g2")
        groups = near_duplicate_groups([r1, r2], threshold=0.4)
        self.assertTrue(len(groups) >= 1)

    def test_leakage_report_clean(self):
        r1 = _rec("a", record_id="a", split_group="g1", split="train", domain="general")
        r2 = _rec("b", record_id="b", split_group="g2", split="heldout", domain="general")
        report = leakage_report([r1, r2])
        self.assertTrue(report["clean"])

    def test_leakage_report_dirty(self):
        r1 = _rec("a", record_id="a", split_group="g1", split="train", domain="general")
        r2 = _rec("b", record_id="b", split_group="g1", split="heldout", domain="general")
        with self.assertRaises(LeakageError):
            assert_no_group_leakage([r1, r2])

    def test_corpus_hash_deterministic(self):
        r1 = _rec("a", record_id="a", domain="general", split_group="g1")
        h1 = corpus_hash([r1])
        h2 = corpus_hash([r1])
        self.assertEqual(h1, h2)


class TestMixer(unittest.TestCase):
    def setUp(self):
        recs = []
        for i in range(20):
            recs.append(_rec(f"code sample {i}", record_id=f"c{i}", domain="code", split_group=f"cg{i}"))
        for i in range(20):
            recs.append(_rec(f"general sample {i}", record_id=f"g{i}", domain="general", split_group=f"gg{i}"))
        self.mixer = DataMixer(recs, seed=42, max_per_source=10)

    def test_mixture_weights_sum_to_one(self):
        for name, w in MIXTURES.items():
            self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
            MixtureSpec(name=name, weights=dict(w))

    def test_mix_counts(self):
        report = self.mixer.mix("balanced", 10)
        self.assertEqual(report.requested, 10)
        self.assertLessEqual(len(report.records), 10)
        self.assertIn("counts", report.manifest())

    def test_determinism(self):
        r1 = self.mixer.mix("balanced", 10)
        r2 = self.mixer.mix("balanced", 10)
        self.assertEqual([r.record_id for r in r1.records], [r.record_id for r in r2.records])

    def test_shortfall_reported(self):
        # request more than available
        m = DataMixer([_rec("x", record_id="x", domain="code", split_group="g")], seed=1)
        report = m.mix("code_forward", 50)
        self.assertTrue(report.shortfall)

    def test_compare_no_winner(self):
        out = self.mixer.compare(10)
        self.assertIsNone(out["winner"])

    def test_invalid_mixture(self):
        with self.assertRaises(KeyError):
            self.mixer.mix("nope", 5)


class TestContamination(unittest.TestCase):
    def test_clean_corpus(self):
        train = [_rec("completely unrelated text", record_id="t1", domain="general", split_group="g1")]
        held = [_rec("another different document", record_id="h1", domain="general", split_group="g2")]
        report = contamination_scan(train, held)
        self.assertFalse(report.contaminated)
        self.assertTrue(report.clean)

    def test_exact_containment_flagged(self):
        train = [_rec("the secret benchmark answer is 42", record_id="t1", domain="general", split_group="g1")]
        held = [_rec("the secret benchmark answer is 42", record_id="h1", domain="general", split_group="g2")]
        report = contamination_scan(train, held)
        self.assertTrue(report.contaminated)
        self.assertTrue(report.exact_matches)

    def test_ngram_overlap_flagged(self):
        long_text = " ".join(["word"] * 20)
        train = [_rec(long_text + " train", record_id="t1", domain="general", split_group="g1")]
        held = [_rec(long_text + " held", record_id="h1", domain="general", split_group="g2")]
        report = contamination_scan(train, held)
        self.assertTrue(report.contaminated)

    def test_assert_no_contamination_raises(self):
        train = [_rec("same text here", record_id="t1", domain="general", split_group="g1")]
        held = [_rec("same text here", record_id="h1", domain="general", split_group="g2")]
        with self.assertRaises(ContaminationError):
            assert_no_contamination(train, held)

    def test_assert_no_contamination_clean(self):
        train = [_rec("alpha beta gamma", record_id="t1", domain="general", split_group="g1")]
        held = [_rec("delta epsilon zeta", record_id="h1", domain="general", split_group="g2")]
        report = assert_no_contamination(train, held)
        self.assertIsInstance(report, ContaminationReport)

    def test_protect_heldout_removes(self):
        train = [_rec("same text here", record_id="t1", domain="general", split_group="g1")]
        held = [_rec("same text here", record_id="h1", domain="general", split_group="g2")]
        cleaned = protect_heldout(train, held)
        self.assertEqual(len(cleaned), 0)
        self.assertIsNotNone(cleaned.report)
        self.assertTrue(cleaned.report.contaminated)

    def test_benchmark_contamination(self):
        train = [_rec("leaked benchmark item", record_id="t1", domain="general", split_group="g1")]
        bench = [_rec("leaked benchmark item", record_id="b1", domain="general", split_group="g3")]
        report = contamination_scan(train, [], bench)
        self.assertTrue(report.contaminated)
        self.assertTrue(report.contaminated_benchmark_ids)

    def test_ngram_overlap_helper(self):
        r = ngram_overlap("a b c d e f", "a b c d e g", n=3)
        self.assertGreater(r["shared_count"], 0)

    def test_exact_containment_helper(self):
        self.assertTrue(exact_containment("hello world", "world"))
        self.assertFalse(exact_containment("abc", "xyz"))

    def test_strings_input(self):
        report = contamination_scan(["same text"], ["same text"])
        self.assertTrue(report.contaminated)


if __name__ == "__main__":
    unittest.main(verbosity=2)
