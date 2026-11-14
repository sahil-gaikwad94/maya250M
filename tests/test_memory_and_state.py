"""Phase 1-2 tests: TaskSpec, ReasoningState, adaptive depth, memory + retrieval."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.memory import (
    EvidenceContextBuilder,
    HashingEmbedder,
    HybridRetriever,
    MayaMemoryStore,
    MemoryConsolidator,
    MemoryProjection,
    MemoryRecord,
    SemanticTriple,
    SharedTrunkNeuralReranker,
    VectorIndex,
    WorkingMemoryState,
    memory_projection_parameters,
)
from maya250m.reasoning import (
    PHASE_DONE,
    PHASE_HYPOTHESIZE,
    PHASE_REVIEW,
    PHASE_UNDERSTAND,
    PHASE_VERIFY,
    AdaptiveLoopController,
    CorrectionPolicy,
    Decomposition,
    DecompositionError,
    ReasoningState,
    ReasoningStateError,
    StrategySelector,
    Subproblem,
    decompose_task,
    diagnose_failure,
)
from maya250m.understanding import (
    TaskSpec,
    lexical_requirement_coverage,
    parse_task_spec,
    validate_task_spec,
)
from maya250m.verification import VerificationResult, failure, ok


class TestTaskSpec(unittest.TestCase):
    def test_parse_bug_report(self):
        prompt = (
            "Fix the crash in maya250m/train.py when grad_accum_steps > 1. "
            "You must preserve the public API and do not add new dependencies. "
            "Add a regression test in tests/test_train.py."
        )
        spec = parse_task_spec(
            prompt,
            repo_files=["maya250m/train.py", "tests/test_train.py"],
            repo_symbols=["train_maya", "grad_accum_steps"],
        )
        self.assertEqual(validate_task_spec(spec), [])
        self.assertEqual(spec.task_type, "bug_fix")
        self.assertIn("maya250m/train.py", spec.files_mentioned)
        self.assertTrue(spec.constraints)
        self.assertGreater(spec.confidence, 0.5)
        self.assertIn("OBJECTIVE:", spec.prompt_block())

    def test_ambiguous_prompt_is_flagged(self):
        spec = parse_task_spec("make it better somehow")
        self.assertIn("underspecified_criterion", spec.ambiguity_flags)
        self.assertEqual(StrategySelector().select(spec), "abstain")

    def test_schema_rejects_bad_values(self):
        spec = TaskSpec(objective="x", task_type="not_a_type")
        self.assertIn("unknown_task_type:not_a_type", validate_task_spec(spec))
        with self.assertRaises(ValueError):
            parse_task_spec("   ")

    def test_lexical_requirement_coverage_detects_missing_constraints(self):
        spec = parse_task_spec(
            "Fix the bug. You must preserve the public API and must not add new dependencies."
        )
        coverage = lexical_requirement_coverage(spec, "The public API is preserved; tests pass.")
        self.assertEqual(len(coverage), len(spec.constraints))
        self.assertTrue(any(coverage.values()))
        self.assertTrue(any(not v for v in coverage.values()))


class TestReasoningState(unittest.TestCase):
    def _state(self) -> ReasoningState:
        return ReasoningState(
            objective="fix train.py",
            constraints=["preserve public API"],
            max_steps=4,
            subgoals=["localize", "patch"],
        )

    def test_phase_graph_enforced(self):
        state = self._state()
        self.assertEqual(state.phase, PHASE_UNDERSTAND)
        state.transition("REPRODUCE_OR_INSPECT")
        state.transition(PHASE_HYPOTHESIZE)
        state.transition(PHASE_VERIFY)
        with self.assertRaises(ReasoningStateError):
            state.transition(PHASE_UNDERSTAND)  # no backward edge into understanding

    def test_failed_verification_returns_to_hypothesis(self):
        state = self._state()
        state.transition("REPRODUCE_OR_INSPECT")
        state.transition(PHASE_HYPOTHESIZE)
        state.transition(PHASE_VERIFY)
        state.record_hypothesis("change arity in train_maya")
        state.record_verification(failure("tests_failed", "assert 3 != 2"))
        self.assertEqual(state.phase, PHASE_HYPOTHESIZE)
        self.assertEqual(state.last_failure_category, "tests_failed")
        self.assertTrue(state.rejected_approaches)

        state.transition(PHASE_VERIFY)
        state.record_hypothesis("fix arity again")
        state.record_verification(ok("all tests pass"))
        self.assertEqual(state.phase, PHASE_REVIEW)
        state.finalize(True)
        self.assertTrue(state.finished)
        self.assertEqual(state.phase, PHASE_DONE)

    def test_budget_and_serialization(self):
        state = self._state()
        for _ in range(4):
            state.consume_step("tool")
        self.assertTrue(state.budget_exhausted)
        with self.assertRaises(ReasoningStateError):
            state.consume_step("over budget")

        restored = ReasoningState.from_dict(state.to_dict())
        self.assertEqual(restored.to_dict(), state.to_dict())

    def test_pinned_context_and_constraint_status(self):
        state = self._state()
        state.pin_fact("train_maya: min_arity=3")
        state.record_constraint_coverage({"constraint_0": False})
        text = state.pinned_context()
        self.assertIn("UNCOVERED", text)
        self.assertIn("train_maya", text)


class TestReasoningPlanning(unittest.TestCase):
    def test_adaptive_loop_controller_scales_with_difficulty(self):
        easy = parse_task_spec("Explain how the cache works.")
        hard = parse_task_spec(
            "Fix the crash in a.py and b.py and c.py. You must preserve the API, "
            "must not add new dependencies, and must keep runtime O(n). "
            "Add tests in tests/test_a.py and tests/test_b.py."
        )
        controller = AdaptiveLoopController()
        self.assertLess(
            controller.select_loops(spec=easy), controller.select_loops(spec=hard)
        )
        self.assertEqual(controller.select_loops(spec=easy, fixed=3), 3)
        self.assertIn("difficulty_score", controller.explain(spec=hard))

    def test_decomposition_is_a_valid_dag(self):
        spec = parse_task_spec("Fix the bug in maya250m/train.py")
        dag = decompose_task(spec)
        order = dag.topological_order()
        self.assertEqual(order[0], "localize")
        self.assertEqual(order[-1], "verify")
        self.assertTrue(dag.ready_subproblems())

        with self.assertRaises(DecompositionError):
            Decomposition(
                subproblems=[
                    Subproblem("a", "a", depends_on=["b"]),
                    Subproblem("b", "b", depends_on=["a"]),
                ]
            )

    def test_correction_policy_abstains_after_repeats(self):
        plan = diagnose_failure(failure("patch_ambiguous", "matched 3 locations"))
        self.assertEqual(plan.action, "narrow_context")

        policy = CorrectionPolicy(max_attempts=2)
        for _ in range(2):
            policy.next_plan(failure("tests_failed", "boom"))
        escalated = policy.next_plan(failure("tests_failed", "boom again"))
        self.assertEqual(escalated.action, "abstain")
        self.assertTrue(policy.should_abstain())


class TestVerificationResult(unittest.TestCase):
    def test_invariants(self):
        self.assertTrue(ok("fine"))
        result = failure("syntax_error", "bad indent")
        self.assertFalse(result.passed)
        self.assertTrue(result.is_retryable)
        with self.assertRaises(ValueError):
            failure("not_a_category", "x")
        with self.assertRaises(ValueError):
            VerificationResult(passed=True, failure_category="syntax_error")
        self.assertEqual(
            VerificationResult.from_dict(result.to_dict()).to_dict(), result.to_dict()
        )


class TestMemorySubsystem(unittest.TestCase):
    def _store_with_records(self) -> MayaMemoryStore:
        store = MayaMemoryStore(":memory:")
        store.add_memory(
            MemoryRecord(
                memory_id="",
                kind="episodic",
                content="Root cause: the recurrent core cache was shared across loops.",
                source="episode:1",
                confidence=0.9,
            )
        )
        store.add_memory(
            MemoryRecord(
                memory_id="",
                kind="semantic",
                content="Gated DeltaNet stores a 128x128 FP32 state per head.",
                source="doc:gdn",
                confidence=0.8,
            )
        )
        store.add_memory(
            MemoryRecord(
                memory_id="",
                kind="code",
                content="def prefill(self, input_ids, doc_ids=None) -> tuple: ...",
                source="maya250m/model.py",
                confidence=0.7,
            )
        )
        return store

    def test_store_search_and_stats(self):
        store = self._store_with_records()
        self.assertEqual(store.stats()["memories"], 3)
        hits = store.lex_search("recurrent core cache", limit=5)
        self.assertTrue(hits)
        self.assertIn("recurrent core cache", hits[0][0].content)
        self.assertEqual(hits[0][0].kind, "episodic")

        self.assertEqual(store.get_memory("missing"), None)
        store.close()

    def test_supersession_is_reversible(self):
        store = self._store_with_records()
        records = store.all_memories(kind="semantic")
        old = records[0]
        new = MemoryRecord(
            memory_id="semantic_new",
            kind="semantic",
            content="Gated DeltaNet stores a 128x128 BF16 state per head (updated).",
            source="doc:gdn2",
        )
        store.add_memory(new)
        self.assertTrue(store.supersede(old.memory_id, new.memory_id))
        self.assertTrue(store.get_memory(old.memory_id).is_superseded)
        self.assertTrue(store.restore_supersession(old.memory_id))
        self.assertFalse(store.get_memory(old.memory_id).is_superseded)
        store.close()

    def test_triples_and_consolidation(self):
        store = self._store_with_records()
        keep = MemoryRecord(memory_id="sem_a", kind="semantic", content="A fact", source="doc")
        store.add_memory(keep)
        store.add_triple(
            SemanticTriple("prefill", "returns", "tuple", source_id=keep.content, confidence=0.9, created_at=100.0)
        )
        store.add_triple(
            SemanticTriple("prefill", "returns", "dict", source_id=keep.content, confidence=0.4, created_at=200.0)
        )
        consolidator = MemoryConsolidator(store)
        report = consolidator.consolidate_triples()
        self.assertEqual(report.triples_superseded, 1)
        self.assertTrue(report.conflicts)
        active = store.query_triples(subject="prefill")
        self.assertEqual([t.obj for t in active], ["dict"])

        self.assertEqual(consolidator.undo(report), 1)
        self.assertEqual(len(store.query_triples(subject="prefill")), 2)
        store.close()

    def test_dedupe_merges_identical_content(self):
        store = MayaMemoryStore(":memory:")
        store.add_memory(MemoryRecord("m1", "semantic", "Same fact", created_at=1.0))
        store.add_memory(MemoryRecord("m2", "semantic", "Same   fact", created_at=2.0))
        report = MemoryConsolidator(store).dedupe()
        self.assertEqual(report.duplicates_merged, 1)
        self.assertTrue(store.get_memory("m2").is_superseded)
        self.assertFalse(store.get_memory("m1").is_superseded)
        store.close()

    def test_vector_index_and_hybrid_retrieval(self):
        store = self._store_with_records()
        embedder = HashingEmbedder(dim=128)
        index = VectorIndex(dim=128)
        for record in store.all_memories():
            index.add(record.memory_id, embedder.encode(record.content))
        self.assertEqual(len(index), 3)

        retrieved = index.search(embedder.encode("recurrent core cache"), k=2)
        self.assertEqual(len(retrieved), 2)
        self.assertGreaterEqual(retrieved[0][1], retrieved[1][1])

        retriever = HybridRetriever(
            store,
            index=index,
            embed=embedder.encode,
            reranker=SharedTrunkNeuralReranker(),
        )
        candidates = retriever.retrieve("recurrent core cache", k=2)
        self.assertTrue(candidates)
        trace = retriever.last_trace
        self.assertIn("lexical", trace.channels_used)
        self.assertIn("vector", trace.channels_used)
        self.assertLessEqual(len(trace.reranked_ids), 2)
        self.assertGreaterEqual(retriever.recall_at_k([candidates[0].memory_id], k=2), 1.0)
        store.close()

    def test_context_builder_respects_budget(self):
        store = self._store_with_records()
        retriever = HybridRetriever(store)
        candidates = retriever.retrieve("gated deltanet state", k=3)
        builder = EvidenceContextBuilder(token_budget=60)
        bundle = builder.build(candidates, header="MEMORY", pinned="objective: fix bug")
        self.assertLessEqual(bundle.tokens_used, bundle.token_budget + 5)
        self.assertTrue(bundle.used_ids)
        self.assertEqual(len(bundle.provenance), len(bundle.used_ids))
        self.assertIn(bundle.used_ids[0], bundle.text)
        store.close()

    def test_projection_parameter_budget(self):
        self.assertEqual(memory_projection_parameters(768), 98_304)
        proj = MemoryProjection(d=768)
        self.assertEqual(proj.n_parameters, 98_304)
        projected = proj.project(np.ones(768, dtype=np.float32))
        self.assertEqual(projected.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(projected)), 1.0, places=5)
        # 250,007,328 + 98,304 = 250,105,632 (< 251,000,000 ceiling)
        self.assertEqual(250_007_328 + proj.n_parameters, 250_105_632)

    def test_projection_state_dict_round_trip(self):
        original = MemoryProjection(d=64, dim=8, seed=3)
        restored = MemoryProjection(d=64, dim=8, seed=999)
        restored.load_state_dict(original.state_dict())
        np.testing.assert_allclose(restored.weight, original.weight)
        probe = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
        np.testing.assert_allclose(
            restored.project(probe), original.project(probe), rtol=1e-5, atol=1e-6
        )
        with self.assertRaises(ValueError):
            MemoryProjection(d=32, dim=8).load_state_dict(original.state_dict())

    def test_working_memory_eviction_keeps_pinned(self):
        wm = WorkingMemoryState(max_facts=2)
        wm.pin("critical invariant")
        wm.add_fact("observation 1")
        wm.add_fact("observation 2")
        texts = [f["text"] for f in wm.known_facts]
        self.assertIn("critical invariant", texts)
        self.assertEqual(len(texts), 2)
        wm.record_failure("bad patch")
        self.assertIn("LATEST_FAILURE", wm.pinned_context())


if __name__ == "__main__":
    unittest.main(verbosity=2)
