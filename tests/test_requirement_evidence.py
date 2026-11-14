"""P0.4 tests: typed requirement evidence, never lexical-overlap verification."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.requirement_evidence import (
    RequirementStatus,
    requirement_verifiers_for_text,
    verify_requirements,
    verify_task_spec_requirements,
)
from maya250m.understanding import (
    TaskSpec,
    lexical_requirement_coverage,
    parse_task_spec,
)


class TestLexicalCoverageIsNotVerification(unittest.TestCase):
    def test_lexical_coverage_is_named_as_an_addressing_audit(self):
        spec = parse_task_spec(
            "Fix the bug. You must preserve the public API and must not add new dependencies."
        )
        coverage = lexical_requirement_coverage(spec, "The public API is preserved; tests pass.")
        self.assertEqual(len(coverage), len(spec.constraints))
        self.assertTrue(any(coverage.values()))
        self.assertTrue(any(not value for value in coverage.values()))
        # And the old, ambiguous name must be gone.
        import maya250m.understanding as understanding

        self.assertFalse(hasattr(understanding, "constraint_coverage"))

    def test_requirements_without_verifiers_are_unverified(self):
        requirement = {"requirement_id": "r1", "text": "be nice", "verifier_ids": []}
        report = verify_requirements([requirement], {})
        self.assertEqual(report["r1"].status, RequirementStatus.UNVERIFIED)

    def test_missing_context_is_unverified_not_verified(self):
        requirement = {
            "requirement_id": "r1",
            "text": "must pass tests",
            "verifier_ids": ["test_outcomes"],
        }
        report = verify_requirements([requirement], {})
        self.assertEqual(report["r1"].status, RequirementStatus.UNVERIFIED)


class TestTypedVerifiers(unittest.TestCase):
    def test_api_signature_preservation(self):
        requirement = {
            "requirement_id": "r1",
            "verifier_ids": ["api_signature"],
        }
        kept = verify_requirements(
            [requirement],
            {
                "expected_signatures": {"add": "(a, b)"},
                "observed_signatures": {"add": "(a, b)"},
            },
        )["r1"]
        self.assertEqual(kept.status, RequirementStatus.VERIFIED)
        self.assertEqual(kept.evidence_ids, ("signature:add",))

        broken = verify_requirements(
            [requirement],
            {
                "expected_signatures": {"add": "(a, b)"},
                "observed_signatures": {"add": "(a, b, c)"},
            },
        )["r1"]
        self.assertEqual(broken.status, RequirementStatus.FAILED)

    def test_test_outcomes(self):
        requirement = {"requirement_id": "r1", "verifier_ids": ["test_outcomes"]}
        passed = verify_requirements(
            [requirement],
            {"required_test_ids": ["tests/test_a.py"], "test_results": {"tests/test_a.py": True}},
        )["r1"]
        self.assertEqual(passed.status, RequirementStatus.VERIFIED)
        failed = verify_requirements(
            [requirement],
            {"required_test_ids": ["tests/test_a.py"], "test_results": {"tests/test_a.py": False}},
        )["r1"]
        self.assertEqual(failed.status, RequirementStatus.FAILED)

    def test_dependency_restrictions(self):
        requirement = {"requirement_id": "r1", "verifier_ids": ["dependency_restrictions"]}
        clean = verify_requirements(
            [requirement],
            {"observed_dependencies": ["stdlib:heapq"], "forbidden_dependencies": ["requests"]},
        )["r1"]
        self.assertEqual(clean.status, RequirementStatus.VERIFIED)
        dirty = verify_requirements(
            [requirement],
            {"observed_dependencies": ["requests"], "forbidden_dependencies": ["requests"]},
        )["r1"]
        self.assertEqual(dirty.status, RequirementStatus.FAILED)

    def test_output_schema_and_complexity(self):
        requirement = {
            "requirement_id": "r1",
            "verifier_ids": ["output_schema", "complexity"],
        }
        report = verify_requirements(
            [requirement],
            {
                "expected_format": "json",
                "observed_format": "json",
                "expected_complexity": "O(n log n)",
                "measured_complexity": "O(n log n)",
            },
        )["r1"]
        self.assertEqual(report.status, RequirementStatus.VERIFIED)
        mixed = verify_requirements(
            [requirement],
            {
                "expected_format": "json",
                "observed_format": "yaml",
                "expected_complexity": "O(n log n)",
                "measured_complexity": "O(n log n)",
            },
        )["r1"]
        self.assertEqual(mixed.status, RequirementStatus.FAILED)

    def test_task_spec_requirements_are_wired(self):
        spec = TaskSpec(
            objective="x",
            requirements=["must preserve the public API signature"],
            constraints=["must not add new dependencies"],
        )
        self.assertIn("api_signature", requirement_verifiers_for_text(spec.requirements[0]))
        report = verify_task_spec_requirements(
            spec,
            {
                "expected_signatures": {"add": "(a, b)"},
                "observed_signatures": {"add": "(a, b)"},
                "observed_dependencies": ["stdlib:heapq"],
                "forbidden_dependencies": [],
            },
        )
        self.assertEqual(report["requirement_0"].status, RequirementStatus.VERIFIED)
        self.assertEqual(report["constraint_0"].status, RequirementStatus.VERIFIED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
