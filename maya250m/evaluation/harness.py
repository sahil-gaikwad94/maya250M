"""Deterministic evaluation harness with explicit NOT_RUN semantics.

The harness never fabricates a score. If no generator is supplied the report is
``NOT_RUN`` with a machine-readable reason. When a generator is supplied, each
sample is checked by the task's *verifier* (unit tests, exact match, constraint
coverage), never by string similarity to a reference answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..understanding import TaskSpec, constraint_coverage
from ..verification import VerificationResult, failure, ok
from .identity import (
    EvalReport,
    Measurement,
    MeasurementError,
    MeasurementIdentity,
    MeasurementStatus,
)

__all__ = [
    "EvalTask",
    "TaskOutcome",
    "pass_at_k",
    "run_tasks",
    "run_benchmark",
    "exact_match_verifier",
    "contains_all_verifier",
    "constraint_verifier",
    "unit_test_verifier",
    "composed_verifier",
]

GenerateFn = Callable[[str], str]
Verifier = Callable[[str], VerificationResult]


@dataclass
class EvalTask:
    """One evaluation item with an executable acceptance check."""

    task_id: str
    prompt: str
    verifier: Verifier
    identity: MeasurementIdentity = MeasurementIdentity.BENCHMARK
    samples_per_task: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.samples_per_task < 1:
            raise ValueError("samples_per_task must be >= 1")
        if not callable(self.verifier):
            raise TypeError("verifier must be callable")


@dataclass
class TaskOutcome:
    """Per-task sampling result, kept even when the task fails."""

    task_id: str
    passed_samples: int
    total_samples: int
    verifications: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.passed_samples > 0

    @property
    def pass_rate(self) -> float:
        return self.passed_samples / max(1, self.total_samples)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if n < 0 or c < 0 or k < 1:
        raise ValueError("n, c must be >= 0 and k >= 1")
    if c > n:
        raise ValueError("c cannot exceed n")
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    # 1 - C(n-c, k) / C(n, k), computed in log space for stability.
    log_ratio = sum(math.log(n - c - i) - math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_ratio)


def run_tasks(
    tasks: Sequence[EvalTask],
    generate: Optional[GenerateFn] = None,
    title: str = "benchmark",
    k: int = 1,
    reason: str = "no generator supplied; benchmark not executed in this environment",
) -> EvalReport:
    """Evaluate ``tasks`` or return an explicit NOT_RUN report."""
    if not tasks:
        raise ValueError("no evaluation tasks supplied")

    identity = tasks[0].identity
    report = EvalReport(title=title)

    if generate is None:
        for task in tasks:
            report.add(
                Measurement(
                    identity=identity,
                    name=task.task_id,
                    status=MeasurementStatus.NOT_RUN,
                    metadata={"reason": reason, "samples": 0},
                )
            )
        return report

    outcomes: List[TaskOutcome] = []
    for task in tasks:
        verifications: List[Dict[str, Any]] = []
        passed = 0
        for _ in range(int(task.samples_per_task)):
            try:
                sample = generate(task.prompt)
            except Exception as exc:  # noqa: BLE001 - a generator crash is a task failure
                verifications.append(
                    {"passed": False, "failure_category": "generation_error", "message": str(exc)}
                )
                continue
            result = task.verifier(sample)
            passed += int(result.passed)
            verifications.append(result.to_dict())
        outcome = TaskOutcome(
            task_id=task.task_id,
            passed_samples=passed,
            total_samples=int(task.samples_per_task),
            verifications=verifications,
        )
        outcomes.append(outcome)
        report.add(
            Measurement(
                identity=identity,
                name=task.task_id,
                value=outcome.pass_rate,
                status=MeasurementStatus.MEASURED,
                n=outcome.total_samples,
                unit="pass_rate",
                metadata={
                    "passed_samples": passed,
                    "pass_at_k": pass_at_k(outcome.total_samples, passed, min(k, outcome.total_samples)),
                },
            )
        )

    total_samples = sum(o.total_samples for o in outcomes)
    total_passed = sum(o.passed_samples for o in outcomes)
    # Aggregate pass@k is the mean of per-task unbiased estimators. Pooling every
    # sample into one binomial would let an easy task's extra samples inflate it.
    per_task = [
        pass_at_k(o.total_samples, o.passed_samples, min(k, o.total_samples))
        for o in outcomes
    ]
    report.add(
        Measurement(
            identity=identity,
            name=f"pass@{k}",
            value=float(sum(per_task) / len(per_task)) if per_task else 0.0,
            status=MeasurementStatus.MEASURED,
            n=total_samples,
            unit="pass_rate",
            metadata={
                "tasks": len(outcomes),
                "solved_tasks": sum(1 for o in outcomes if o.passed),
                "micro_pass_rate": total_passed / max(1, total_samples),
                "aggregation": "mean_of_per_task_pass_at_k",
            },
        )
    )
    return report


def run_benchmark(tasks: Sequence[EvalTask], generate: Optional[GenerateFn], **kwargs: Any) -> EvalReport:
    """Benchmark identity wrapper (cannot be confused with validation loss)."""
    for task in tasks:
        if task.identity is not MeasurementIdentity.BENCHMARK:
            raise MeasurementError("run_benchmark only accepts BENCHMARK tasks")
    return run_tasks(tasks, generate=generate, **kwargs)


# ------------------------------------------------------------------ verifiers


def exact_match_verifier(expected: str, normalize: bool = True) -> Verifier:
    def verify(sample: str) -> VerificationResult:
        lhs = sample.strip()
        rhs = expected.strip()
        if normalize:
            lhs = " ".join(lhs.split())
            rhs = " ".join(rhs.split())
        if lhs == rhs:
            return ok("exact match")
        return failure("wrong_answer", "sample does not match the expected answer")

    return verify


def contains_all_verifier(required: Sequence[str], case_sensitive: bool = False) -> Verifier:
    needles = [str(x) for x in required]

    def verify(sample: str) -> VerificationResult:
        haystack = sample if case_sensitive else sample.lower()
        missing = [
            needle
            for needle in needles
            if (needle if case_sensitive else needle.lower()) not in haystack
        ]
        if missing:
            return failure(
                "missing_constraint", f"missing {len(missing)} required element(s)", missing=missing
            )
        return ok("all required elements present")

    return verify


def constraint_verifier(spec: TaskSpec) -> Verifier:
    """Verify every parsed constraint of a :class:`TaskSpec` is satisfied."""

    def verify(sample: str) -> VerificationResult:
        coverage = constraint_coverage(spec, sample)
        missing = [key for key, covered in coverage.items() if not covered]
        if missing:
            return failure(
                "missing_constraint", f"{len(missing)} constraint(s) uncovered", missing=missing
            )
        return ok("all constraints covered")

    return verify


def unit_test_verifier(
    fn_name: str, test_cases: Sequence[Dict[str, Any]], vexx: Any = None
) -> Verifier:
    """Execute the sample against Vexx Tier-5 tests in the sandbox."""

    def verify(sample: str) -> VerificationResult:
        controller = vexx
        if controller is None:
            from ..vexx import Vexx35Controller

            controller = Vexx35Controller()
        passed, total, errors = controller.safe_execute_python(sample, fn_name, list(test_cases))
        if total > 0 and passed == total:
            return ok("all unit tests passed", passed=passed, total=total)
        return failure(
            "tests_failed",
            f"{passed}/{total} unit tests passed",
            errors=list(errors)[:5],
        )

    return verify


def composed_verifier(*verifiers: Verifier) -> Verifier:
    """AND-compose verifiers; the first failure is returned verbatim."""

    def verify(sample: str) -> VerificationResult:
        for verifier in verifiers:
            result = verifier(sample)
            if not result.passed:
                return result
        return ok("all composed checks passed")

    return verify
