"""Reasoning-aware inference: adaptive depth, abstention, verified revision.

This is the bridge between the reasoning/memory/verification modules and the
model itself. Nothing here is a new architecture — it only decides *how* to run
the existing recurrent core for a given request:

1. parse the request into a :class:`~maya250m.understanding.TaskSpec`;
2. refuse critically ambiguous requests instead of guessing
   (``StrategySelector``'s critical flags);
3. choose ``R`` (recurrent-core loop count) from observable difficulty signals
   (:class:`~maya250m.reasoning.controller.AdaptiveLoopController`);
4. decompose the work into a DAG of subgoals for the record;
5. verify the answer with the caller's executable verifier and, on failure, run a
   *bounded* verifier-driven revision (never blind retries);
6. emit a :class:`SolveResult` that reports the loop count, attempts, evidence and
   a correctly-identified :class:`~maya250m.evaluation.identity.Measurement`.

Ablation flags (``disabled``) allow any of these stages to be switched off, which
is what the A–P matrix drives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from .evaluation.identity import Measurement, MeasurementIdentity, MeasurementStatus
from .generate import PLAIN, GenerationResult, generate_maya
from .reasoning.controller import AdaptiveLoopController
from .reasoning.correction import CorrectionPolicy
from .reasoning.decompose import CRITICAL_AMBIGUITY_FLAGS, decompose_task
from .reasoning.state import ReasoningState
from .understanding import TaskSpec, parse_task_spec
from .verification import VerificationResult, failure, ok

__all__ = [
    "SolveConfig",
    "SolveResult",
    "ABLATION_FLAGS",
    "select_loops",
    "generate_adaptive",
    "solve_task",
]

#: Stages that the evaluation ablation matrix may disable at inference time.
ABLATION_FLAGS = frozenset(
    {"adaptive_depth", "self_correction", "decomposition", "constraint_check"}
)

GenerateFn = Callable[[str, int], str]


@dataclass
class SolveConfig:
    """Bounded inference configuration; all fields map to the spec's controls."""

    max_new_tokens: int = 128
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = None
    mode: str = PLAIN
    seed: Optional[int] = None
    adaptive_depth: bool = True
    min_loops: int = 1
    max_loops: int = 4
    max_attempts: int = 2
    constraint_check: bool = True
    identity: MeasurementIdentity = MeasurementIdentity.BENCHMARK

    def validate(self) -> "SolveConfig":
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if not 1 <= int(self.min_loops) <= int(self.max_loops) <= 8:
            raise ValueError("require 1 <= min_loops <= max_loops <= 8")
        return self


@dataclass
class SolveResult:
    """Everything a reader needs to audit one solved (or refused) request."""

    prompt: str
    spec: TaskSpec
    answer: str = ""
    loops_used: int = 1
    attempts: int = 0
    abstained: bool = False
    verification: Optional[VerificationResult] = None
    depth_explain: Dict[str, Any] = field(default_factory=dict)
    subgoals: List[str] = field(default_factory=list)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    identity: MeasurementIdentity = MeasurementIdentity.BENCHMARK
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return bool(self.verification is not None and self.verification.passed)

    def measurement(self) -> Measurement:
        """Pass/fail measurement, or ``NOT_RUN`` when nothing verified the answer."""
        if self.verification is None:
            return Measurement(
                identity=self.identity,
                name="solve_verified",
                status=MeasurementStatus.NOT_RUN,
                metadata={"reason": "no executable verifier was supplied", "prompt": self.prompt[:120]},
            )
        return Measurement(
            identity=self.identity,
            name="solve_verified",
            value=1.0 if self.verification.passed else 0.0,
            n=1,
            unit="pass_rate",
            metadata={
                "loops": self.loops_used,
                "attempts": self.attempts,
                "abstained": self.abstained,
                "failure_category": self.verification.failure_category,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "task_type": self.spec.task_type,
            "answer": self.answer,
            "loops_used": self.loops_used,
            "attempts": self.attempts,
            "abstained": self.abstained,
            "verified": self.verified,
            "verification": self.verification.to_dict() if self.verification else None,
            "depth_explain": dict(self.depth_explain),
            "subgoals": list(self.subgoals),
            "trajectory": list(self.trajectory),
            "finish_reason": self.finish_reason,
            "identity": self.identity.value,
        }


def select_loops(
    spec: Optional[TaskSpec],
    prompt_tokens: int,
    controller: Optional[AdaptiveLoopController] = None,
    adaptive: bool = True,
    default_loops: int = 1,
    state: Optional[ReasoningState] = None,
    disabled: FrozenSet[str] = frozenset(),
) -> Tuple[int, Dict[str, Any]]:
    """Choose the recurrent-core loop count, with an auditable explanation."""
    controller = controller or AdaptiveLoopController()
    if (not adaptive) or ("adaptive_depth" in disabled):
        loops = max(1, min(int(controller.max_loops), int(default_loops)))
        return loops, {
            "loops": loops,
            "mode": "fixed",
            "reason": "adaptive depth disabled" if "adaptive_depth" in disabled else "adaptive=False",
        }
    explain = controller.explain(spec=spec, state=state, prompt_tokens=prompt_tokens)
    explain["mode"] = "adaptive"
    return int(explain["loops"]), explain


def generate_adaptive(
    model,
    tokenizer,
    prompt: str,
    config: Optional[SolveConfig] = None,
    spec: Optional[TaskSpec] = None,
    controller: Optional[AdaptiveLoopController] = None,
    vexx: Any = None,
    disabled: FrozenSet[str] = frozenset(),
    state: Optional[ReasoningState] = None,
) -> Tuple[GenerationResult, int, Dict[str, Any]]:
    """Generate with the loop count chosen for this prompt."""
    cfg = (config or SolveConfig()).validate()
    spec = spec or parse_task_spec(prompt)
    prompt_tokens = len(tokenizer.encode(prompt, add_bos=False, add_eos=False))
    loops, explain = select_loops(
        spec,
        prompt_tokens,
        controller=controller,
        adaptive=cfg.adaptive_depth,
        default_loops=int(getattr(model, "recurrent_loops", 1)),
        state=state,
        disabled=disabled,
    )
    requested = int(loops)
    loops = max(int(cfg.min_loops), min(int(cfg.max_loops), loops))
    if requested != loops:
        explain["loops_requested"] = requested
        explain["clamped_to"] = [int(cfg.min_loops), int(cfg.max_loops)]
    explain["loops"] = loops
    result = generate_maya(
        model,
        tokenizer,
        prompt,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        vexx=vexx,
        seed=cfg.seed,
        mode=cfg.mode,
        num_loops=loops,
    )
    explain["identity"] = cfg.identity.value
    return result, loops, explain


def _subgoals_for(spec: TaskSpec, disabled: FrozenSet[str]) -> List[str]:
    if "decomposition" in disabled:
        return []
    decomposition = decompose_task(spec)
    return [
        f"{sub.subproblem_id}: {sub.description}"
        for sub in sorted(decomposition.subproblems, key=lambda s: s.subproblem_id)
    ]


def solve_task(
    model,
    tokenizer,
    prompt: str,
    verifier: Optional[Callable[[str], VerificationResult]] = None,
    config: Optional[SolveConfig] = None,
    vexx: Any = None,
    disabled: FrozenSet[str] = frozenset(),
    generate_fn: Optional[GenerateFn] = None,
    controller: Optional[AdaptiveLoopController] = None,
    spec: Optional[TaskSpec] = None,
) -> SolveResult:
    """Solve one request end-to-end with bounded, verifier-driven revision."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    cfg = (config or SolveConfig()).validate()
    if generate_fn is None and (model is None or tokenizer is None):
        raise ValueError("solve_task needs (model, tokenizer) or a generate_fn")

    spec = spec or parse_task_spec(prompt)
    state = ReasoningState(
        objective=prompt[:400],
        task_type=spec.task_type,
        constraints=list(spec.constraints),
        max_steps=max(2, int(cfg.max_attempts) + 1),
    )
    result = SolveResult(prompt=prompt, spec=spec, identity=cfg.identity)

    if "decomposition" not in disabled:
        try:
            result.subgoals = _subgoals_for(spec, disabled)
            state.subgoals = [line.split(": ", 1)[-1] for line in result.subgoals]
        except Exception:  # noqa: BLE001 - decomposition is advisory, never fatal
            result.subgoals = []

    # 1. Refuse critically ambiguous requests instead of guessing.
    critical = set(spec.ambiguity_flags) & set(CRITICAL_AMBIGUITY_FLAGS)
    if critical and "self_correction" not in disabled:
        result.abstained = True
        result.verification = failure(
            "ambiguous_request",
            f"request is critically ambiguous ({', '.join(sorted(critical))}); "
            "asking instead of guessing",
            ambiguity_flags=sorted(critical),
        )
        result.depth_explain = {"loops": 0, "mode": "abstained"}
        result.trajectory.append(
            {
                "attempt": 0,
                "loops": 0,
                "abstained": True,
                "reason": "critical_ambiguity",
                "flags": sorted(critical),
            }
        )
        return result

    # 2. Verify with an executable check whenever one exists.
    check = verifier
    if check is None and cfg.constraint_check and spec.constraints and "constraint_check" not in disabled:
        from .evaluation.harness import constraint_verifier

        check = constraint_verifier(spec)
    if "constraint_check" in disabled:
        check = verifier
    policy = CorrectionPolicy(max_attempts=max(1, int(cfg.max_attempts) - 1))

    feedback = ""
    for attempt in range(int(cfg.max_attempts)):
        attempt_prompt = prompt if not feedback else prompt + feedback

        if generate_fn is not None:
            prompt_tokens = (
                len(tokenizer.encode(attempt_prompt, add_bos=False, add_eos=False))
                if tokenizer is not None
                else len(attempt_prompt.split())
            )
            loops, explain = select_loops(
                spec,
                prompt_tokens,
                controller=controller,
                adaptive=cfg.adaptive_depth,
                default_loops=int(getattr(model, "recurrent_loops", 1)) if model is not None else 1,
                state=state,
                disabled=disabled,
            )
            requested = int(loops)
            loops = max(int(cfg.min_loops), min(int(cfg.max_loops), loops))
            if requested != loops:
                explain["loops_requested"] = requested
                explain["clamped_to"] = [int(cfg.min_loops), int(cfg.max_loops)]
            explain["loops"] = loops
            answer = str(generate_fn(attempt_prompt, loops))
            finish_reason = "external_generator"
        else:
            generation, loops, explain = generate_adaptive(
                model,
                tokenizer,
                attempt_prompt,
                config=cfg,
                spec=spec,
                controller=controller,
                vexx=vexx,
                disabled=disabled,
                state=state,
            )
            answer = generation.continuation
            finish_reason = generation.finish_reason

        verification = check(answer) if check is not None else None
        result.attempts = attempt + 1
        result.answer = answer
        result.loops_used = loops
        result.depth_explain = explain
        result.finish_reason = finish_reason
        result.verification = verification
        result.trajectory.append(
            {
                "attempt": attempt + 1,
                "loops": loops,
                "difficulty_score": explain.get("difficulty_score", 0.0),
                "finish_reason": finish_reason,
                "verification": verification.to_dict() if verification else None,
                "answer_chars": len(answer),
            }
        )

        if verification is None or verification.passed:
            break
        if "self_correction" in disabled:
            break

        plan = policy.next_plan(verification)
        state.rejected_approaches.append(
            f"{verification.failure_category}: {verification.message}".strip()
        )
        if plan.action == "abstain":
            result.abstained = True
            result.trajectory[-1]["correction"] = "abstain"
            break
        feedback = f"\n\n[REVISION: {plan.instruction}]"
        result.trajectory[-1]["correction"] = plan.action
        if attempt + 1 >= int(cfg.max_attempts):
            break

    return result
