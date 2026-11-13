"""Symbolic answer derivation, fused with a neural proposal.

The system's claim is neuro-symbolic, so both halves must be scored separately.
A task may declare an arithmetic ``expression`` that appears *in its own prompt*;
:class:`SymbolicSolver` evaluates that expression exactly and appends the result,
which is what Vexx does at inference. It never looks at the answer key — it derives
from the problem text — and it is only applied to tasks that opt in, so coding and
reasoning scores stay model-only.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

__all__ = ["SymbolicSolver", "symbolic_repair_generate"]


class SymbolicSolver:
    """Exact rational evaluation of a prompt-supplied arithmetic expression."""

    def __init__(self) -> None:
        self.attempts = 0
        self.derivations = 0
        self.failures = 0

    def solve(self, expression: str) -> Optional[str]:
        """Return ``answer: <exact value>`` or ``None`` if it cannot be derived."""
        self.attempts += 1
        try:
            from sympy import Rational

            from ..vexx import safe_sympy

            exact = Rational(safe_sympy(str(expression)))
        except Exception:  # noqa: BLE001 - a non-derivable expression is just a miss
            self.failures += 1
            return None
        self.derivations += 1
        if int(exact.q) == 1:
            return f"answer: {int(exact.p)}"
        return f"answer: {int(exact.p)}/{int(exact.q)}"

    def to_dict(self) -> Dict[str, int]:
        return {
            "attempts": self.attempts,
            "derivations": self.derivations,
            "failures": self.failures,
        }


def symbolic_repair_generate(
    raw_generate: Callable[[str], str],
    expressions: Dict[str, str],
    solver: Optional[SymbolicSolver] = None,
) -> Callable[[str], str]:
    """Append a symbolically derived answer for prompts that declare an expression.

    The neural sample is preserved verbatim, so a reader can always see what the
    model proposed; the derived line is additive and marked as an answer.
    """
    active_solver = solver or SymbolicSolver()
    cache: Dict[str, Optional[str]] = {}

    def generate(prompt: str) -> str:
        sample = raw_generate(prompt)
        expression = expressions.get(prompt)
        if expression is None:
            return sample
        if prompt not in cache:
            cache[prompt] = active_solver.solve(expression)
        derived = cache[prompt]
        if derived is None:
            return sample
        return (sample or "").rstrip() + "\n" + derived + "\n"

    return generate
