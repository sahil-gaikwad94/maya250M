"""Five executable domain batteries: coding, maths, reasoning, research, neural.

Every task is scored by *running* something — unit tests in the Vexx sandbox, exact
rational arithmetic, or a parse of the completion. None of them compares strings to
a reference completion, so a model cannot score by echoing the prompt, and a battery
bug shows up as a gold solution failing rather than as an inflated score.

Domains:

* ``coding``    — complete a function; verified by executing unit tests
* ``maths``     — state an exact value; verified as a ``Fraction``, plus inline
                  ``<<expr=value>>`` markers must be arithmetically self-consistent
* ``reasoning`` — complete a function under forbidden-token constraints *and* pass
                  its unit tests (spec adherence, not just output)
* ``research``  — multi-step numeric derivation; the reported value must be exact
* ``neural``    — does the model's own continuation parse as Python at all

The last domain is deliberately model-only: it measures the network's generative
ability with no symbolic help, which is what "the model itself can do" means.
"""

from __future__ import annotations

import ast
import re
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence

from ..verification import VerificationResult, failure, ok
from .harness import (
    EvalTask,
    Verifier,
    contains_all_verifier,
    composed_verifier,
    unit_test_verifier,
)
from .identity import MeasurementIdentity

__all__ = [
    "DOMAINS",
    "all_tasks",
    "domain_tasks",
    "expressions_by_prompt",
    "exact_answer_verifier",
    "extract_stated_answer",
    "forbidden_tokens_verifier",
    "marker_self_consistency_verifier",
    "completion_syntax_verifier",
    "completion_diversity",
]

_NUMBER = r"-?\d+(?:\.\d+)?(?:/\d+)?"

#: Ordered by specificity; the *last* stated answer in a sample is the final one.
_ANSWER_PATTERNS = (
    re.compile(r"<<\s*[^<>=]+?=\s*(" + _NUMBER + r")\s*>>"),
    re.compile(r"RESULT\s*=\s*(" + _NUMBER + r")"),
    re.compile(r"answer\s*(?:is|:|=)\s*(" + _NUMBER + r")", re.IGNORECASE),
)

_MARKER = re.compile(r"<<\s*([^<>=]+?)\s*=\s*(" + _NUMBER + r")\s*>>")

#: Tokens no sandboxed candidate needs, and which widen the attack surface.
UNSAFE_TOKENS = ("import ", "eval(", "exec(", "open(", "__import__", "os.system")


class _LazyVexx:
    """Defer building the controller until a verifier actually runs."""

    def __init__(self) -> None:
        self._controller = None

    def safe_execute_python(self, *args: Any, **kwargs: Any):
        if self._controller is None:
            from ..vexx import Vexx35Controller

            self._controller = Vexx35Controller()
        return self._controller.safe_execute_python(*args, **kwargs)


_VEXX = _LazyVexx()


# ------------------------------------------------------------------ extraction


def extract_stated_answer(sample: str) -> Optional[str]:
    """Return the final numeric answer the sample claims, or ``None``.

    Appending a corrected answer is a normal pattern, so the last marker wins.
    """
    for pattern in _ANSWER_PATTERNS:
        matches = pattern.findall(sample or "")
        if matches:
            return matches[-1]
    return None


def _to_fraction(text: Optional[str]) -> Optional[Fraction]:
    if text is None:
        return None
    cleaned = str(text).strip().replace(",", "").replace("−", "-")
    if not cleaned:
        return None
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        try:
            return Fraction(float(cleaned))
        except (TypeError, ValueError):
            return None


# ------------------------------------------------------------------- verifiers


def exact_answer_verifier(expected: str) -> Verifier:
    """The stated answer must equal ``expected`` as an exact rational."""
    want = _to_fraction(expected)
    if want is None:
        raise ValueError(f"expected answer {expected!r} is not numeric")

    def verify(sample: str) -> VerificationResult:
        stated = extract_stated_answer(sample)
        if stated is None:
            return failure(
                "missing_constraint",
                "no stated answer (expected <<expr=value>>, RESULT = ..., or 'answer: ...')",
            )
        got = _to_fraction(stated)
        if got is None:
            return failure("schema_violation", f"could not parse {stated!r} as a number")
        if got != want:
            return failure("wrong_answer", f"stated {got}, expected {want}")
        return ok("exact rational match", stated=str(got))

    return verify


def marker_self_consistency_verifier() -> Verifier:
    """Every ``<<expr=value>>`` marker must satisfy ``expr == value`` exactly."""

    def verify(sample: str) -> VerificationResult:
        markers = _MARKER.findall(sample or "")
        if not markers:
            return ok("no inline marker to check")
        from sympy import Rational

        from ..vexx import safe_sympy

        for expression, value in markers:
            try:
                exact = Rational(safe_sympy(expression))
            except Exception as exc:  # noqa: BLE001 - any parse/eval failure is the result
                return failure("syntax_error", f"marker {expression!r} failed: {exc}")
            want = _to_fraction(value)
            got = Fraction(int(exact.p), int(exact.q))
            if want is None or got != want:
                return failure(
                    "wrong_answer",
                    f"{expression} evaluates to {got}, marker claims {value}",
                )
        return ok("all inline markers self-consistent", markers=len(markers))

    return verify


def forbidden_tokens_verifier(tokens: Sequence[str]) -> Verifier:
    """Reject samples containing any forbidden token (case-insensitive)."""
    needles = [token.lower() for token in tokens]

    def verify(sample: str) -> VerificationResult:
        haystack = (sample or "").lower()
        present = [needle for needle in needles if needle in haystack]
        if present:
            return failure(
                "ast_policy_violation",
                f"used forbidden token(s) {present}",
                forbidden=present,
            )
        return ok("no forbidden tokens")

    return verify


def _prompt_definition_name(prompt: str) -> Optional[str]:
    match = re.search(r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z_]\w*)", prompt, re.M)
    return match.group(1) if match else None


def _statement_body(node: ast.AST) -> List[ast.stmt]:
    """Body statements with the docstring removed."""
    body = list(getattr(node, "body", []) or [])
    if body:
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(getattr(first, "value", None), ast.Constant)
            and isinstance(first.value.value, str)
        ):
            body = body[1:]
    return body


def completion_syntax_verifier(
    prompt: str, min_new_chars: int = 8, required_name: Optional[str] = None
) -> Verifier:
    """The continuation must parse *and* implement the definition it was asked for.

    Parsing alone is too generous: a model that ignores the prompt and emits an
    unrelated (but valid) function scored 1.000 here. Requiring the named
    definition to have a real body beyond its docstring is what distinguishes
    "continued this function" from "produced any syntactically valid text".
    """
    name = required_name if required_name is not None else _prompt_definition_name(prompt)

    def verify(sample: str) -> VerificationResult:
        text = sample or ""
        if not text.startswith(prompt):
            return failure("schema_violation", "sample does not start with the prompt")
        continuation = text[len(prompt) :]
        if len(continuation.strip()) < int(min_new_chars):
            return failure(
                "schema_violation",
                f"only {len(continuation.strip())} characters generated",
            )
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return failure("syntax_error", f"{exc.msg} (line {exc.lineno})")
        if name:
            definitions = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == name
            ]
            if not definitions:
                return failure(
                    "schema_violation", f"the sample never defines {name!r}"
                )
            if not _statement_body(definitions[0]):
                return failure(
                    "schema_violation", f"{name!r} has no implementation beyond its docstring"
                )
        return ok("completion parses as Python", defines=name)

    return verify


def completion_diversity(samples: Sequence[str]) -> Dict[str, Any]:
    """How many distinct completions were produced (mode-collapse detector).

    A model stuck on one memorized template can look perfect on any single
    syntactic check; a unique ratio near zero is the evidence that it is not
    answering the prompts at all.
    """
    # Digits are normalized away on purpose: a model stuck on one template emits
    # ``sum_to_21``, ``sum_to_26``, ``sum_to_32`` — different text, same template.
    normalized = [
        re.sub(r"\d+", "N", " ".join((sample or "").split())) for sample in samples
    ]
    unique = len(set(normalized))
    return {
        "samples": len(normalized),
        "unique": unique,
        "unique_ratio": unique / max(1, len(normalized)),
        "collapse_detected": unique <= max(1, len(normalized) // 3),
    }


# --------------------------------------------------------------- task payloads

CODING_SPECS: List[Dict[str, Any]] = [
    {
        "task_id": "coding_add",
        "stem": 'def add(a: int, b: int) -> int:\n    """Return a + b."""\n',
        "fn_name": "add",
        "tests": [{"input": [2, 3], "expected": 5}, {"input": [-1, 1], "expected": 0}],
        "gold": "    return a + b\n",
    },
    {
        "task_id": "coding_square",
        "stem": 'def square(n: int) -> int:\n    """Return n squared."""\n',
        "fn_name": "square",
        "tests": [{"input": [5], "expected": 25}, {"input": [-3], "expected": 9}],
        "gold": "    return n * n\n",
    },
    {
        "task_id": "coding_max_of_three",
        "stem": (
            'def max_of_three(a: int, b: int, c: int) -> int:\n'
            '    """Return the largest of the three values."""\n'
        ),
        "fn_name": "max_of_three",
        "tests": [
            {"input": [1, 5, 3], "expected": 5},
            {"input": [-2, -5, -9], "expected": -2},
        ],
        "gold": "    return max(a, b, c)\n",
    },
    {
        "task_id": "coding_count_vowels",
        "stem": (
            'def count_vowels(text: str) -> int:\n'
            '    """Count lowercase vowels (a, e, i, o, u) in text."""\n'
        ),
        "fn_name": "count_vowels",
        "tests": [
            {"input": ["hello"], "expected": 2},
            {"input": ["xyz"], "expected": 0},
        ],
        "gold": "    return sum(1 for ch in text if ch in \"aeiou\")\n",
    },
    {
        "task_id": "coding_factorial",
        "stem": 'def factorial(n: int) -> int:\n    """Return n! with 0! == 1."""\n',
        "fn_name": "factorial",
        "tests": [{"input": [0], "expected": 1}, {"input": [5], "expected": 120}],
        "gold": "    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n",
    },
    {
        "task_id": "coding_reverse_words",
        "stem": (
            'def reverse_words(text: str) -> str:\n'
            '    """Reverse word order, keeping single spaces."""\n'
        ),
        "fn_name": "reverse_words",
        "tests": [
            {"input": ["a b c"], "expected": "c b a"},
            {"input": ["one"], "expected": "one"},
        ],
        "gold": "    return \" \".join(reversed(text.split()))\n",
    },
]

MATH_SPECS: List[Dict[str, Any]] = [
    {"task_id": "maths_mul_add", "expression": "17 * 23 + 9", "expected": "400"},
    {"task_id": "maths_power", "expression": "2 ** 10", "expected": "1024"},
    {"task_id": "maths_gauss", "expression": "100 * 101 / 2", "expected": "5050"},
    {"task_id": "maths_fraction_add", "expression": "7 / 8 + 1 / 8", "expected": "1"},
    {"task_id": "maths_sub_mul", "expression": "1234 - 5 * 6", "expected": "1204"},
    {"task_id": "maths_fraction_mul", "expression": "(3 / 4) * (8 / 9)", "expected": "2/3"},
]
MATH_PROMPT = (
    "Compute the value and state it as `answer: <value>` (or a `<<expr=value>>` marker).\n"
)

REASONING_SPECS: List[Dict[str, Any]] = [
    {
        "task_id": "reasoning_safe_div",
        "stem": (
            'def safe_div(a: float, b: float):\n'
            '    """Return a / b, or None when b is zero. Do not raise."""\n'
        ),
        "fn_name": "safe_div",
        "tests": [
            {"input": [6, 3], "expected": 2.0},
            {"input": [1, 0], "expected": None},
        ],
        "gold": "    if b == 0:\n        return None\n    return a / b\n",
    },
    {
        "task_id": "reasoning_clamp",
        "stem": (
            'def clamp(x: int, lo: int, hi: int) -> int:\n'
            '    """Return x limited to the inclusive range [lo, hi]."""\n'
        ),
        "fn_name": "clamp",
        "tests": [
            {"input": [5, 0, 3], "expected": 3},
            {"input": [-1, 0, 3], "expected": 0},
            {"input": [2, 0, 3], "expected": 2},
        ],
        "gold": "    return max(lo, min(hi, x))\n",
    },
    {
        "task_id": "reasoning_second_largest",
        "stem": (
            'def second_largest(values):\n'
            '    """Return the second largest distinct value, or None if absent."""\n'
        ),
        "fn_name": "second_largest",
        "tests": [
            {"input": [[3, 1, 2]], "expected": 2},
            {"input": [[5, 5]], "expected": None},
        ],
        "gold": (
            "    unique = sorted(set(values))\n"
            "    return unique[-2] if len(unique) > 1 else None\n"
        ),
    },
    {
        "task_id": "reasoning_is_even",
        "stem": (
            'def is_even(n: int) -> bool:\n'
            '    """Return True when n is even. No imports allowed."""\n'
        ),
        "fn_name": "is_even",
        "tests": [
            {"input": [4], "expected": True},
            {"input": [7], "expected": False},
        ],
        "gold": "    return n % 2 == 0\n",
    },
]

RESEARCH_SPECS: List[Dict[str, Any]] = [
    {
        "task_id": "research_compound",
        "expression": "1000 * 1.05 ** 2",
        "expected": "1102.5",
        "question": "A 1000 unit fund grows 5% twice. What is the final value?",
    },
    {
        "task_id": "research_even_sum",
        "expression": "12 * 13",
        "expected": "156",
        "question": "What is the sum of the first 12 positive even numbers?",
    },
    {
        "task_id": "research_choose",
        "expression": "10 * 9 * 8 / 6",
        "expected": "120",
        "question": "How many 3-element subsets does a 10-element set have?",
    },
    {
        "task_id": "research_mean",
        "expression": "(4 + 9 + 17) / 3",
        "expected": "10",
        "question": "What is the arithmetic mean of 4, 9 and 17?",
    },
]
RESEARCH_PROMPT = (
    "Derive the value step by step, then state the final value as `answer: <value>`.\n"
)

NEURAL_STEMS: List[str] = [
    'def total(items):\n    """Sum every item in the list."""\n',
    'def first_char(text):\n    """Return the first character of text."""\n',
    'class Counter:\n    """Count how many times a value was seen."""\n',
    'def is_blank(text):\n    """Return True when text is empty after stripping."""\n',
    'def double_all(values):\n    """Return a new list with each value doubled."""\n',
    'def last_item(values):\n    """Return the final element of the list."""\n',
]

DOMAINS = ("coding", "maths", "reasoning", "research", "neural")


def _benchmark(task: EvalTask) -> EvalTask:
    task.identity = MeasurementIdentity.BENCHMARK
    return task


def coding_tasks() -> List[EvalTask]:
    return [
        EvalTask(
            task_id=spec["task_id"],
            prompt=spec["stem"],
            verifier=unit_test_verifier(spec["fn_name"], spec["tests"], vexx=_VEXX),
            metadata={"domain": "coding", "fn_name": spec["fn_name"], "gold": spec["gold"]},
        )
        for spec in CODING_SPECS
    ]


def maths_tasks() -> List[EvalTask]:
    tasks = []
    for spec in MATH_SPECS:
        prompt = (
            f"{MATH_PROMPT}Problem: evaluate `{spec['expression']}`.\n"
            f"State the exact value.\n"
        )
        tasks.append(
            EvalTask(
                task_id=spec["task_id"],
                prompt=prompt,
                verifier=composed_verifier(
                    exact_answer_verifier(spec["expected"]),
                    marker_self_consistency_verifier(),
                ),
                metadata={
                    "domain": "maths",
                    "expression": spec["expression"],
                    "expected": spec["expected"],
                    "allow_symbolic_repair": True,
                    "gold": f"answer: {spec['expected']}\n",
                },
            )
        )
    return tasks


def reasoning_tasks() -> List[EvalTask]:
    tasks = []
    for spec in REASONING_SPECS:
        tasks.append(
            EvalTask(
                task_id=spec["task_id"],
                prompt=spec["stem"],
                verifier=composed_verifier(
                    forbidden_tokens_verifier(UNSAFE_TOKENS),
                    unit_test_verifier(spec["fn_name"], spec["tests"], vexx=_VEXX),
                ),
                metadata={"domain": "reasoning", "fn_name": spec["fn_name"], "gold": spec["gold"]},
            )
        )
    return tasks


def research_tasks() -> List[EvalTask]:
    tasks = []
    for spec in RESEARCH_SPECS:
        prompt = f"{RESEARCH_PROMPT}Question: {spec['question']}\n"
        tasks.append(
            EvalTask(
                task_id=spec["task_id"],
                prompt=prompt,
                verifier=composed_verifier(
                    exact_answer_verifier(spec["expected"]),
                    marker_self_consistency_verifier(),
                ),
                metadata={
                    "domain": "research",
                    "expression": spec["expression"],
                    "expected": spec["expected"],
                    "allow_symbolic_repair": True,
                    "gold": f"answer: {spec['expected']}\n",
                },
            )
        )
    return tasks


def neural_tasks() -> List[EvalTask]:
    return [
        EvalTask(
            task_id=f"neural_syntax_{index}",
            prompt=stem,
            verifier=completion_syntax_verifier(
                stem, required_name=_prompt_definition_name(stem)
            ),
            metadata={"domain": "neural", "required_name": _prompt_definition_name(stem)},
        )
        for index, stem in enumerate(NEURAL_STEMS, start=1)
    ]


_BUILDERS = {
    "coding": coding_tasks,
    "maths": maths_tasks,
    "reasoning": reasoning_tasks,
    "research": research_tasks,
    "neural": neural_tasks,
}


def domain_tasks(domain: Optional[str] = None) -> Dict[str, List[EvalTask]]:
    """Build every battery (or one), asserting the benchmark identity."""
    names = [domain] if domain else list(DOMAINS)
    built: Dict[str, List[EvalTask]] = {}
    for name in names:
        if name not in _BUILDERS:
            raise KeyError(f"unknown domain {name!r}; valid: {sorted(_BUILDERS)}")
        built[name] = [_benchmark(task) for task in _BUILDERS[name]()]
    return built


def all_tasks() -> List[EvalTask]:
    return [task for tasks in domain_tasks().values() for task in tasks]


def expressions_by_prompt(tasks: Sequence[EvalTask]) -> Dict[str, str]:
    """Prompt -> arithmetic expression for tasks that may use symbolic repair."""
    mapping: Dict[str, str] = {}
    for task in tasks:
        expression = task.metadata.get("expression")
        if expression and task.metadata.get("allow_symbolic_repair"):
            mapping[task.prompt] = str(expression)
    return mapping
