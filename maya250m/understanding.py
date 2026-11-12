"""Typed task/issue understanding for Maya-250M.

A ``TaskSpec`` is the structured form of "what is actually being asked":
objective, task type, explicit requirements, constraints, entities, referenced
files/symbols, desired output format, ambiguity flags and provenance.

Two parsers are provided:

* ``parse_task_spec`` — a deterministic, dependency-free heuristic parser. It is
  the default because it can never hallucinate a requirement that is not in the
  prompt, and its output is auditable.
* ``NeuralTaskSpecParser`` — a small trainable head over the shared Maya trunk
  (``d -> task-type logits + ambiguity score``). It is wired for Stage-2 SFT but
  is *off by default*: until it is trained and evaluated, the heuristic parser
  stays authoritative (see the "no cosmetic intelligence upgrades" rule).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "TaskSpec",
    "TASK_TYPES",
    "OUTPUT_FORMATS",
    "parse_task_spec",
    "validate_task_spec",
    "constraint_coverage",
    "NeuralTaskSpecParser",
]

TASK_TYPES = (
    "bug_fix",
    "code_edit",
    "code_generation",
    "question",
    "math",
    "research",
    "refactor",
    "test_writing",
    "explanation",
    "unknown",
)

OUTPUT_FORMATS = (
    "patch",
    "code_block",
    "explanation",
    "numbered_list",
    "short_answer",
    "derivation",
    "json",
    "unknown",
)

_MODAL_PATTERN = (
    r"(?:must(?:\s+not)?|do\s+not|don't|should\s+not|never|without|at\s+most|at\s+least|only)"
)
_COORDINATED_MODAL = re.compile(
    r"\s+and\s+(?=(?:must|do\s+not|don't|should\s+not|never|also|keep|preserve|only|without)\b)",
    re.IGNORECASE,
)
_CONSTRAINT_PATTERNS = (
    r"\bmust\b[^.\n;]*",
    r"\bdo not\b[^.\n;]*",
    r"\bdon't\b[^.\n;]*",
    r"\bshould not\b[^.\n;]*",
    r"\bwithout\b[^.\n;]*",
    r"\bat most\b[^.\n;]*",
    r"\bat least\b[^.\n;]*",
    r"\bonly\b[^.\n;]*",
    r"\bkeep\b[^.\n;]*",
    r"\bpreserve\b[^.\n;]*",
    r"\bbackwards?[ -]compatib[^.\n;]*",
    r"\bO\([^)]+\)[^.\n;]*",
    r"\bno new (?:dependenc|file|token|API)[^.\n;]*",
)


def _extract_constraints(text: str) -> List[str]:
    """Extract clause-level constraints, splitting coordinated modal clauses.

    "must preserve the API and must not add deps" must yield *two* independent
    constraints, otherwise one satisfied clause silently masks an unmet one during
    constraint-coverage checks.
    """
    constraints: List[str] = []
    for match in re.finditer(rf"\b{_MODAL_PATTERN}\b[^.;\n]*", text, flags=re.IGNORECASE):
        span = match.group(0).strip(" ,")
        for piece in _COORDINATED_MODAL.split(span):
            piece = piece.strip(" ,")
            if piece and piece not in constraints:
                constraints.append(piece)
    return constraints

_FILE_PATTERN = re.compile(r"[\w./-]+\.(?:py|md|json|toml|yaml|yml|cfg|txt|cu|cpp|h|hpp|rs|go|ts|tsx|js)")
_BACKTICK_PATTERN = re.compile(r"`([^`\n]{1,200})`")
_IDENT_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
_TRACEBACK_FRAME = re.compile(r'File "([^"]+)", line (\d+), in ([\w.<>]+)')
_TASK_KEYWORDS = {
    "bug_fix": (
        "fix", "bug", "crash", "traceback", "error", "fails", "failing", "broken",
        "regression", "exception", "off-by-one", "wrong result",
    ),
    "code_edit": ("add", "implement", "update", "modify", "change", "rename", "remove", "patch"),
    "code_generation": ("write a function", "generate", "create a class", "scaffold"),
    "math": ("prove", "derive", "solve", "integral", "derivative", "probability", "compute"),
    "research": ("paper", "hypothesis", "experiment", "literature", "evidence", "cite"),
    "refactor": ("refactor", "clean up", "simplify", "extract method", "decouple"),
    "test_writing": ("write tests", "unit test", "pytest", "coverage"),
    "explanation": ("explain", "why does", "how does", "walk me through"),
    "question": ("what is", "how many", "which", "is it true", "?"),
}

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "when", "then",
    "than", "have", "has", "had", "are", "was", "were", "will", "would", "should",
    "could", "can", "may", "might", "must", "not", "but", "all", "any", "each",
    "you", "your", "our", "their", "its", "it", "is", "be", "to", "of", "in",
    "on", "at", "by", "as", "or", "if", "so", "we", "i", "a", "an", "do", "does",
    "please", "need", "want", "make", "get", "set", "use", "using",
}


@dataclass
class TaskSpec:
    """Structured, serializable statement of an objective and its constraints."""

    objective: str
    task_type: str = "unknown"
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    files_mentioned: List[str] = field(default_factory=list)
    symbols_mentioned: List[str] = field(default_factory=list)
    output_format: str = "unknown"
    ambiguity_flags: List[str] = field(default_factory=list)
    confidence: float = 0.5
    provenance: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TaskSpec":
        return cls(
            objective=str(payload.get("objective", "")),
            task_type=str(payload.get("task_type", "unknown")),
            requirements=list(payload.get("requirements") or []),
            constraints=list(payload.get("constraints") or []),
            entities=list(payload.get("entities") or []),
            files_mentioned=list(payload.get("files_mentioned") or []),
            symbols_mentioned=list(payload.get("symbols_mentioned") or []),
            output_format=str(payload.get("output_format", "unknown")),
            ambiguity_flags=list(payload.get("ambiguity_flags") or []),
            confidence=float(payload.get("confidence", 0.5)),
            provenance=str(payload.get("provenance", "heuristic")),
        )

    def prompt_block(self) -> str:
        """Render the spec as a compact, stable context block for Maya."""
        lines = [f"OBJECTIVE: {self.objective}", f"TASK TYPE: {self.task_type}"]
        if self.files_mentioned:
            lines.append("FILES: " + ", ".join(self.files_mentioned))
        if self.symbols_mentioned:
            lines.append("SYMBOLS: " + ", ".join(self.symbols_mentioned))
        for idx, req in enumerate(self.requirements, 1):
            lines.append(f"REQ {idx}: {req}")
        for idx, con in enumerate(self.constraints, 1):
            lines.append(f"CONSTRAINT {idx}: {con}")
        if self.output_format != "unknown":
            lines.append(f"OUTPUT FORMAT: {self.output_format}")
        if self.ambiguity_flags:
            lines.append("AMBIGUITIES: " + "; ".join(self.ambiguity_flags))
        return "\n".join(lines)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _infer_task_type(text_lower: str) -> str:
    scores: Dict[str, int] = {}
    for task_type, keywords in _TASK_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits:
            scores[task_type] = hits
    if not scores:
        return "unknown"
    # Deterministic tie-breaking by declaration order of ``_TASK_KEYWORDS``.
    order = {name: idx for idx, name in enumerate(_TASK_KEYWORDS)}
    return max(scores.items(), key=lambda kv: (kv[1], -order[kv[0]]))[0]


def _infer_output_format(text_lower: str) -> str:
    if "search/replace" in text_lower or "patch" in text_lower or "diff" in text_lower:
        return "patch"
    if "```" in text_lower or "code block" in text_lower:
        return "code_block"
    if "json" in text_lower:
        return "json"
    if "step by step" in text_lower or "steps" in text_lower or "numbered" in text_lower:
        return "numbered_list"
    if "derive" in text_lower or "proof" in text_lower:
        return "derivation"
    if "explain" in text_lower:
        return "explanation"
    if text_lower.strip().endswith("?"):
        return "short_answer"
    return "unknown"


def parse_task_spec(
    text: str,
    repo_files: Optional[Sequence[str]] = None,
    repo_symbols: Optional[Sequence[str]] = None,
) -> TaskSpec:
    """Heuristically parse an issue/prompt into a validated ``TaskSpec``.

    ``repo_files``/``repo_symbols`` (when supplied, e.g. from
    ``RepoGraphIndexer``) prune entity extraction to identifiers that actually
    exist in the repository.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("parse_task_spec requires a non-empty prompt string")

    text = text.strip()
    lower = text.lower()
    sentences = _split_sentences(text)

    objective = sentences[0][:400] if sentences else text[:400]

    constraints = _extract_constraints(lower)

    requirements: List[str] = []
    for sentence in sentences[1:6]:
        s_lower = sentence.lower()
        if any(s_lower.startswith(v) for v in ("add", "implement", "ensure", "make sure", "update", "fix", "return", "preserve", "keep")):
            if sentence not in constraints:
                requirements.append(sentence)

    files_mentioned: List[str] = []
    for match in _FILE_PATTERN.finditer(text):
        candidate = match.group(0)
        if repo_files is None or candidate in set(repo_files):
            if candidate not in files_mentioned:
                files_mentioned.append(candidate)
    for match in _TRACEBACK_FRAME.finditer(text):
        candidate = match.group(1)
        if candidate not in files_mentioned:
            files_mentioned.append(candidate)

    backticked = [m.group(1).strip() for m in _BACKTICK_PATTERN.finditer(text)]
    symbol_universe = set(repo_symbols) if repo_symbols is not None else None
    symbols_mentioned: List[str] = []
    for token in backticked:
        ident = token.split("(")[0].split(".")[-1].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ident):
            if symbol_universe is None or ident in symbol_universe:
                if ident not in symbols_mentioned:
                    symbols_mentioned.append(ident)
    if symbol_universe is not None:
        for match in _IDENT_PATTERN.finditer(text):
            ident = match.group(0)
            if ident in symbol_universe and ident not in symbols_mentioned:
                symbols_mentioned.append(ident)

    entities: List[str] = []
    for match in _IDENT_PATTERN.finditer(text):
        token = match.group(0)
        if token.lower() in _STOPWORDS:
            continue
        if token not in entities:
            entities.append(token)
        if len(entities) >= 40:
            break

    ambiguity: List[str] = []
    if len(text.split()) < 4:
        ambiguity.append("prompt_too_short")
    if re.search(r"\bor\b.*\bor\b", lower) and "either" in lower:
        ambiguity.append("multiple_alternatives")
    if "somehow" in lower or "appropriate" in lower or "as needed" in lower:
        ambiguity.append("underspecified_criterion")
    if "?" in text and len(sentences) > 1 and not files_mentioned:
        ambiguity.append("question_without_artifact")
    if not files_mentioned and any(
        kw in lower for kw in ("file", "function", "module", "class ", "repo")
    ):
        ambiguity.append("missing_target_file")

    task_type = _infer_task_type(lower)
    confidence = 0.35
    if task_type != "unknown":
        confidence += 0.25
    if files_mentioned:
        confidence += 0.15
    if constraints:
        confidence += 0.1
    if not ambiguity:
        confidence += 0.1
    confidence = float(min(0.95, confidence))

    spec = TaskSpec(
        objective=objective,
        task_type=task_type,
        requirements=requirements,
        constraints=constraints,
        entities=entities,
        files_mentioned=files_mentioned,
        symbols_mentioned=symbols_mentioned,
        output_format=_infer_output_format(lower),
        ambiguity_flags=ambiguity,
        confidence=confidence,
        provenance="heuristic",
    )
    problems = validate_task_spec(spec)
    if problems:
        spec.ambiguity_flags.extend(f"schema:{p}" for p in problems)
    return spec


def sentences_containing(text: str, offset: int) -> str:
    """Return the sentence of ``text`` containing character ``offset``."""
    start = max(
        text.rfind(".", 0, offset),
        text.rfind("\n", 0, offset),
        text.rfind(";", 0, offset),
    )
    end_candidates = [
        pos
        for pos in (text.find(".", offset), text.find("\n", offset), text.find(";", offset))
        if pos != -1
    ]
    end = min(end_candidates) if end_candidates else len(text)
    return text[start + 1 : end].strip()


def validate_task_spec(spec: TaskSpec) -> List[str]:
    """Return a list of schema problems (empty list means valid)."""
    problems: List[str] = []
    if not isinstance(spec.objective, str) or not spec.objective.strip():
        problems.append("missing_objective")
    if spec.task_type not in TASK_TYPES:
        problems.append(f"unknown_task_type:{spec.task_type}")
    if spec.output_format not in OUTPUT_FORMATS:
        problems.append(f"unknown_output_format:{spec.output_format}")
    if not 0.0 <= float(spec.confidence) <= 1.0:
        problems.append("confidence_out_of_range")
    for collection, name in (
        (spec.requirements, "requirements"),
        (spec.constraints, "constraints"),
        (spec.files_mentioned, "files_mentioned"),
        (spec.symbols_mentioned, "symbols_mentioned"),
    ):
        if not isinstance(collection, list) or any(not isinstance(x, str) for x in collection):
            problems.append(f"invalid_{name}")
    if len(set(spec.files_mentioned)) != len(spec.files_mentioned):
        problems.append("duplicate_files")
    return problems


_ALIASES = {
    "return": ("return", "returns"),
    "test": ("test", "tests", "pytest", "unittest", "coverage"),
    "preserve": ("preserve", "keep", "unchanged", "compatible", "backwards"),
    "no_new_dependency": ("no new dependenc", "without adding", "standard library", "stdlib"),
    "signature": ("signature", "argument", "parameter", "arity"),
    "complexity": ("o(", "complexity", "linear time", "constant time"),
    "keep_public_api": ("public api", "api stays", "backwards compatible"),
}


def constraint_coverage(spec: TaskSpec, answer_text: str) -> Dict[str, bool]:
    """Check whether each spec constraint is *addressed* by an answer or patch.

    This is a deterministic lexical audit, not a truth judgement: it exists so a
    reasoning state can flag constraints that were never mentioned at all before
    claiming completion.
    """
    if not isinstance(answer_text, str):
        raise TypeError("answer_text must be a string")
    answer_lower = answer_text.lower()
    coverage: Dict[str, bool] = {}
    for idx, constraint in enumerate(spec.constraints):
        c_lower = constraint.lower()
        key_terms = [t for t in re.findall(r"[a-z_]{4,}", c_lower) if t not in _STOPWORDS]
        covered = any(term in answer_lower for term in key_terms) if key_terms else False
        if not covered:
            for alias_key, alias_terms in _ALIASES.items():
                if alias_key in c_lower and any(term in answer_lower for term in alias_terms):
                    covered = True
                    break
        coverage[f"constraint_{idx}"] = bool(covered)
    for idx, requirement in enumerate(spec.requirements):
        r_lower = requirement.lower()
        key_terms = [t for t in re.findall(r"[a-z_]{4,}", r_lower) if t not in _STOPWORDS]
        coverage[f"requirement_{idx}"] = bool(
            any(term in answer_lower for term in key_terms) if key_terms else False
        )
    return coverage


class NeuralTaskSpecParser:
    """Optional trainable TaskSpec head over the shared Maya trunk.

    Two lightweight outputs are produced from a pooled trunk hidden state:

    * ``task_type`` logits over :data:`TASK_TYPES`
    * a scalar ``ambiguity`` logit

    The module is intentionally tiny (``d * len(TASK_TYPES) + d`` parameters) and
    is *not* used unless it has been trained and validated on held-out
    understanding data; ``parse_task_spec`` is the default at inference time.
    """

    def __init__(self, d: int = 768, dropout: float = 0.0):
        self.d = int(d)
        self.dropout = float(dropout)
        self.impl = None
        self._dirty = True

    @property
    def n_parameters(self) -> int:
        return self.d * len(TASK_TYPES) + self.d

    def build(self):
        """Materialise the torch module (lazy so CPU-only workers can import)."""
        import torch
        import torch.nn as nn

        d = self.d

        class _Impl(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.task_head = nn.Linear(d, len(TASK_TYPES), bias=True)
                self.ambiguity_head = nn.Linear(d, 1, bias=True)
                self.drop = nn.Dropout(dropout)

            def forward(self, hidden):  # hidden: [b, d]
                pooled = self.drop(hidden)
                return {
                    "task_type_logits": self.task_head(pooled),
                    "ambiguity_logit": self.ambiguity_head(pooled).squeeze(-1),
                }

        self.impl = _Impl()
        self._dirty = False
        return self.impl

    def predict(self, hidden) -> TaskSpec:
        """Predict a ``TaskSpec`` from a pooled hidden state (requires ``build``)."""
        import torch

        if self.impl is None:
            self.build()
        self.impl.eval()
        with torch.no_grad():
            out = self.impl(hidden)
            idx = int(torch.argmax(out["task_type_logits"], dim=-1).item())
            ambiguity = float(torch.sigmoid(out["ambiguity_logit"]).item())
        return TaskSpec(
            objective="",
            task_type=TASK_TYPES[idx],
            ambiguity_flags=["neural_ambiguity_flag"] if ambiguity > 0.5 else [],
            confidence=float(1.0 - ambiguity),
            provenance="neural",
        )
