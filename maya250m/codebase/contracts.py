"""Full Python calling-convention contracts.

The v2 implementation approximated a function as ``(min_arity, max_arity)``. That
is wrong in ways that matter when a model edits real code:

* ``**kwargs`` permits extra *keywords* but never extra *positional* args, yet it
  was treated as unbounded positional arity;
* positional-only, keyword-only, ``*args`` and defaults were collapsed into one
  number;
* ``self``/``cls`` binding depended on whether a method was static;
* functions with the same unqualified name in different classes were merged.

This module models the real convention with :class:`ParameterContract` and
:class:`CallableContract`, keeps overload candidates instead of overwriting them,
labels confidence (``exact_ast`` / ``partial_ast`` / ``inferred`` / ``unknown``),
and returns ``UNKNOWN_DYNAMIC_DISPATCH`` rather than guessing when a call cannot be
checked statically.

:func:`check_call` is deliberately conservative: it never rejects a call merely
because the call site uses ``**kwargs``, and it never treats ``**kwargs`` in the
callee as positional varargs.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "POS_ONLY",
    "POS_OR_KEYWORD",
    "VAR_POSITIONAL",
    "KEYWORD_ONLY",
    "VAR_KEYWORD",
    "ParameterContract",
    "CallableContract",
    "CallCheck",
    "contract_from_node",
    "build_contracts",
    "contract_from_source",
    "signature_table_from_source",
    "header_contract_from_text",
    "check_call",
    "UNKNOWN_DYNAMIC_DISPATCH",
]

POS_ONLY = "pos_only"
POS_OR_KEYWORD = "pos_or_keyword"
VAR_POSITIONAL = "var_positional"
KEYWORD_ONLY = "keyword_only"
VAR_KEYWORD = "var_keyword"

UNKNOWN_DYNAMIC_DISPATCH = "unknown_dynamic_dispatch"

_PARAMETER_KINDS = (POS_ONLY, POS_OR_KEYWORD, VAR_POSITIONAL, KEYWORD_ONLY, VAR_KEYWORD)


@dataclass(frozen=True)
class ParameterContract:
    """One parameter, with its real calling convention."""

    name: str
    kind: str
    required: bool
    annotation: Optional[str] = None
    default_repr: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in _PARAMETER_KINDS:
            raise ValueError(f"unknown parameter kind {self.kind!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "required": bool(self.required),
            "annotation": self.annotation,
            "default_repr": self.default_repr,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ParameterContract":
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            required=bool(payload.get("required", False)),
            annotation=payload.get("annotation"),
            default_repr=payload.get("default_repr"),
        )


@dataclass(frozen=True)
class CallableContract:
    """A complete callable signature with binding, overloads and confidence."""

    symbol_id: str
    parameters: Tuple[ParameterContract, ...] = ()
    return_annotation: Optional[str] = None
    binding: str = "function"  # function | instance | class | static
    overload_group: Optional[str] = None
    confidence: str = "exact_ast"  # exact_ast | partial_ast | inferred | unknown

    def __post_init__(self) -> None:
        if self.binding not in ("function", "instance", "class", "static"):
            raise ValueError(f"unknown binding {self.binding!r}")
        if self.confidence not in ("exact_ast", "partial_ast", "inferred", "unknown"):
            raise ValueError(f"unknown confidence {self.confidence!r}")

    # ------------------------------------------------------------- structure

    @property
    def positional_parameters(self) -> Tuple[ParameterContract, ...]:
        return tuple(p for p in self.parameters if p.kind in (POS_ONLY, POS_OR_KEYWORD))

    @property
    def has_var_positional(self) -> bool:
        return any(p.kind == VAR_POSITIONAL for p in self.parameters)

    @property
    def has_var_keyword(self) -> bool:
        return any(p.kind == VAR_KEYWORD for p in self.parameters)

    def positional_bounds(self) -> Tuple[int, Optional[int]]:
        """``(minimum, maximum)`` positional args; ``None`` maximum means unbounded."""
        positional = self.positional_parameters
        minimum = sum(1 for p in positional if p.required)
        maximum: Optional[int] = None if self.has_var_positional else len(positional)
        return minimum, maximum

    def accepts_keyword(self, name: str) -> bool:
        if self.has_var_keyword:
            return True
        return any(
            p.name == name and p.kind in (POS_OR_KEYWORD, KEYWORD_ONLY)
            for p in self.parameters
        )

    def required_keyword_names(self) -> List[str]:
        return [
            p.name
            for p in self.parameters
            if p.kind == KEYWORD_ONLY and p.required
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "parameters": [p.to_dict() for p in self.parameters],
            "return_annotation": self.return_annotation,
            "binding": self.binding,
            "overload_group": self.overload_group,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CallableContract":
        return cls(
            symbol_id=str(payload.get("symbol_id", "")),
            parameters=tuple(
                ParameterContract.from_dict(p) for p in payload.get("parameters") or []
            ),
            return_annotation=payload.get("return_annotation"),
            binding=str(payload.get("binding", "function")),
            overload_group=payload.get("overload_group"),
            confidence=str(payload.get("confidence", "exact_ast")),
        )

    def to_symbol_table_entry(self) -> Dict[str, Any]:
        """Entry shape consumed by Vexx's in-flight masker (keeps v2 keys)."""
        minimum, maximum = self.positional_bounds()
        return {
            "min_arity": minimum,
            "max_arity": maximum,
            "return_type": self.return_annotation or "any",
            "binding": self.binding,
            "confidence": self.confidence,
            "keyword_only": [
                p.name for p in self.parameters if p.kind == KEYWORD_ONLY
            ],
            "required_keywords": self.required_keyword_names(),
            "has_var_positional": self.has_var_positional,
            "has_var_keyword": self.has_var_keyword,
        }

    @classmethod
    def unknown(cls, symbol_id: str, reason: str = "") -> "CallableContract":
        return cls(
            symbol_id=symbol_id,
            parameters=(),
            return_annotation=None,
            binding="function",
            confidence="unknown",
        )


@dataclass(frozen=True)
class CallCheck:
    """Result of statically checking one call site against a contract."""

    status: str
    message: str = ""
    missing_positional: Tuple[str, ...] = ()
    missing_keyword: Tuple[str, ...] = ()
    unexpected_keyword: Tuple[str, ...] = ()
    duplicate: Tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "ok"

    @property
    def kind(self) -> str:
        if self.status in ("too_few_positional", "too_many_positional"):
            return "positional"
        if self.status in ("missing_keyword", "unexpected_keyword", "duplicate_argument"):
            return "keyword"
        return "unknown" if self.status == UNKNOWN_DYNAMIC_DISPATCH else "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "kind": self.kind,
            "message": self.message,
            "missing_positional": list(self.missing_positional),
            "missing_keyword": list(self.missing_keyword),
            "unexpected_keyword": list(self.unexpected_keyword),
            "duplicate": list(self.duplicate),
        }


# --------------------------------------------------------------------- building


def _decorator_names(node: ast.AST) -> List[str]:
    names: List[str] = []
    for decorator in getattr(node, "decorator_list", []) or []:
        try:
            names.append(ast.unparse(decorator))
        except Exception:  # pragma: no cover - ast.unparse failures are rare
            names.append("")
    return names


def _binding_for(class_prefix: str, decorators: Sequence[str]) -> str:
    simple = [d.split("(")[0].strip().split(".")[-1] for d in decorators]
    if "staticmethod" in simple:
        return "static"
    if "classmethod" in simple:
        return "class"
    return "instance" if class_prefix else "function"


def _parameter_list(node: ast.AST, binding: str) -> Tuple[ParameterContract, ...]:
    args = getattr(node, "args", None)
    if args is None:
        return ()

    def render(annotation: Optional[ast.AST]) -> Optional[str]:
        if annotation is None:
            return None
        try:
            return ast.unparse(annotation)
        except Exception:  # pragma: no cover
            return "?"

    params: List[ParameterContract] = []
    positional = list(getattr(args, "posonlyargs", []) or []) + list(args.args or [])
    defaults = list(getattr(args, "defaults", []) or [])
    n_required = len(positional) - len(defaults)
    n_posonly = len(getattr(args, "posonlyargs", []) or [])

    for index, arg in enumerate(positional):
        if index == 0 and binding in ("instance", "class") and arg.arg in ("self", "cls"):
            continue
        default_node = defaults[index - n_required] if index >= n_required else None
        params.append(
            ParameterContract(
                name=arg.arg,
                kind=POS_ONLY if index < n_posonly else POS_OR_KEYWORD,
                required=default_node is None,
                annotation=render(arg.annotation),
                default_repr=(
                    None
                    if default_node is None
                    else _safe_unparse(default_node)
                ),
            )
        )

    vararg = getattr(args, "vararg", None)
    if vararg is not None:
        params.append(
            ParameterContract(
                name=vararg.arg,
                kind=VAR_POSITIONAL,
                required=False,
                annotation=render(vararg.annotation),
            )
        )

    for arg, default in zip(
        getattr(args, "kwonlyargs", []) or [], getattr(args, "kw_defaults", []) or []
    ):
        params.append(
            ParameterContract(
                name=arg.arg,
                kind=KEYWORD_ONLY,
                required=default is None,
                annotation=render(arg.annotation),
                default_repr=None if default is None else _safe_unparse(default),
            )
        )

    kwarg = getattr(args, "kwarg", None)
    if kwarg is not None:
        params.append(
            ParameterContract(
                name=kwarg.arg,
                kind=VAR_KEYWORD,
                required=False,
                annotation=render(kwarg.annotation),
            )
        )
    return tuple(params)


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover
        return "..."


def contract_from_node(
    node: ast.AST,
    module: str = "<module>",
    class_prefix: str = "",
    symbol_id: Optional[str] = None,
) -> CallableContract:
    """Build an exact contract from a parsed ``FunctionDef``/``AsyncFunctionDef``."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise TypeError("contract_from_node requires a function definition")
    name = getattr(node, "name", "<lambda>")
    qualname = f"{class_prefix}{name}"
    decorators = _decorator_names(node)
    binding = _binding_for(class_prefix, decorators)
    is_overload = any(d.split("(")[0].strip().split(".")[-1] == "overload" for d in decorators)
    return CallableContract(
        symbol_id=symbol_id or f"{module}::{qualname}",
        parameters=_parameter_list(node, binding),
        return_annotation=(
            _safe_unparse(node.returns) if getattr(node, "returns", None) is not None else None
        ),
        binding=binding,
        overload_group=f"{module}::{qualname}" if is_overload else None,
        confidence="exact_ast",
    )


def build_contracts(
    source: str, module: str = "<module>"
) -> Dict[str, List[CallableContract]]:
    """All callable contracts in ``source``; overloads are kept as separate entries."""
    tree = ast.parse(source)
    contracts: Dict[str, List[CallableContract]] = {}

    def visit(body: Sequence[ast.stmt], class_prefix: str = "") -> None:
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                visit(stmt.body, class_prefix=f"{class_prefix}{stmt.name}.")
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                contract = contract_from_node(stmt, module, class_prefix)
                qualname = f"{class_prefix}{stmt.name}"
                contracts.setdefault(qualname, []).append(contract)
                visit(stmt.body, class_prefix=class_prefix)

    visit(tree.body)
    return contracts


def contract_from_source(
    source: str, qualname: Optional[str] = None, module: str = "<source>"
) -> Optional[CallableContract]:
    """Contract for one callable in a complete source string, or ``None``."""
    try:
        contracts = build_contracts(source, module)
    except SyntaxError:
        return None
    if qualname is None:
        if len(contracts) != 1:
            return None
        return next(iter(contracts.values()))[-1]
    candidates = contracts.get(qualname) or [
        value[-1] for key, value in contracts.items() if key.endswith("." + qualname)
    ]
    if not candidates:
        return None
    implemented = [c for c in candidates if c.overload_group is None]
    return (implemented or candidates)[-1]


def signature_table_from_source(source: str) -> Dict[str, Dict[str, Any]]:
    """``name -> {min_arity, max_arity, return_type}`` (widest bounds per name)."""
    table: Dict[str, Dict[str, Any]] = {}
    for qualname, contracts in build_contracts(source).items():
        name = qualname.rsplit(".", 1)[-1]
        bounds = [c.positional_bounds() for c in contracts]
        minimum = min(b[0] for b in bounds)
        maximum = (
            None
            if any(b[1] is None for b in bounds)
            else max(int(b[1]) for b in bounds)
        )
        entry = {
            "min_arity": minimum,
            "max_arity": maximum,
            "return_type": contracts[-1].return_annotation or "any",
            "keyword_only": sorted(
                {p.name for c in contracts for p in c.parameters if p.kind == KEYWORD_ONLY}
            ),
            "required_keywords": [
                p.name
                for c in contracts
                for p in c.parameters
                if p.kind == KEYWORD_ONLY and p.required
            ],
            "has_var_positional": any(c.has_var_positional for c in contracts),
            "has_var_keyword": any(c.has_var_keyword for c in contracts),
            "confidence": (
                "exact_ast"
                if all(c.confidence == "exact_ast" for c in contracts)
                else "partial_ast"
            ),
        }
        if name in table:
            existing = table[name]
            entry["min_arity"] = min(existing["min_arity"], entry["min_arity"])
            entry["max_arity"] = (
                None
                if existing["max_arity"] is None or entry["max_arity"] is None
                else max(existing["max_arity"], entry["max_arity"])
            )
        table[name] = entry
    return table


# ------------------------------------------------------------- partial headers

_DEF_RE = re.compile(r"(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)\s*\(")
_PAIRS = {"(": ")", "[": "]", "{": "}"}


def _extract_balanced(text: str, open_index: int) -> Tuple[str, int]:
    """Return ``(inside, index_after_close)`` for the bracket starting at ``open_index``."""
    opener = text[open_index]
    stack: List[str] = []
    index = open_index
    quote: Optional[str] = None
    triple = False
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == "\\":
                index += 2
                continue
            if triple:
                if text.startswith(quote * 3, index):
                    quote = None
                    triple = False
                    index += 3
                    continue
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in ("'", '"'):
            if text.startswith(char * 3, index):
                quote, triple = char, True
                index += 3
                continue
            quote, triple = char, False
            index += 1
            continue
        if char in _PAIRS:
            stack.append(_PAIRS[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return text[open_index + 1 : index], index + 1
        index += 1
    raise ValueError("unbalanced parentheses in function header")


def _split_top_level(params: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    quote: Optional[str] = None
    for char in params:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            current.append(char)
            continue
        if char in _PAIRS:
            depth += 1
        elif char in (")", "]", "}"):
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _fallback_parameters(params_text: str) -> Tuple[ParameterContract, ...]:
    params: List[ParameterContract] = []
    for raw in _split_top_level(params_text):
        if not raw or raw in ("/", "*"):
            continue
        kind = POS_OR_KEYWORD
        if raw.startswith("**"):
            kind, raw = VAR_KEYWORD, raw[2:]
        elif raw.startswith("*"):
            kind, raw = VAR_POSITIONAL, raw[1:]
        name, _, default = raw.partition("=")
        name = name.partition(":")[0].strip()
        if not name:
            continue
        params.append(
            ParameterContract(
                name=name,
                kind=kind,
                required=(kind in (POS_OR_KEYWORD, POS_ONLY) and not default),
                default_repr=default.strip() or None,
            )
        )
    return tuple(params)


def header_contract_from_text(text: str, symbol_id: str = "<partial>") -> Optional[CallableContract]:
    """Contract for an in-flight (possibly incomplete) function header.

    Uses a token-aware balanced-delimiter scan, not a regular expression over the
    whole header, and labels the result ``partial_ast`` so callers never treat it
    as exact.
    """
    match = _DEF_RE.search(text)
    if not match:
        return None
    name = match.group("name")
    open_index = match.end() - 1
    try:
        params_text, after = _extract_balanced(text, open_index)
    except ValueError:
        # Truncated mid-signature (the model is still generating the header):
        # recover whatever parameters exist and label them partial, never exact.
        params_text = text[open_index + 1 :]
        after = len(text)

    return_text = ""
    rest = text[after:]
    if rest.lstrip().startswith("->"):
        arrow = rest.index("->")
        tail = rest[arrow + 2 :]
        colon = tail.find(":")
        return_text = (tail if colon == -1 else tail[:colon]).strip()

    stub = f"def {name}({params_text})" + (f" -> {return_text}" if return_text else "") + ":\n    pass\n"
    try:
        tree = ast.parse(stub)
        node = tree.body[0]
        contract = contract_from_node(node, module=symbol_id)
        return CallableContract(
            symbol_id=contract.symbol_id,
            parameters=contract.parameters,
            return_annotation=contract.return_annotation,
            binding=contract.binding,
            overload_group=None,
            confidence="partial_ast",
        )
    except SyntaxError:
        return CallableContract(
            symbol_id=f"{symbol_id}::{name}",
            parameters=_fallback_parameters(params_text),
            return_annotation=return_text or None,
            confidence="partial_ast",
        )


# ------------------------------------------------------------------- checking


def check_call(
    contract: CallableContract,
    n_positional: int = 0,
    keyword_names: Sequence[str] = (),
    has_starargs: bool = False,
    has_kwargs: bool = False,
) -> CallCheck:
    """Statically check one call site against a contract (conservative)."""
    if contract.confidence == "unknown":
        return CallCheck(
            status=UNKNOWN_DYNAMIC_DISPATCH,
            message="signature could not be recovered statically",
        )

    keywords = [k for k in keyword_names if k]
    positional = contract.positional_parameters
    minimum, maximum = contract.positional_bounds()

    duplicates = tuple(
        param.name
        for index, param in enumerate(positional)
        if index < int(n_positional) and param.name in keywords
    )
    if duplicates:
        return CallCheck(
            status="duplicate_argument",
            message=f"argument(s) passed both positionally and by keyword: {list(duplicates)}",
            duplicate=duplicates,
        )

    if not has_starargs:
        if maximum is not None and int(n_positional) > maximum:
            return CallCheck(
                status="too_many_positional",
                message=(
                    f"call passes {n_positional} positional argument(s) but the contract "
                    f"allows at most {maximum}"
                ),
            )
        if int(n_positional) < minimum:
            missing = tuple(
                param.name
                for index, param in enumerate(positional)
                if index >= int(n_positional)
                and param.required
                and param.name not in keywords
            )
            if missing:
                if has_kwargs:
                    return CallCheck(
                        status="ok",
                        message="required positional arguments may be supplied via **kwargs",
                    )
                return CallCheck(
                    status="too_few_positional",
                    message=(
                        f"call passes {n_positional} positional argument(s) but requires "
                        f"at least {minimum}"
                    ),
                    missing_positional=missing,
                )

    if not has_kwargs:
        unexpected = tuple(name for name in keywords if not contract.accepts_keyword(name))
        if unexpected:
            return CallCheck(
                status="unexpected_keyword",
                message=f"unexpected keyword argument(s): {list(unexpected)}",
                unexpected_keyword=unexpected,
            )
        missing_keyword = tuple(
            name for name in contract.required_keyword_names() if name not in keywords
        )
        if missing_keyword:
            return CallCheck(
                status="missing_keyword",
                message=f"missing required keyword-only argument(s): {list(missing_keyword)}",
                missing_keyword=missing_keyword,
            )

    return CallCheck(status="ok")
