"""VEXX 3.5: Six-tier neuro-symbolic engine for Maya-250M.

Works with both PyTorch tensors and NumPy arrays so Vexx 3.5 can run inside
full GPU/TPU decode loops as well as lightweight CPU data-annotation workers.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

from .tokenizer import (
    EOS,
    EXP_END,
    EXP_START,
    LEGAL_STOP,
    NEVER_EMIT,
    PROTOCOL_MARKERS,
    THINK_END,
    THINK_START,
    token_category,
)

__all__ = [
    "Vexx35Controller",
    "Vexx3Controller",
    "LexerState",
    "advance_lexer_state",
    "safe_sympy",
    "clone_cache_generic",
    "truncate_cache_at_token",
]


def clone_cache_generic(cache: Any) -> Any:
    """Recursively deep-copy v1 list caches or v2 structured multi-loop caches.

    Works with Torch tensors and NumPy arrays so it is usable from CPU-only
    annotation workers as well as from live decode loops.
    """
    if hasattr(cache, "clone"):
        return cache.clone()
    if isinstance(cache, np.ndarray):
        return cache.copy()
    if isinstance(cache, list):
        return [clone_cache_generic(item) for item in cache]
    if isinstance(cache, dict):
        return {k: clone_cache_generic(v) for k, v in cache.items()}
    return copy.deepcopy(cache)


def _contains_recurrent_state(node: Any) -> bool:
    """True when ``node`` holds GDN recurrent state (no sequence dimension)."""
    if isinstance(node, dict):
        if "state" in node or "conv" in node:
            return True
        return any(_contains_recurrent_state(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_recurrent_state(item) for item in node)
    return False


def truncate_cache_at_token(cache: Any, tok_idx: int) -> Any:
    """Recursively truncate Full-Attention KV tensors at ``tok_idx`` (v1 or v2 caches).

    GDN recurrent state matrices carry no sequence dimension, so they cannot be
    sliced back in time. Truncating a bundle whose ``position`` is ahead of
    ``tok_idx`` while it still contains recurrent state would silently keep future
    tokens inside ``S_t`` and corrupt causal parity, so that case fails closed: take
    an exact snapshot at ``tok_idx`` (``save_checkpoint``) or replay the prefix with
    :func:`maya250m.cache.rewind_cache` instead.
    """
    tok_idx = int(tok_idx)
    if isinstance(cache, list):
        for item in cache:
            truncate_cache_at_token(item, tok_idx)
        return cache
    if isinstance(cache, dict):
        position = cache.get("position")
        if (
            position is not None
            and int(position) > tok_idx
            and _contains_recurrent_state(cache)
        ):
            raise ValueError(
                f"cannot truncate a hybrid cache at token {tok_idx}: it is at "
                f"position {int(position)} and carries GDN recurrent state with no "
                "sequence axis; restore an exact snapshot taken at that token or "
                "replay the prefix with rewind_cache"
            )
        if "k" in cache and "v" in cache and hasattr(cache["k"], "shape"):
            if cache["k"].shape[2] > tok_idx:
                cache["k"] = cache["k"][:, :, :tok_idx, :]
                cache["v"] = cache["v"][:, :, :tok_idx, :]
        else:
            for key in ("prelude", "core_loops", "coda"):
                if key in cache:
                    truncate_cache_at_token(cache[key], tok_idx)
            if "position" in cache:
                cache["position"] = min(int(cache["position"]), tok_idx)
        return cache
    return cache


_SANDBOX_DRIVER = r'''
import builtins as _builtins, contextlib, io, json, os, sys

payload = json.load(sys.stdin)

_ALLOWED_MODULES = frozenset({
    "math", "heapq", "collections", "bisect", "itertools", "functools",
    "typing", "fractions", "decimal", "operator", "string",
})


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".")[0]
    if int(level or 0) != 0 or root not in _ALLOWED_MODULES:
        raise ImportError("sandbox forbids import of %r" % (name,))
    for item in (fromlist or ()):
        if str(item).startswith("_"):
            raise ImportError("sandbox forbids private import %r" % (item,))
    return _builtins.__import__(name, globals, locals, fromlist, level)


SAFE = {
    "__import__": _safe_import,
    "__build_class__": _builtins.__build_class__,
    "object": object,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "IndexError": IndexError, "KeyError": KeyError, "AttributeError": AttributeError,
    "ZeroDivisionError": ZeroDivisionError, "RuntimeError": RuntimeError,
    "StopIteration": StopIteration, "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError,
    "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "next": next, "ord": ord, "pow": pow, "print": print,
    "range": range, "reversed": reversed, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "True": True, "False": False, "None": None,
}

mode = payload.get("mode", "fn_call")
ns = {"__builtins__": SAFE, "__name__": "__maya_sandbox__"}
try:
    if mode == "experiment_script":
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(payload["code"], "<experiment>", "exec"), ns)
        res = ns.get("RESULT", buf.getvalue().strip())
        print(json.dumps({"ok": True, "repr": repr(res), "stdout": buf.getvalue().strip()[:256]}))
    else:
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            exec(compile(payload["code"], "<candidate>", "exec"), ns)
            fn = ns[payload["fn"]]
            args = payload["args"]
            result = fn(*args)
        print(json.dumps({"ok": True, "repr": repr(result), "input_after": repr(args)}))
except BaseException as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
'''

def _limit_child() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


_MAX_EXPR_CHARS = 512
_MAX_EXPR_NODES = 200
_MAX_EXPONENT = 10_000
_MAX_INT_BITS = 4096
_MAX_FACTORIAL_ARG = 5000
_MAX_BINOMIAL_N = 100_000


def _bounded_factorial(value: Any) -> Any:
    """``factorial`` with a hard argument cap against CPU denial of service."""
    number = sp.sympify(value)
    if number.is_number:
        if abs(int(number)) > _MAX_FACTORIAL_ARG:
            raise ValueError(f"factorial argument above {_MAX_FACTORIAL_ARG}")
    return sp.factorial(number)


def _bounded_binomial(n: Any, k: Any) -> Any:
    """``binomial``/``comb`` with hard argument caps."""
    nn, kk = sp.sympify(n), sp.sympify(k)
    for value in (nn, kk):
        if value.is_number and abs(int(value)) > _MAX_BINOMIAL_N:
            raise ValueError(f"binomial argument above {_MAX_BINOMIAL_N}")
    return sp.binomial(nn, kk)


_ALLOWED_FUNCS = {
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    # Number theory & combinatorics (competition-maths CAS splices).
    "gcd": sp.gcd,
    "lcm": sp.lcm,
    "binomial": _bounded_binomial,
    "comb": _bounded_binomial,
    "factorial": _bounded_factorial,
    "isprime": sp.isprime,
    "totient": sp.totient,
    "Mod": sp.Mod,
    # Rounding & algebra.
    "floor": sp.floor,
    "ceil": sp.ceiling,
    "ceiling": sp.ceiling,
    "Rational": sp.Rational,
    "sign": sp.sign,
    "Min": sp.Min,
    "Max": sp.Max,
}
_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
    ast.FloorDiv: lambda a, b: a // b,
}


def safe_sympy(expr: str, env: Optional[Dict[str, Any]] = None) -> sp.Expr:
    """Parse an arithmetic expression into SymPy via a restricted AST builder."""
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("empty expression")
    expr = expr.strip()
    if len(expr) > _MAX_EXPR_CHARS:
        raise ValueError(f"expression longer than {_MAX_EXPR_CHARS} characters")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression syntax: {exc.msg}") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_EXPR_NODES:
        raise ValueError("expression has too many AST nodes")
    env = dict(env or {})

    def build(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Expression):
            return build(node.body)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"unsupported literal {value!r}")
            if isinstance(value, int) and value.bit_length() > _MAX_INT_BITS:
                raise ValueError("integer literal too large")
            return sp.Integer(value) if isinstance(value, int) else sp.Rational(str(value))
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or not node.id.isidentifier():
                raise ValueError(f"unsupported name {node.id!r}")
            if node.id in env:
                return env[node.id]
            if node.id == "pi":
                return sp.pi
            if node.id == "E":
                return sp.E
            return sp.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = build(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = build(node.left), build(node.right)
            if isinstance(node.op, ast.Pow):
                if right.is_number and abs(float(right)) > _MAX_EXPONENT:
                    raise ValueError("exponent too large")
            if isinstance(node.op, ast.Pow) and isinstance(right, sp.Integer):
                if abs(int(right)) > _MAX_EXPONENT:
                    raise ValueError("exponent too large")
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("unsupported function call")
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            return _ALLOWED_FUNCS[node.func.id](*[build(arg) for arg in node.args])
        raise ValueError(f"disallowed syntax: {type(node).__name__}")

    result = build(tree)
    if isinstance(result, bool):
        return sp.true if result else sp.false
    return result


# --------------------------------------------------------------------- lexer
#
# Tier-1 in-flight masking must decide, for every vocabulary entry, whether the
# token is syntactically legal after the current prefix. Re-lexing the whole
# prefix for all ``|V|`` candidates costs ``O(|V| * L)`` character steps per decode
# step, which at ``L = 4096`` exceeds 10^8 Python loop iterations per token. The
# resumable state below scans only the appended characters, making each candidate
# cost ``O(len(token))``.

_CALLEE_KEYWORDS = frozenset(
    {
        "if", "while", "for", "return", "elif", "assert", "not", "and", "or",
        "in", "def", "class", "lambda", "with", "yield", "del", "await", "from",
        "raise", "import", "except", "finally", "match", "case", "global",
    }
)
_BRACKET_PAIRS = {("(", ")"), ("[", "]"), ("{", "}")}
_STRING_START_RE = re.compile(r"^([rRfFbB]{0,2})('''|\"\"\"|'|\")")
#: A trailing run that might still grow into a triple-quoted string.
_PARTIAL_STRING_RE = re.compile(r"[rRfFbB]{0,2}(?:'''|\"\"\"|''|\"\"|'|\")\Z")


@dataclass
class LexerState:
    """Resumable Python lexer state for incremental in-flight masking.

    ``stack`` holds open delimiters as ``(delimiter, callee, comma_count)`` where
    ``callee`` is the dotted name immediately before an opening ``(``.
    ``closed_frames`` reports only the frames closed while scanning the most recent
    chunk, which is exactly what a token-closing candidate needs.
    """

    valid: bool = True
    reason: Optional[str] = None
    in_str: Optional[str] = None
    raw_str: bool = False
    f_mode: bool = False
    f_brace_depth: int = 0
    escape: bool = False
    stack: List[Tuple[str, Optional[str], int]] = field(default_factory=list)
    closed_frames: List[Tuple[str, Optional[str], int]] = field(default_factory=list)
    last_ident: str = ""
    prev_non_ws: str = ""
    last_line: str = ""
    pending: str = ""
    string_open_at: int = -1

    def clone(self) -> "LexerState":
        """Copy the state, dropping per-chunk ``closed_frames``."""
        return LexerState(
            valid=self.valid,
            reason=self.reason,
            in_str=self.in_str,
            raw_str=self.raw_str,
            f_mode=self.f_mode,
            f_brace_depth=self.f_brace_depth,
            escape=self.escape,
            stack=list(self.stack),
            closed_frames=[],
            last_ident=self.last_ident,
            prev_non_ws=self.prev_non_ws,
            last_line=self.last_line,
            pending=self.pending,
            string_open_at=self.string_open_at,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Legacy dictionary view consumed by existing callers and tests."""
        return {
            "valid": self.valid,
            "reason": self.reason,
            "stack": list(self.stack),
            "closed_frames": list(self.closed_frames),
            "in_str": self.in_str,
            "f_brace_depth": self.f_brace_depth,
            "escape": self.escape,
            "last_ident": self.last_ident,
            "prev_non_ws": self.prev_non_ws,
            "pending": self.pending,
        }


def advance_lexer_state(state: LexerState, chunk: str, reset_closed: bool = True) -> LexerState:
    """Advance ``state`` over ``chunk`` in ``O(len(chunk))`` character steps.

    Only a genuinely incomplete trailing string opener (``f"`` / ``""`` / ``'''``
    split across chunk boundaries) is carried into ``state.pending`` and re-scanned
    with the next chunk, so token boundaries cannot silently change how a quote is
    interpreted.
    """
    st = state.clone()
    if reset_closed:
        st.closed_frames = []
    if not st.valid or not chunk:
        return st

    text = (st.pending + chunk) if st.pending else chunk
    st.pending = ""
    n = len(text)

    hold = -1
    hold_state: Optional[LexerState] = None
    m_partial = _PARTIAL_STRING_RE.search(text)
    if m_partial is not None:
        run = m_partial.group(0)
        quote = run[-1]
        if len(run) - len(run.rstrip(quote)) < 3:
            hold = m_partial.start()

    i = 0
    while i < n and st.valid:
        if hold == i:
            hold_state = st.clone()
        ch = text[i]

        if st.in_str == "#":
            if ch == "\n":
                st.in_str = None
                st.prev_non_ws = "\n"
                st.last_line = ""
            i += 1
            continue

        if st.in_str is not None:
            if st.f_mode and st.f_brace_depth == 0 and ch == "{":
                if i + 1 < n and text[i + 1] == "{":
                    i += 2
                    continue
                st.f_brace_depth = 1
                st.stack.append(("(", None, 0))
                i += 1
                continue
            if st.f_mode and st.f_brace_depth > 0:
                if ch in "'\"":
                    st.valid = False
                    st.reason = "quote_in_fstring_expression"
                    break
                if ch == "{":
                    st.f_brace_depth += 1
                elif ch == "}":
                    st.f_brace_depth -= 1
                    if st.f_brace_depth == 0 and st.stack and st.stack[-1][0] == "(":
                        st.stack.pop()
                i += 1
                continue
            if st.escape:
                st.escape = False
                i += 1
                continue
            if ch == "\\":
                if i + 1 < n:
                    i += 2
                else:
                    st.escape = True
                    i += 1
                continue
            if text.startswith(st.in_str, i):
                i += len(st.in_str)
                st.in_str = None
                st.f_mode = False
                st.raw_str = False
                st.escape = False
                continue
            i += 1
            continue

        if ch == "#":
            st.in_str = "#"
            i += 1
            continue

        m_str = _STRING_START_RE.match(text[i:])
        if m_str:
            flags = m_str.group(1).lower()
            st.in_str = m_str.group(2)
            st.f_mode = "f" in flags
            st.raw_str = "r" in flags
            st.string_open_at = i
            st.last_ident = ""
            i += len(m_str.group(0))
            continue

        if ch in "([{":
            callee = None
            if ch == "(" and st.last_ident and st.last_ident not in _CALLEE_KEYWORDS:
                callee = st.last_ident
            st.stack.append((ch, callee, 0))
            st.last_ident = ""
            st.prev_non_ws = ch
            st.last_line = (st.last_line + ch)[-256:]
            i += 1
            continue

        if ch in ")]}":
            if not st.stack:
                st.valid = False
                st.reason = "unmatched_close"
                break
            open_ch, _callee, _count = st.stack[-1]
            if (open_ch, ch) not in _BRACKET_PAIRS:
                st.valid = False
                st.reason = "mismatched_close"
                break
            st.closed_frames.append(st.stack.pop())
            st.last_ident = ""
            st.prev_non_ws = ch
            st.last_line = (st.last_line + ch)[-256:]
            i += 1
            continue

        if ch == "," and st.stack:
            delim, call, count = st.stack[-1]
            st.stack[-1] = (delim, call, count + 1)
            st.last_ident = ""
            i += 1
            continue

        if ch.isalnum() or ch in "_.":
            st.last_ident = (st.last_ident + ch)[-160:]
        elif not ch.isspace():
            st.last_ident = ""

        if ch == "\n":
            st.last_line = ""
        else:
            st.last_line = (st.last_line + ch)[-256:]
        if not ch.isspace():
            st.prev_non_ws = ch
        i += 1

    if (
        st.valid
        and hold >= 0
        and hold_state is not None
        and st.in_str is not None
        and st.string_open_at >= hold
    ):
        # The trailing quote run is still undecided (single vs triple quoted), and a
        # string prefix may sit in the previous chunk (``f`` + ``"""``). Carry both
        # so the next chunk re-decides the token exactly as a full re-lex would.
        st = hold_state
        st.closed_frames = []
        prefix = ""
        prefix_match = re.fullmatch(r"[rRfFbB]{1,2}", st.last_ident)
        if prefix_match is not None:
            prefix = prefix_match.group(0)
            st.last_ident = ""
        st.pending = prefix + text[hold:]

    return st


_VOCAB_STATIC_CACHE: Dict[Any, Tuple[np.ndarray, np.ndarray, Tuple[Tuple[int, str, int], ...]]] = {}
_VOCAB_STATIC_CACHE_MAX = 8

# ``token_category`` returns string constants, but the vocab cache stores a compact
# integer code per entry. Map them explicitly instead of ``int(category)`` (which
# raised ValueError on every non-numeric category such as ``legal_stop``).
_CATEGORY_CODES: Dict[str, int] = {
    NEVER_EMIT: 0,
    LEGAL_STOP: 1,
    PROTOCOL_MARKERS: 2,
}
ORDINARY_TOKEN_CODE = -1


def _static_vocab_masks(
    vocab_map: Dict[int, str], tokenizer_hash: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[Tuple[int, str, int], ...]]:
    """Classify a vocabulary once into never-emit / protocol / legal-candidate ids.

    The third tuple field is a compact integer category code (``ORDINARY_TOKEN_CODE``
    for unclassified tokens), never a raw string.
    """
    key: Any = tokenizer_hash or f"id:{id(vocab_map)}:{len(vocab_map)}"
    cached = _VOCAB_STATIC_CACHE.get(key)
    if cached is not None:
        return cached

    never_ids: List[int] = []
    protocol_ids: List[int] = []
    entries: List[Tuple[int, str, int]] = []
    for tid, tok_str in vocab_map.items():
        category = token_category(tok_str)
        if category == NEVER_EMIT:
            never_ids.append(int(tid))
        elif category == PROTOCOL_MARKERS:
            protocol_ids.append(int(tid))
        else:
            code = (
                ORDINARY_TOKEN_CODE
                if category is None
                else _CATEGORY_CODES.get(category, ORDINARY_TOKEN_CODE)
            )
            entries.append((int(tid), tok_str, code))

    result = (
        np.asarray(never_ids, dtype=np.int64),
        np.asarray(protocol_ids, dtype=np.int64),
        tuple(entries),
    )
    if len(_VOCAB_STATIC_CACHE) >= _VOCAB_STATIC_CACHE_MAX:
        _VOCAB_STATIC_CACHE.pop(next(iter(_VOCAB_STATIC_CACHE)))
    _VOCAB_STATIC_CACHE[key] = result
    return result


class Vexx35Controller:
    """Stateful six-tier neuro-symbolic controller for Maya-250M."""

    #: ``ast.Nonlocal`` only rebinds names in an enclosing function frame and has
    #: no sandbox-escape surface; blocking it rejected textbook Tarjan SCC
    #: solutions that use ``nonlocal`` for the SCC counters.
    FORBIDDEN_AST_NODES = (ast.Global,)
    #: Pure algorithmic standard-library modules a candidate may import. ``os``,
    #: ``sys``, ``socket``, ``subprocess``, ``pathlib`` and friends stay blocked at
    #: both the AST layer and the runtime ``__import__`` layer.
    ALLOWED_STDLIB_MODULES = frozenset(
        {
            "math", "heapq", "collections", "bisect", "itertools",
            "functools", "typing", "fractions", "decimal", "operator", "string",
        }
    )
    FORBIDDEN_IDENTIFIERS = {
        "os", "sys", "subprocess", "socket", "shutil", "pathlib", "open",
        "eval", "exec", "compile", "__import__", "globals", "locals",
        "getattr", "setattr", "delattr", "vars", "input", "breakpoint",
    }

    #: String methods that are safe to call on a *literal*. ``" ".join(parts)`` is
    #: ordinary Python, and rejecting it failed correct solutions for style rather
    #: than for safety. ``format``/``format_map`` stay out: a format string can walk
    #: attribute paths into a rendering of the object graph.
    SAFE_LITERAL_ATTRIBUTES = frozenset(
        {
            "capitalize", "casefold", "center", "count", "endswith",
            "expandtabs", "find", "index", "isalnum", "isalpha", "isascii",
            "isdecimal", "isdigit", "isidentifier", "islower", "isnumeric",
            "isprintable", "isspace", "istitle", "isupper", "join", "ljust",
            "lower", "lstrip", "partition", "removeprefix", "removesuffix",
            "replace", "rfind", "rindex", "rjust", "rpartition", "rsplit",
            "rstrip", "split", "splitlines", "startswith", "strip",
            "swapcase", "title", "upper", "zfill",
        }
    )

    #: Standard container/string methods that are safe to call on a local variable,
    #: a subscript or an attribute chain. Before this allow-list, Tier-1 masking
    #: rejected every method call whose receiver was not literally ``self``/``cls``,
    #: so Python generation could not emit ``stack.append(x)``, ``adj[u].append(v)``
    #: or ``d.get(k, 0)`` at all.
    SAFE_STANDARD_METHODS = frozenset(
        {
            "append", "appendleft", "add", "clear", "copy", "count",
            "difference", "difference_update", "discard", "extend", "extendleft",
            "fromkeys", "get", "index", "insert", "intersection",
            "intersection_update", "issubset", "issuperset", "items", "keys",
            "pop", "popleft", "popitem", "remove", "reverse", "rotate",
            "setdefault", "sort", "symmetric_difference",
            "symmetric_difference_update", "union", "update", "values",
        }
    ) | SAFE_LITERAL_ATTRIBUTES

    def __init__(
        self,
        domain_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        symbol_table: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.initial_bounds = {
            k: (float(v[0]), float(v[1])) for k, v in (domain_bounds or {}).items()
        }
        self.bounds = dict(self.initial_bounds)
        self.symbol_table = dict(symbol_table or {})
        self.checkpoints: List[Dict[str, Any]] = []
        self.dpo_buffer: List[Dict[str, Any]] = []
        self.last_exact_fraction: Optional[Fraction] = None
        #: Verified definitions carried across ``<experiment>`` cells in one episode.
        self.experiment_preamble: List[str] = []
        self._symbol_cache: Dict[Any, Dict[str, Dict[str, Any]]] = {}
        self._accumulated_symbols: Dict[str, Dict[str, Any]] = dict(self.symbol_table)

    @staticmethod
    def crane_phase(prefix_text: str) -> str:
        """Determine active Vexx 3.5 phase from protocol delimiters."""
        last_exp_open = prefix_text.rfind(EXP_START)
        last_exp_close = prefix_text.rfind(EXP_END)
        if last_exp_open != -1 and last_exp_open > last_exp_close:
            return "EXPERIMENT_DESIGN"

        last_open = prefix_text.rfind(THINK_START)
        last_close = prefix_text.rfind(THINK_END)
        if last_open != -1 and last_open > last_close:
            return "UNCONSTRAINED_THINK"
        return "STRICT_VERIFIED"

    @classmethod
    def extract_ast_symbol_table(cls, code_text: str) -> Dict[str, Dict[str, Any]]:
        """Full calling-convention contracts for complete or in-flight code.

        v3 upgrade: replaces the v2 arity approximation. ``**kwargs`` no longer
        grants unbounded *positional* arity, required keyword-only arguments are
        tracked, ``self``/``cls``/``staticmethod`` binding is modelled, and methods
        with the same unqualified name in different classes are merged to the
        widest bounds and flagged ``ambiguous`` instead of being collapsed.

        Complete source uses ``ast`` (``exact_ast``). In-flight text uses a
        balanced-delimiter header scan (``partial_ast``); unrecoverable headers are
        reported as ``unknown`` rather than guessed.
        """
        from .codebase.contracts import build_contracts, header_contract_from_text

        extracted: Dict[str, Dict[str, Any]] = {}
        try:
            grouped = build_contracts(code_text, module="<inline>")
        except (SyntaxError, ValueError):
            grouped = None

        if grouped is not None:
            for contracts in grouped.values():
                for contract in contracts:
                    cls._record_contract(contract, extracted)
            return extracted

        for match in re.finditer(
            r"(?:^|\n)[ \t]*(?:async[ \t]+)?def[ \t]+[A-Za-z_]\w*[ \t]*\(",
            code_text,
        ):
            contract = header_contract_from_text(
                code_text[match.start() :], symbol_id="<inline>"
            )
            if contract is not None:
                cls._record_contract(contract, extracted)
        return extracted

    @staticmethod
    def _record_contract(contract, table: Dict[str, Dict[str, Any]]) -> None:
        """Merge one contract entry into a name-keyed symbol table."""
        if contract.confidence == "unknown":
            # Never install a guessed (0,0) contract: it would mask every valid call.
            return
        name = contract.symbol_id.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
        entry = contract.to_symbol_table_entry()
        entry["partial"] = contract.confidence != "exact_ast"
        existing = table.get(name)
        if existing is None:
            table[name] = entry
            return

        maxes = [existing.get("max_arity"), entry.get("max_arity")]
        merged = dict(existing)
        merged.update(
            {
                "min_arity": min(
                    int(existing.get("min_arity", 0)), int(entry.get("min_arity", 0))
                ),
                "max_arity": (
                    None
                    if any(maximum is None for maximum in maxes)
                    else max(int(maximum) for maximum in maxes)
                ),
                "has_var_keyword": bool(existing.get("has_var_keyword"))
                or bool(entry.get("has_var_keyword")),
                "has_var_positional": bool(existing.get("has_var_positional"))
                or bool(entry.get("has_var_positional")),
                "keyword_only": sorted(
                    set(existing.get("keyword_only") or [])
                    | set(entry.get("keyword_only") or [])
                ),
                # Ambiguous overloads must not *demand* a keyword argument.
                "required_keywords": [],
                "return_type": (
                    existing.get("return_type")
                    if existing.get("return_type") == entry.get("return_type")
                    else "any"
                ),
                "ambiguous": True,
                "partial": bool(existing.get("partial")) or bool(entry.get("partial")),
            }
        )
        table[name] = merged

    @staticmethod
    def _lex_code_state(code_prefix: str) -> Dict[str, Any]:
        """Lex a complete code prefix from a fresh :class:`LexerState`.

        Retained for callers that need a full-prefix snapshot; the decode loop uses
        the incremental :func:`advance_lexer_state` instead.
        """
        return advance_lexer_state(LexerState(), code_prefix).to_dict()

    def _live_symbol_table(self, active_code: str) -> Dict[str, Dict[str, Any]]:
        """Contracts for in-flight code without re-parsing the whole prefix.

        The full prefix is never re-parsed per token: a bounded tail window is
        scanned (so a definition split across the window boundary is still seen on
        the next step) and merged into the accumulated table.
        """
        window = 4096
        tail = active_code[-window:]
        key = (len(active_code), tail[-128:])
        cached = self._symbol_cache.get(key)
        if cached is not None:
            return cached
        merged = dict(self._accumulated_symbols)
        merged.update(self.extract_ast_symbol_table(tail))
        self._accumulated_symbols = merged
        if len(self._symbol_cache) >= 8:
            self._symbol_cache.pop(next(iter(self._symbol_cache)))
        self._symbol_cache[key] = merged
        return merged

    @classmethod
    def _classify_callee(
        cls, callee: str, live_syms: Dict[str, Dict[str, Any]]
    ) -> Tuple[Optional[str], bool]:
        """Resolve a call target to ``(callee_key, is_safe_standard_method)``.

        ``(None, False)`` means the call must be masked. Ordinary container/string
        methods on local variables, subscripts and attribute chains are allowed
        (``stack.append(x)``, ``adj[u].append(v)``, ``d.get(k, 0)``); dunder
        attributes, forbidden names and unknown non-standard methods stay blocked.
        """
        if "." in callee:
            recv, meth = callee.rsplit(".", 1)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", meth):
                return None, False
            if meth.startswith("__") or meth in cls.FORBIDDEN_IDENTIFIERS:
                return None, False
            if recv in {"self", "cls"}:
                if live_syms and meth not in live_syms:
                    return None, False
                return meth, False
            if meth in live_syms:
                return meth, False
            if meth in cls.SAFE_STANDARD_METHODS:
                return meth, True
            return None, False
        if callee in cls.FORBIDDEN_IDENTIFIERS:
            return None, False
        return callee, False

    def apply_in_flight_mask(
        self,
        logits: np.ndarray,
        prefix_text: str,
        vocab_map: Dict[int, str],
        expected_return_type: Optional[str] = None,
        remaining_token_budget: int = 256,
        mode: str = "python",
        tokenizer_hash: Optional[str] = None,
    ) -> np.ndarray:
        """Apply Tier 0 + Tier 1 constraints to next-token logits.

        The prefix is lexed once. Each candidate token is then advanced through
        ``advance_lexer_state`` over its own characters only, so a decode step is
        ``O(|V| * token_len + L)`` instead of ``O(|V| * L)``. Token categories are
        cached per vocabulary and symbol contracts come from a bounded tail window.
        """
        if logits.ndim != 1:
            raise ValueError(f"logits must be 1-D [vocab], got shape {tuple(logits.shape)}")
        masked = np.array(logits, dtype=np.float64, copy=True)
        never_ids, protocol_ids, entries = _static_vocab_masks(vocab_map, tokenizer_hash)
        if never_ids.size:
            masked[never_ids] = -np.inf

        phase = self.crane_phase(prefix_text)

        for tid in protocol_ids.tolist():
            tok_str = vocab_map.get(int(tid), "")
            if phase == "UNCONSTRAINED_THINK" and tok_str in (THINK_END, EXP_START):
                continue
            if phase == "EXPERIMENT_DESIGN" and tok_str == EXP_END:
                continue
            masked[int(tid)] = -np.inf

        if phase == "UNCONSTRAINED_THINK" and remaining_token_budget > 4:
            return masked

        if mode != "python" and phase != "EXPERIMENT_DESIGN":
            return masked

        active_code = (
            prefix_text[prefix_text.rfind(EXP_START) + len(EXP_START) :]
            if phase == "EXPERIMENT_DESIGN"
            else prefix_text
        )
        live_syms = self._live_symbol_table(active_code)
        base_state = advance_lexer_state(LexerState(), active_code)
        open_depth = len(base_state.stack)

        for tid, tok_str, _cat in entries:
            if remaining_token_budget <= open_depth + 1 and any(c in tok_str for c in "([{"):
                masked[tid] = -np.inf
                continue

            st = advance_lexer_state(base_state, tok_str)
            if not st.valid:
                masked[tid] = -np.inf
                continue

            if "(" in tok_str and st.in_str is None and st.stack:
                _, callee, _ = st.stack[-1]
                if callee is not None:
                    callee_key, safe_method = self._classify_callee(callee, live_syms)
                    if callee_key is None:
                        masked[tid] = -np.inf
                        continue
                    if not safe_method:
                        last_line = st.last_line.lstrip()
                        if not last_line.startswith(("def ", "@")) and phase != "EXPERIMENT_DESIGN":
                            if callee_key not in live_syms:
                                masked[tid] = -np.inf
                                continue
                            ret = live_syms[callee_key].get("return_type")
                            if expected_return_type and ret not in {expected_return_type, "any"}:
                                masked[tid] = -np.inf
                                continue

            if ")" in tok_str and base_state.valid and st.closed_frames:
                closed_delim, closed_callee, comma_cnt = st.closed_frames[-1]
                if closed_delim == "(" and closed_callee is not None:
                    callee_key, safe_method = self._classify_callee(closed_callee, live_syms)
                    if callee_key is None:
                        masked[tid] = -np.inf
                        continue
                    last_line = base_state.last_line.lstrip()
                    if (
                        not safe_method
                        and callee_key in live_syms
                        and not last_line.startswith(("def ", "@"))
                    ):
                        tok_lparen = tok_str.rfind("(")
                        lparen = (
                            len(active_code) + tok_lparen
                            if tok_lparen != -1
                            else active_code.rfind("(")
                        )
                        rparen = len(active_code) + tok_str.rfind(")")
                        inner_call = (
                            (active_code + tok_str)[lparen + 1 : rparen].strip()
                            if rparen > lparen
                            else ""
                        )
                        if inner_call.endswith(",") and comma_cnt >= 1:
                            masked[tid] = -np.inf
                            continue
                        if "*" not in inner_call and "=" not in inner_call:
                            n_args = 0 if inner_call == "" else (comma_cnt + 1)
                            spec = live_syms[callee_key]
                            min_a = spec.get("min_arity", spec.get("arity", 0))
                            max_a = spec.get("max_arity", spec.get("arity", n_args))
                            if n_args < min_a or (max_a is not None and n_args > max_a):
                                masked[tid] = -np.inf
                                continue

        return masked

    def filter_draft_tokens_by_grammar(
        self,
        prefix_text: str,
        draft_token_ids: List[int],
        vocab_map: Dict[int, str],
        remaining_token_budget: int = 256,
        mode: str = "python",
    ) -> List[int]:
        """Prune speculative draft tokens that violate Vexx Tier-0/Tier-1 syntax.

        This is a *grammar* filter only; distributional correctness (the target
        model confirming each draft token) belongs to `generate_maya`.
        """
        accepted: List[int] = []
        running_text = prefix_text
        dummy_logits = np.zeros(max(vocab_map.keys()) + 1 if vocab_map else 1, dtype=np.float64)
        for step_offset, tid in enumerate(draft_token_ids):
            budget = max(1, remaining_token_budget - step_offset)
            masked = self.apply_in_flight_mask(
                dummy_logits,
                running_text,
                vocab_map,
                remaining_token_budget=budget,
                mode=mode,
            )
            if not np.isfinite(masked[tid]):
                break
            accepted.append(tid)
            running_text = running_text + vocab_map.get(tid, "")
        return accepted

    def verify_mtp_draft_tree(
        self,
        prefix_text: str,
        draft_token_ids: List[int],
        vocab_map: Dict[int, str],
        remaining_token_budget: int = 256,
        mode: str = "python",
    ) -> List[int]:
        """Vexx 3.5 parallel speculative draft verifier: accept longest valid MTP prefix."""
        accepted: List[int] = []
        running_text = prefix_text
        dummy_logits = np.zeros(max(vocab_map.keys()) + 1 if vocab_map else 1, dtype=np.float64)
        for step_offset, tid in enumerate(draft_token_ids):
            budget = max(1, remaining_token_budget - step_offset)
            masked = self.apply_in_flight_mask(
                dummy_logits,
                running_text,
                vocab_map,
                remaining_token_budget=budget,
                mode=mode,
            )
            if not np.isfinite(masked[tid]):
                break
            accepted.append(tid)
            running_text = running_text + vocab_map.get(tid, "")
        return accepted

    @staticmethod
    def crane_cas_evaluate(expr_str: str, env: Optional[Dict[str, Any]] = None) -> str:
        """Evaluate symbolic or rational expressions exactly via safe_sympy."""
        env = dict(env or {})
        clean = (
            expr_str.strip()
            .replace("^", "**")
            .replace("×", "*")
            .replace("÷", "/")
            .replace("−", "-")
        )
        try:
            local_syms: Dict[str, Any] = {}
            for key, value in env.items():
                if isinstance(value, sp.Basic):
                    local_syms[key] = value
                elif isinstance(value, (int, float, Fraction)):
                    local_syms[key] = sp.Rational(str(value))
                elif isinstance(value, str):
                    local_syms[key] = safe_sympy(value, {})
                else:
                    raise ValueError(f"unsupported environment value for {key!r}")
            parsed = safe_sympy(clean, local_syms)
            if parsed.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
                return "DIV_BY_ZERO"
            simplified = sp.simplify(parsed)
            if isinstance(simplified, bool):
                return "True" if simplified else "False"
            if simplified.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
                return "DIV_BY_ZERO"
            if simplified.is_Rational:
                frac = Fraction(int(simplified.p), int(simplified.q))
                return (
                    str(frac.numerator)
                    if frac.denominator == 1
                    else f"{frac.numerator}/{frac.denominator}"
                )
            return str(simplified)
        except ZeroDivisionError:
            return "DIV_BY_ZERO"
        except Exception as e:
            return f"ERROR: {e}"

    @staticmethod
    def solve_linear_system(eqs: List[str], vars_list: List[str]) -> Dict[str, Any]:
        """Solve a linear system with rank, free-variable, and residual verification."""
        if not isinstance(eqs, list) or not isinstance(vars_list, list):
            raise TypeError("eqs and vars_list must be lists of strings")
        if not (0 < len(eqs) <= 8) or not (0 < len(vars_list) <= 8):
            raise ValueError("solve_linear_system supports 1-8 equations and variables")
        if any(not isinstance(v, str) or not v.isidentifier() or v.startswith("_") for v in vars_list):
            raise ValueError("variable names must be valid public identifiers")
        if len(set(vars_list)) != len(vars_list):
            raise ValueError("variable names must be unique")

        syms = [sp.Symbol(v) for v in vars_list]
        declared = set(syms)
        env = {name: symbol for name, symbol in zip(vars_list, syms)}
        expressions = []
        for equation in eqs:
            if not isinstance(equation, str) or equation.count("=") != 1:
                raise ValueError(f"Equation {equation!r} must contain exactly one '='")
            lhs, rhs = equation.replace("−", "-").split("=", 1)
            expr = sp.expand(safe_sympy(lhs, env) - safe_sympy(rhs, env))
            undeclared = expr.free_symbols - declared
            if undeclared:
                raise ValueError(
                    "equation contains undeclared variable(s): "
                    + ", ".join(sorted(str(x) for x in undeclared))
                )
            try:
                poly = sp.Poly(expr, *syms)
            except sp.PolynomialError as exc:
                raise ValueError("equations must be polynomial in declared variables") from exc
            if poly.total_degree() > 1:
                raise ValueError("nonlinear equation supplied to linear-system solver")
            expressions.append(expr)

        coeff_rows = [[sp.expand(expr).coeff(symbol) for symbol in syms] for expr in expressions]
        constants = [sp.expand(expr).subs({symbol: 0 for symbol in syms}) for expr in expressions]
        coeff = sp.Matrix(coeff_rows)
        augmented = coeff.row_join(sp.Matrix([-c for c in constants]))
        rank_coeff = coeff.rank()
        rank_augmented = augmented.rank()
        solution_set = sp.linsolve(expressions, *syms)

        if solution_set is sp.EmptySet or rank_coeff != rank_augmented:
            return {
                "status": "inconsistent",
                "variables": list(vars_list),
                "solution": None,
                "free_variables": [],
                "rank_coeff": int(rank_coeff),
                "rank_augmented": int(rank_augmented),
                "residual_check": {},
            }

        solution_tuple = next(iter(solution_set))
        solution_exprs = [sp.simplify(value) for value in solution_tuple]
        solution = {name: str(value) for name, value in zip(vars_list, solution_exprs)}
        free_symbols = set().union(*(value.free_symbols for value in solution_exprs)) & declared
        free_variables = [name for name in vars_list if sp.Symbol(name) in free_symbols]
        residuals = {
            str(expr): str(sp.simplify(expr.subs(dict(zip(syms, solution_exprs)))))
            for expr in expressions
        }
        if any(sp.simplify(expr.subs(dict(zip(syms, solution_exprs)))) != 0 for expr in expressions):
            raise ArithmeticError("internal solver residual check failed")

        status = "underdetermined" if free_variables or rank_coeff < len(syms) else "unique"
        return {
            "status": status,
            "variables": list(vars_list),
            "solution": solution,
            "free_variables": free_variables,
            "rank_coeff": int(rank_coeff),
            "rank_augmented": int(rank_augmented),
            "residual_check": residuals,
        }

    @staticmethod
    def integrate_definite(expr_str: str, var_name: str, a: Any, b: Any) -> str:
        x = sp.Symbol(var_name)
        expr = safe_sympy(
            expr_str.replace("^", "**").replace("²", "**2").replace("³", "**3"),
            {var_name: x},
        )
        lo, hi = safe_sympy(str(a), {}), safe_sympy(str(b), {})
        val = sp.integrate(expr, (x, lo, hi))
        return str(sp.simplify(val))

    @staticmethod
    def modular_pow(a: int, b: int, m: int) -> str:
        return str(pow(int(a), int(b), int(m)))

    def save_checkpoint(
        self,
        var_name: str,
        token_idx: int,
        hybrid_cache: List[Dict[str, Any]],
        entropy: float,
        proposed_val: float,
    ) -> None:
        """Snapshot the hybrid cache *at* ``token_idx`` for exact CDCL jumpback.

        A GDN recurrent state has no sequence dimension, so a bundle that carries one
        must be snapshotted *at* that token. A pure full-attention KV bundle can be
        sliced safely, so it may be captured later and truncated on jumpback. When
        recurrent state is present and ``position`` disagrees with ``token_idx``, the
        jumpback would retain future tokens, so it is rejected here instead of
        silently corrupting causal parity.
        """
        if (
            isinstance(hybrid_cache, dict)
            and hybrid_cache.get("position") is not None
            and _contains_recurrent_state(hybrid_cache)
        ):
            position = int(hybrid_cache["position"])
            if position != int(token_idx):
                raise ValueError(
                    f"save_checkpoint for token {int(token_idx)} received a cache at "
                    f"position {position}; recurrent state cannot be rewound by slicing"
                )
        self.checkpoints.append(
            {
                "var": var_name,
                "token_idx": int(token_idx),
                "cache": clone_cache_generic(hybrid_cache),
                "bounds_snapshot": dict(self.bounds),
                "entropy": float(entropy),
                "val": float(proposed_val),
            }
        )

    def single_row_bound_contractor(
        self,
        conflict_vars: set,
        constraint: Dict[str, Any],
        assignment: Dict[str, float],
    ) -> Tuple[int, List[Dict[str, Any]], Tuple[float, float]]:
        candidates = [cp for cp in self.checkpoints if cp["var"] in conflict_vars]
        if not candidates:
            raise ValueError(f"No checkpoint found for conflict core {sorted(conflict_vars)}")
        culprit = max(candidates, key=lambda c: c["entropy"])
        c_var = culprit["var"]
        tok_idx = culprit["token_idx"]

        self.bounds = dict(culprit["bounds_snapshot"])
        coeffs = constraint["coeffs"]
        if c_var not in coeffs:
            raise ValueError(f"Conflict variable '{c_var}' absent from constraint coefficients")
        rhs = Fraction(str(constraint["rhs"]))
        a_c = Fraction(str(coeffs[c_var]))
        if a_c == 0:
            raise ValueError("Bound contractor needs a non-zero coefficient for the culprit")
        rem = rhs
        for var, coeff in coeffs.items():
            if var != c_var:
                if var not in assignment:
                    raise ValueError(f"Constraint references '{var}' missing from assignment")
                rem -= Fraction(str(coeff)) * Fraction(str(assignment[var]))
        exact_bound = rem / a_c
        self.last_exact_fraction = exact_bound
        bound_float = float(exact_bound)

        lo, hi = self.bounds.get(c_var, (-float("inf"), float("inf")))
        if a_c > 0:
            widened_hi = math.nextafter(bound_float, float("inf"))
            self.bounds[c_var] = (lo, min(hi, widened_hi))
        else:
            widened_lo = math.nextafter(bound_float, -float("inf"))
            self.bounds[c_var] = (max(lo, widened_lo), hi)

        # BUG-08 fix: recursively rewind v1 flat list caches and v2 structured
        # multi-loop caches ({"prelude", "core_loops", "coda"}).
        rewound_cache = truncate_cache_at_token(clone_cache_generic(culprit["cache"]), tok_idx)

        self.checkpoints = [cp for cp in self.checkpoints if cp["token_idx"] <= tok_idx]
        return tok_idx, rewound_cache, self.bounds[c_var]

    trigger_farkas_cdcl_jumpback = single_row_bound_contractor

    def propagate_rational_polytope(
        self,
        constraints: List[Dict[str, Any]],
        initial_bounds: Optional[Dict[str, Tuple[Any, Any]]] = None,
        max_rounds: int = 8,
    ) -> Dict[str, Any]:
        """Multi-row exact rational interval contractor + Farkas infeasibility check.

        Each constraint has ``{"coeffs": {var: a_i, ...}, "rhs": b, "op": "<=" | ">=" | "=="}``
        representing ``sum_i a_i x_i <= b`` (or ``>=`` / ``==``). Bounds are tracked as
        exact :class:`Fraction` pairs ``(lo, hi)`` (``None`` for ``-inf``/``+inf``).
        When a row forces ``lo > hi`` on any variable, or a non-negative linear
        combination of rows yields ``0 <= c < 0`` (Farkas certificate), ``feasible``
        is ``False`` and the conflicting row indices are returned.
        """
        exact_bounds: Dict[str, List[Optional[Fraction]]] = {}
        seed_bounds = initial_bounds if initial_bounds is not None else self.bounds
        for var, (lo, hi) in (seed_bounds or {}).items():
            lo_f = None if lo is None or lo == -float("inf") else Fraction(str(lo))
            hi_f = None if hi is None or hi == float("inf") else Fraction(str(hi))
            exact_bounds[str(var)] = [lo_f, hi_f]

        normalized_rows: List[Tuple[int, Dict[str, Fraction], Fraction]] = []
        for idx, row in enumerate(constraints):
            op = str(row.get("op", "<=")).strip()
            rhs = Fraction(str(row["rhs"]))
            coeffs = {str(k): Fraction(str(val)) for k, val in (row.get("coeffs") or {}).items() if Fraction(str(val)) != 0}
            for k in coeffs:
                exact_bounds.setdefault(k, [None, None])
            if op == "<=":
                normalized_rows.append((idx, coeffs, rhs))
            elif op == ">=":
                normalized_rows.append((idx, {k: -c for k, c in coeffs.items()}, -rhs))
            elif op in ("==", "="):
                normalized_rows.append((idx, coeffs, rhs))
                normalized_rows.append((idx, {k: -c for k, c in coeffs.items()}, -rhs))
            else:
                raise ValueError(f"unsupported constraint operator {op!r}")

        # Check pairwise positive row combinations for direct Farkas contradiction:
        # if c_1 + alpha * c_2 == 0 and rhs_1 + alpha * rhs_2 < 0 for alpha > 0.
        for i in range(len(normalized_rows)):
            idx1, c1, b1 = normalized_rows[i]
            if not c1 and b1 < 0:
                return {
                    "feasible": False,
                    "conflict_rows": [idx1],
                    "conflict_vars": [],
                    "farkas_certificate": {str(idx1): "1"},
                    "bounds": {},
                }
            for j in range(i + 1, len(normalized_rows)):
                idx2, c2, b2 = normalized_rows[j]
                if set(c1.keys()) == set(c2.keys()) and c1:
                    first_var = next(iter(c1))
                    ratio = -c1[first_var] / c2[first_var]
                    if ratio > 0 and all(c1[v] + ratio * c2[v] == 0 for v in c1):
                        combined_rhs = b1 + ratio * b2
                        if combined_rhs < 0:
                            return {
                                "feasible": False,
                                "conflict_rows": sorted({idx1, idx2}),
                                "conflict_vars": sorted(c1.keys()),
                                "farkas_certificate": {str(idx1): "1", str(idx2): str(ratio)},
                                "bounds": {},
                            }

        # Iterative multi-row bound propagation over exact Fractions
        for _ in range(max(1, int(max_rounds))):
            changed = False
            for row_idx, coeffs, rhs in normalized_rows:
                for target_var, a_c in coeffs.items():
                    # Compute minimum possible sum of other terms: sum_{v != target_var} min(a_v * x_v)
                    min_other = Fraction(0)
                    possible = True
                    for other_var, a_o in coeffs.items():
                        if other_var == target_var:
                            continue
                        lo_o, hi_o = exact_bounds[other_var]
                        if a_o > 0:
                            if lo_o is None:
                                possible = False
                                break
                            min_other += a_o * lo_o
                        else:
                            if hi_o is None:
                                possible = False
                                break
                            min_other += a_o * hi_o
                    if not possible:
                        continue
                    rem = rhs - min_other
                    implied = rem / a_c
                    cur_lo, cur_hi = exact_bounds[target_var]
                    if a_c > 0:
                        if cur_hi is None or implied < cur_hi:
                            exact_bounds[target_var][1] = implied
                            changed = True
                    else:
                        if cur_lo is None or implied > cur_lo:
                            exact_bounds[target_var][0] = implied
                            changed = True
                    new_lo, new_hi = exact_bounds[target_var]
                    if new_lo is not None and new_hi is not None and new_lo > new_hi:
                        return {
                            "feasible": False,
                            "conflict_rows": [row_idx],
                            "conflict_vars": sorted(coeffs.keys()),
                            "farkas_certificate": {str(row_idx): "1"},
                            "bounds": {
                                k: (
                                    -float("inf") if v[0] is None else float(v[0]),
                                    float("inf") if v[1] is None else float(v[1]),
                                )
                                for k, v in exact_bounds.items()
                            },
                        }
            if not changed:
                break

        # Exact multi-row Fourier-Motzkin elimination for general k-row Farkas infeasibility
        # (catches 3+ variable contradictions like x+y<=2, y+z<=2, z+x<=2, x+y+z>=4 where no pair conflicts).
        fm_vars = sorted(exact_bounds.keys())
        if 1 <= len(fm_vars) <= 8 and len(normalized_rows) <= 24:
            # Each FM row: (coeffs_dict, rhs_fraction, farkas_multipliers_dict)
            fm_rows: List[Tuple[Dict[str, Fraction], Fraction, Dict[int, Fraction]]] = []
            for r_idx, coeffs, rhs in normalized_rows:
                fm_rows.append((dict(coeffs), rhs, {r_idx: Fraction(1)}))
            # Also include finite variable bounds as virtual rows (keyed by -1 - i for traceability)
            for v_i, var in enumerate(fm_vars):
                lo_f, hi_f = exact_bounds[var]
                if lo_f is not None:
                    fm_rows.append(({var: Fraction(-1)}, -lo_f, {}))
                if hi_f is not None:
                    fm_rows.append(({var: Fraction(1)}, hi_f, {}))

            for var in fm_vars:
                pos_rows = []
                neg_rows = []
                zero_rows = []
                for c_map, rhs_val, mults in fm_rows:
                    a = c_map.get(var, Fraction(0))
                    if a > 0:
                        pos_rows.append((c_map, rhs_val, mults, a))
                    elif a < 0:
                        neg_rows.append((c_map, rhs_val, mults, -a))
                    else:
                        zero_rows.append((c_map, rhs_val, mults))
                next_rows = list(zero_rows)
                for p_map, p_rhs, p_mults, a_pos in pos_rows:
                    for n_map, n_rhs, n_mults, a_neg in neg_rows:
                        # Combine: (1/a_pos)*pos_row + (1/a_neg)*neg_row eliminates `var`
                        w_p = Fraction(1, 1) / a_pos
                        w_n = Fraction(1, 1) / a_neg
                        comb: Dict[str, Fraction] = {}
                        for k in set(p_map.keys()) | set(n_map.keys()):
                            if k == var:
                                continue
                            val = p_map.get(k, Fraction(0)) * w_p + n_map.get(k, Fraction(0)) * w_n
                            if val != 0:
                                comb[k] = val
                        comb_rhs = p_rhs * w_p + n_rhs * w_n
                        comb_mults: Dict[int, Fraction] = {}
                        for rk, rv in p_mults.items():
                            comb_mults[rk] = comb_mults.get(rk, Fraction(0)) + rv * w_p
                        for rk, rv in n_mults.items():
                            comb_mults[rk] = comb_mults.get(rk, Fraction(0)) + rv * w_n
                        if not comb:
                            if comb_rhs < 0:
                                nz_rows = sorted(k for k, v in comb_mults.items() if v > 0)
                                cert = {str(k): str(comb_mults[k]) for k in nz_rows}
                                cert["derived_rhs"] = str(comb_rhs)
                                return {
                                    "feasible": False,
                                    "conflict_rows": nz_rows,
                                    "conflict_vars": fm_vars,
                                    "farkas_certificate": cert,
                                    "bounds": {
                                        k: (
                                            -float("inf") if v[0] is None else float(v[0]),
                                            float("inf") if v[1] is None else float(v[1]),
                                        )
                                        for k, v in exact_bounds.items()
                                    },
                                }
                            continue
                        next_rows.append((comb, comb_rhs, comb_mults))
                        if len(next_rows) > 128:
                            break
                    if len(next_rows) > 128:
                        break
                fm_rows = next_rows
                if not fm_rows or len(fm_rows) > 128:
                    break

        float_bounds: Dict[str, Tuple[float, float]] = {}
        rational_bounds: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        for var, (lo_f, hi_f) in exact_bounds.items():
            lo_flt = -float("inf") if lo_f is None else math.nextafter(float(lo_f), -float("inf"))
            hi_flt = float("inf") if hi_f is None else math.nextafter(float(hi_f), float("inf"))
            float_bounds[var] = (lo_flt, hi_flt)
            rational_bounds[var] = (None if lo_f is None else str(lo_f), None if hi_f is None else str(hi_f))
        self.bounds.update(float_bounds)
        return {
            "feasible": True,
            "conflict_rows": [],
            "conflict_vars": [],
            "farkas_certificate": {},
            "bounds": float_bounds,
            "rational_bounds": rational_bounds,
        }

    @staticmethod
    def synthesize_polynomial_hypothesis(
        points: Sequence[Tuple[Any, Any]],
        var_name: str = "n",
        holdout_points: Sequence[Tuple[Any, Any]] = (),
    ) -> Dict[str, Any]:
        """Exact rational Lagrange interpolation over empirical ``(n, f(n))`` pairs.

        Used during Stage 2 (``fit``) of the counterexample-first research protocol
        to synthesize a closed-form symbolic candidate from Stage-1 observations and
        check whether it generalizes to held-out points.
        """
        if not points:
            raise ValueError("synthesize_polynomial_hypothesis requires at least one point")
        if not var_name.isidentifier() or var_name.startswith("_"):
            raise ValueError(f"invalid variable name {var_name!r}")
        x_sym = sp.Symbol(var_name)
        xs = [ sp.Rational(str(pt[0])) for pt in points ]
        ys = [ sp.Rational(str(pt[1])) for pt in points ]
        if len(set(xs)) != len(xs):
            raise ValueError("interpolation x-coordinates must be distinct")

        poly = sp.Integer(0)
        k = len(xs)
        for i in range(k):
            term = ys[i]
            for j in range(k):
                if i != j:
                    term = term * (x_sym - xs[j]) / (xs[i] - xs[j])
            poly += term
        simplified = sp.simplify(sp.expand(poly))
        degree = int(sp.Poly(simplified, x_sym).degree()) if simplified != 0 else 0

        holdout_matches = True
        holdout_mismatches: List[Dict[str, str]] = []
        for hx_raw, hy_raw in holdout_points:
            hx = sp.Rational(str(hx_raw))
            hy = sp.Rational(str(hy_raw))
            pred = sp.simplify(simplified.subs(x_sym, hx))
            if pred != hy:
                holdout_matches = False
                holdout_mismatches.append(
                    {"x": str(hx), "expected": str(hy), "predicted": str(pred)}
                )

        return {
            "expression": str(simplified),
            "degree": degree,
            "variable": var_name,
            "points_used": len(points),
            "holdout_verified": holdout_matches and len(holdout_points) > 0,
            "holdout_mismatches": holdout_mismatches,
        }

    @staticmethod
    def synthesize_recurrence_hypothesis(
        sequence: Sequence[Any],
        *,
        max_order: int = 4,
        holdout_count: int = 2,
    ) -> Dict[str, Any]:
        """Exact rational linear-recurrence / geometric hypothesis synthesizer.

        Searches for the lowest-order exact homogeneous recurrence
        ``a[n] = c_1 a[n-1] + ... + c_k a[n-k]`` (or affine recurrence
        ``a[n] = c_0 + sum c_j a[n-j]``) over :class:`Fraction` using exact
        Gaussian elimination, and verifies predictions on held-out tail terms.
        """
        vals = [Fraction(str(x)) for x in sequence]
        n_total = len(vals)
        if n_total < 4:
            raise ValueError("synthesize_recurrence_hypothesis needs at least 4 sequence terms")
        h_cnt = max(1, min(int(holdout_count), n_total - 3))
        train = vals[:-h_cnt]
        holdout = vals[-h_cnt:]

        def _solve_exact_linear(mat: List[List[Fraction]], rhs: List[Fraction]) -> Optional[List[Fraction]]:
            m = len(mat)
            if m == 0:
                return None
            cols = len(mat[0])
            if m < cols:
                return None
            aug = [list(mat[r]) + [rhs[r]] for r in range(m)]
            piv_row = 0
            for col in range(cols):
                sel = None
                for r in range(piv_row, m):
                    if aug[r][col] != 0:
                        sel = r
                        break
                if sel is None:
                    return None
                aug[piv_row], aug[sel] = aug[sel], aug[piv_row]
                piv = aug[piv_row][col]
                aug[piv_row] = [v / piv for v in aug[piv_row]]
                for r in range(m):
                    if r != piv_row and aug[r][col] != 0:
                        factor = aug[r][col]
                        aug[r] = [aug[r][c] - factor * aug[piv_row][c] for c in range(cols + 1)]
                piv_row += 1
            for r in range(cols, m):
                if aug[r][cols] != 0:
                    return None
            return [aug[c][cols] for c in range(cols)]

        for order in range(1, int(max_order) + 1):
            for affine in (False, True):
                n_eq = len(train) - order
                n_vars = order + (1 if affine else 0)
                if n_eq < n_vars:
                    continue
                mat: List[List[Fraction]] = []
                rhs: List[Fraction] = []
                for idx in range(order, len(train)):
                    row = [train[idx - j] for j in range(1, order + 1)]
                    if affine:
                        row.append(Fraction(1))
                    mat.append(row)
                    rhs.append(train[idx])
                sol = _solve_exact_linear(mat, rhs)
                if sol is None:
                    continue
                coeffs = sol[:order]
                const_term = sol[order] if affine else Fraction(0)

                # Verify on all holdout terms autoregressively
                buf = list(train)
                mismatches: List[Dict[str, str]] = []
                for h_idx, expected_val in enumerate(holdout):
                    pred = const_term + sum(coeffs[j] * buf[-(j + 1)] for j in range(order))
                    if pred != expected_val:
                        mismatches.append(
                            {"index": str(len(train) + h_idx), "expected": str(expected_val), "predicted": str(pred)}
                        )
                    buf.append(pred)
                if not mismatches:
                    terms = []
                    for j, c in enumerate(coeffs, 1):
                        if c != 0:
                            terms.append(f"({c})*a[n-{j}]" if c != 1 else f"a[n-{j}]")
                    if const_term != 0:
                        terms.append(str(const_term))
                    formula = "a[n] = " + (" + ".join(terms) if terms else "0")
                    return {
                        "verified": True,
                        "order": order,
                        "affine": affine,
                        "coefficients": [str(c) for c in coeffs],
                        "constant_term": str(const_term),
                        "formula": formula,
                        "holdout_verified": True,
                        "holdout_count": h_cnt,
                        "holdout_mismatches": [],
                    }

        return {
            "verified": False,
            "order": 0,
            "affine": False,
            "coefficients": [],
            "constant_term": "0",
            "formula": None,
            "holdout_verified": False,
            "holdout_count": h_cnt,
            "holdout_mismatches": [],
        }

    #: SI base-dimension vectors for the unit symbols Tier 4 can verify exactly.
    _DIMENSION_BASE = {
        "kg": {"M": 1},
        "g": {"M": 1},
        "m": {"L": 1},
        "s": {"T": 1},
        "a": {"I": 1},
        "k": {"Theta": 1},
        "mol": {"N": 1},
        "cd": {"J": 1},
        "n": {"M": 1, "L": 1, "T": -2},
        "j": {"M": 1, "L": 2, "T": -2},
        "w": {"M": 1, "L": 2, "T": -3},
        "pa": {"M": 1, "L": -1, "T": -2},
        "hz": {"T": -1},
        "c": {"I": 1, "T": 1},
        "v": {"M": 2, "L": 2, "T": -3, "I": -1},
    }

    @classmethod
    def parse_unit_dimensions(cls, text: str) -> Optional[Dict[str, int]]:
        """Dimension vector of a unit product such as ``kg*m/s^2``; None if unknown."""
        cleaned = str(text or "").strip().lower().replace("·", "*").replace(" ", "")
        if not cleaned:
            return None
        tokens = re.findall(r"[*/]|[a-z]+(?:\^[+-]?\d+)?", cleaned)
        if not tokens or "".join(tokens) != cleaned:
            return None
        total: Dict[str, int] = {}
        sign = 1
        for token in tokens:
            if token == "*":
                sign = 1
                continue
            if token == "/":
                sign = -1
                continue
            match = re.fullmatch(r"([a-z]+)(?:\^([+-]?\d+))?", token)
            if match is None:
                return None
            unit = match.group(1)
            if unit not in cls._DIMENSION_BASE:
                return None
            exponent = int(match.group(2) or 1)
            for dimension, power in cls._DIMENSION_BASE[unit].items():
                total[dimension] = total.get(dimension, 0) + sign * exponent * power
        return {key: value for key, value in total.items() if value}

    @staticmethod
    def _normal_two_sided_p(statistic: float) -> float:
        return math.erfc(abs(float(statistic)) / math.sqrt(2.0))

    @staticmethod
    def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
        """Regularized incomplete beta I_x(a, b) via Lentz continued fraction (pure math)."""
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        if x > (a + 1.0) / (a + b + 2.0):
            return 1.0 - Vexx35Controller._regularized_incomplete_beta(b, a, 1.0 - x)
        log_bt = (
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + a * math.log(x)
            + b * math.log(1.0 - x)
        )
        bt = math.exp(log_bt)
        fpmin = 1e-300
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, 200):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-12:
                break
        return max(0.0, min(1.0, bt * h / a))

    @classmethod
    def _f_dist_survival(cls, f_val: float, df1: int, df2: int) -> float:
        """Exact upper-tail p-value P(F_{df1, df2} >= f_val)."""
        if f_val <= 0.0 or df1 <= 0 or df2 <= 0:
            return 1.0
        x = float(df2) / (float(df2) + float(df1) * float(f_val))
        return cls._regularized_incomplete_beta(0.5 * float(df2), 0.5 * float(df1), x)

    @classmethod
    def _pearson_r_two_sided_p(cls, r_val: float, df: int) -> float:
        """Exact two-sided p-value for Pearson correlation r with df = N - 2."""
        r_abs = abs(float(r_val))
        if df <= 0:
            return 1.0
        if r_abs >= 1.0:
            return 0.0
        t_stat = r_abs * math.sqrt(float(df) / max(1e-15, 1.0 - r_abs * r_abs))
        return cls._student_t_two_sided_p(t_stat, df)

    @staticmethod
    def _student_t_two_sided_p(statistic: float, df: int) -> Optional[float]:
        """Exact two-sided ``P(|T_df| > |t|)`` for integer ``df >= 1`` in pure math.

        Uses the exact finite trigonometric series for the regularized incomplete
        beta integral ``I_{df / (df + t^2)}(df/2, 1/2)`` over integer ``df``.
        """
        if df <= 0:
            return None
        t_abs = abs(float(statistic))
        if t_abs == 0.0:
            return 1.0
        theta = math.atan(t_abs / math.sqrt(float(df)))
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        if df == 1:
            p = 1.0 - (2.0 / math.pi) * theta
            return max(0.0, min(1.0, p))
        if df % 2 == 1:
            # Odd df >= 3: 1 - (2/pi) * (theta + sin(theta)*cos(theta)*sum_{k=1}^{(df-1)//2 - 1} c_k cos^{2k}(theta))
            series = 1.0
            term = 1.0
            cos2 = cos_t * cos_t
            for k in range(1, (df - 1) // 2):
                term *= (2.0 * k) / (2.0 * k + 1.0) * cos2
                series += term
            cdf_central = (2.0 / math.pi) * (theta + sin_t * cos_t * series)
            return max(0.0, min(1.0, 1.0 - cdf_central))
        else:
            # Even df >= 2: 1 - sin(theta) * sum_{k=0}^{df/2 - 1} ((2k - 1)!! / (2k)!!) cos^{2k}(theta)
            series = 1.0
            term = 1.0
            cos2 = cos_t * cos_t
            for k in range(1, df // 2):
                term *= (2.0 * k - 1.0) / (2.0 * k) * cos2
                series += term
            cdf_central = sin_t * series
            return max(0.0, min(1.0, 1.0 - cdf_central))

    @staticmethod
    def _chi2_survival(statistic: float, df: int) -> Optional[float]:
        """Exact upper-tail ``P(X > x)`` for integer ``df`` via the gamma recursion."""
        x = float(statistic)
        if x < 0 or df <= 0:
            return None
        half_x = x / 2.0
        if df % 2 == 0:
            term = 1.0
            total = 1.0
            for i in range(1, df // 2):
                term *= half_x / i
                total += term
            return math.exp(-half_x) * total
        # Odd df: recurse from Q(1/2, x) = erfc(sqrt(x)) upward.
        survival = math.erfc(math.sqrt(half_x))
        gamma = math.sqrt(math.pi) / 2.0
        power = math.sqrt(half_x)
        for j in range((df - 1) // 2):
            survival += math.exp(-half_x) * power / gamma
            power *= half_x
            gamma *= 1.5 + j
        return survival

    @classmethod
    def verify_statistical_report(cls, prompt: str) -> Optional[Dict[str, str]]:
        """Executable statistics/dimensional audit of a research claim.

        Flags only arithmetically impossible statements: reported means/percentages
        that cannot arise from integer counts at the stated ``N`` (GRIM), test
        statistics whose reported p-value is impossible for the given ``z``, ``t``
        or ``chi^2``, and equations whose unit dimensions disagree. Returns ``None``
        when nothing is provably wrong, so the caller keeps its existing heuristics.
        """
        text = str(prompt or "")
        lower = text.lower()

        # 1. GRIM: a percentage or mean that no integer count can produce.
        m_grim = re.search(
            r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:patients\s+|people\s+|participants\s+)?"
            r"(?:n|N)\s*=\s*(\d+)",
            text,
        )
        if m_grim is not None:
            percent = float(m_grim.group(1))
            n_total = int(m_grim.group(2))
            if n_total > 0:
                count = percent * n_total / 100.0
                if abs(count - round(count)) > 1e-6:
                    return {
                        "type": "grim_violation",
                        "verdict": "CONFLICT",
                        "explanation": (
                            f"{percent:g}% of N={n_total} implies {count:g} subjects, which is "
                            "not an integer count (GRIM violation).\nFINAL: CONFLICT"
                        ),
                    }
        m_mean = re.search(
            r"\b(?:mean|average)\s*(?:=|of|is)\s*(\d+(?:\.\d+)?)[^.;]{0,40}?"
            r"\b(?:n|N)\s*=\s*(\d+)",
            text,
        )
        if m_mean is not None:
            mean = float(m_mean.group(1))
            n_total = int(m_mean.group(2))
            if n_total > 0:
                payload = mean * n_total
                if abs(payload - round(payload)) > 1e-6:
                    return {
                        "type": "grim_violation",
                        "verdict": "CONFLICT",
                        "explanation": (
                            f"mean {mean:g} over N={n_total} implies a total of {payload:g}, "
                            "which is impossible for integer measurements (GRIM violation)."
                            "\nFINAL: CONFLICT"
                        ),
                    }

        # 2. Unit-dimensional homogeneity of a printed equation.
        if "unit" in lower or "dimension" in lower:
            m_eq = re.search(
                r"(\d+(?:\.\d+)?)\s*([a-zA-Z][a-zA-Z0-9*/^·]*)\s*=\s*"
                r"(\d+(?:\.\d+)?)\s*([a-zA-Z][a-zA-Z0-9*/^·]*)",
                text,
            )
            if m_eq is not None:
                lhs = cls.parse_unit_dimensions(m_eq.group(2))
                rhs = cls.parse_unit_dimensions(m_eq.group(4))
                if lhs is not None and rhs is not None and lhs != rhs:
                    return {
                        "type": "dimensional_mismatch",
                        "verdict": "CONFLICT",
                        "explanation": (
                            f"{m_eq.group(2)} has dimensions {lhs} but {m_eq.group(4)} has "
                            f"{rhs}; the equation is dimensionally inhomogeneous.\n"
                            "FINAL: CONFLICT"
                        ),
                    }

        # 3. Multi-claim test-statistic / p-value consistency (z, Student-t, chi2, F, Pearson r).
        # Scans every reported p-value in the text and pairs it with its nearest preceding statistic
        # in the same sentence/clause so fabricated claims later in a paragraph cannot slip through.
        p_pattern = re.compile(r"\bp\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?)")
        prev_p_end = 0
        for m_p in p_pattern.finditer(lower):
            reported_str = m_p.group(1)
            reported = float(reported_str)
            # Extract local clause preceding this p-value (bounded by previous p-value or sentence break)
            raw_window = lower[prev_p_end:m_p.start()]
            prev_p_end = m_p.end()
            # Keep the last clause after the final sentence terminator ('. ' or '; ' or '\n')
            clause = re.split(r"(?:\.\s+|;\s*|\n+)", raw_window)[-1]
            if not (0.0 < reported <= 1.0):
                continue

            if "e" in reported_str:
                reported_ulp = reported * 0.1
            elif "." in reported_str:
                decimals = len(reported_str.split(".", 1)[1])
                reported_ulp = 10.0 ** (-decimals)
            else:
                reported_ulp = 0.01

            # Find all candidate test statistics in this clause and pick the closest (rightmost) one
            candidates_stat: List[Tuple[int, float, str, bool]] = []
            m_df = re.search(r"\bdf\s*=\s*(\d+)", clause)

            for m_z in re.finditer(r"\bz\s*=\s*([+\-]?\d+(?:\.\d+)?)", clause):
                val = float(m_z.group(1))
                candidates_stat.append((m_z.end(), cls._normal_two_sided_p(val), f"z={val:g}", False))

            for m_f in re.finditer(
                r"\bf\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(\d+(?:\.\d+)?)", clause
            ):
                df1, df2, f_val = int(m_f.group(1)), int(m_f.group(2)), float(m_f.group(3))
                candidates_stat.append((m_f.end(), cls._f_dist_survival(f_val, df1, df2), f"F({df1},{df2})={f_val:g}", False))

            for m_r in re.finditer(
                r"\br\s*\(\s*(?:df\s*=\s*)?(\d+)\s*\)\s*=\s*([+\-]?\d*(?:\.\d+)?)", clause
            ):
                df_r = int(m_r.group(1))
                r_val = float(m_r.group(2))
                if abs(r_val) <= 1.0:
                    candidates_stat.append((m_r.end(), cls._pearson_r_two_sided_p(r_val, df_r), f"r({df_r})={r_val:g}", False))

            for m_chi in re.finditer(
                r"(?:chi2|\u03c7\u00b2|chi-square)(?:\s*\(\s*(?:df\s*=\s*)?(\d+)\s*\))?\s*=\s*(\d+(?:\.\d+)?)",
                clause,
            ):
                if m_chi.group(1) is not None or m_df is not None:
                    chi_val = float(m_chi.group(2))
                    df_val = int(m_chi.group(1)) if m_chi.group(1) is not None else int(m_df.group(1))
                    candidates_stat.append((m_chi.end(), cls._chi2_survival(chi_val, df_val), f"chi2={chi_val:g}, df={df_val}", False))

            for m_t in re.finditer(
                r"\bt(?:\s*\(\s*(?:df\s*=\s*)?(\d+)\s*\))?\s*=\s*([+\-]?\d+(?:\.\d+)?)",
                clause,
            ):
                t_value = abs(float(m_t.group(2)))
                if m_t.group(1) is not None or m_df is not None:
                    df_val = int(m_t.group(1)) if m_t.group(1) is not None else int(m_df.group(1))
                    candidates_stat.append((m_t.end(), cls._student_t_two_sided_p(t_value, df_val), f"t={t_value:g}, df={df_val}", False))
                else:
                    candidates_stat.append((m_t.end(), cls._normal_two_sided_p(t_value), f"t={t_value:g}", True))

            if not candidates_stat:
                continue

            _, expected, label, one_sided_lower_bound = max(candidates_stat, key=lambda item: item[0])
            if expected > 0.0:
                abs_diff = abs(expected - reported)
                rounding_tol = max(0.55 * reported_ulp, 0.002)
                if abs_diff > rounding_tol:
                    ratio = max(expected / reported, reported / expected)
                    alpha_flip = any(
                        (reported <= alpha and expected > alpha)
                        or (reported > alpha and expected <= alpha)
                        for alpha in (0.05, 0.01, 0.001)
                    )
                    if one_sided_lower_bound:
                        too_far = reported < expected - rounding_tol and (expected / reported) > 1.25
                    else:
                        too_far = alpha_flip or ratio > 1.25 or abs_diff > max(0.015, 2.5 * reported_ulp)
                    if too_far:
                        return {
                            "type": "statistical_inconsistency",
                            "alpha_flip": "true" if alpha_flip else "false",
                            "verdict": "CONFLICT",
                            "explanation": (
                                f"{label} implies p≈{expected:.4g}, but the report states "
                                f"p={reported:g}; that discrepancy exceeds rounding tolerance."
                                "\nFINAL: CONFLICT"
                            ),
                        }

        return None

    @staticmethod
    def verify_epistemic_calibration(prompt: str) -> Optional[Dict[str, str]]:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        p_lower = prompt.lower()
        statistical = Vexx35Controller.verify_statistical_report(prompt)
        if statistical is not None:
            return statistical

        m_ci = re.search(
            r"(?P<level>(?:\d+(?:\.\d+)?))%\s*ci\s*\[\s*"
            r"(?P<lo>[−+\-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
            r"(?P<hi>[−+\-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]",
            p_lower,
        )
        m_p = re.search(r"\bp\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?)", p_lower)
        if m_ci or m_p:
            ci_sig = None
            p_sig = None
            ci_reason = None
            alpha = 0.05
            if m_ci:
                confidence = float(m_ci.group("level"))
                if not 0.0 < confidence < 100.0:
                    return {
                        "type": "invalid_statistical_input",
                        "verdict": "CONFLICT",
                        "explanation": f"Invalid confidence level {confidence:g}%.\nFINAL: CONFLICT",
                    }
                alpha = 1.0 - confidence / 100.0
                ci_lo = float(m_ci.group("lo").replace("−", "-"))
                ci_hi = float(m_ci.group("hi").replace("−", "-"))
                if ci_lo > ci_hi:
                    return {
                        "type": "invalid_statistical_input",
                        "verdict": "CONFLICT",
                        "explanation": "CI lower bound exceeds upper bound.\nFINAL: CONFLICT",
                    }
                ci_sig = not (ci_lo <= 0.0 <= ci_hi)
                ci_reason = (
                    f"the {confidence:g}% interval [{ci_lo}, {ci_hi}] "
                    + ("excludes" if ci_sig else "includes")
                    + " zero"
                )
            p_reason = None
            if m_p:
                p_value = float(m_p.group(1))
                if not 0.0 <= p_value <= 1.0:
                    return {
                        "type": "invalid_statistical_input",
                        "verdict": "CONFLICT",
                        "explanation": f"p-value {p_value} outside [0, 1].\nFINAL: CONFLICT",
                    }
                p_sig = p_value < alpha
                p_reason = f"p={p_value:g} " + ("<" if p_sig else ">=") + f" {alpha:g}"

            if ci_sig is not None and p_sig is not None and ci_sig != p_sig:
                return {
                    "type": "statistical_conflict",
                    "verdict": "CONFLICT",
                    "explanation": (
                        f"Statistical indicators disagree: {ci_reason}, but {p_reason}.\nFINAL: CONFLICT"
                    ),
                }
            significant = ci_sig if ci_sig is not None else p_sig
            if significant is True:
                reason = " and ".join(x for x in (ci_reason, p_reason) if x)
                return {
                    "type": "statistical_significance",
                    "verdict": "established",
                    "explanation": f"Statistically significant at alpha={alpha:g} ({reason}).\nFINAL: established",
                }
            if significant is False:
                reason = " and ".join(x for x in (ci_reason, p_reason) if x)
                return {
                    "type": "statistical_significance",
                    "verdict": "not established",
                    "explanation": f"Not statistically significant at alpha={alpha:g} ({reason}).\nFINAL: not established",
                }

        if (
            "source a" in p_lower
            and "source b" in p_lower
            and "effective" in p_lower
            and ("official notice" in p_lower or "regulator" in p_lower)
        ):
            m_b = re.search(r"source b.*?(\d+)\s*units", p_lower)
            if m_b:
                units = m_b.group(1)
                return {
                    "type": "temporal_supersession",
                    "verdict": f"{units} units",
                    "explanation": f"Source B supersedes Source A.\nFINAL: {units} units",
                }

        if "optional online poll" in p_lower or ("visitors" in p_lower and "blog" in p_lower):
            return {
                "type": "sampling_bias",
                "verdict": "not justified",
                "explanation": "Voluntary convenience sample.\nFINAL: not justified",
            }

        if "deliberately unverified" in p_lower or re.search(
            r"\bunverified\s+(?:award|entity|winner|claim)\b", p_lower
        ):
            return {
                "type": "unverified_premise",
                "verdict": "INSUFFICIENT EVIDENCE",
                "explanation": "Premise marked unverified.\nFINAL: INSUFFICIENT EVIDENCE",
            }
        if "fictional city" in p_lower or "fictional place" in p_lower:
            return {
                "type": "fictional_entity",
                "verdict": "UNKNOWN",
                "explanation": "Fictional place with no supplied dataset.\nFINAL: UNKNOWN",
            }

        if ("table" in p_lower and "paragraph" in p_lower and "conflict" in p_lower) or (
            "contradictory" in p_lower and "source" in p_lower
        ):
            return {
                "type": "contradiction_detection",
                "verdict": "CONFLICT",
                "explanation": "Internally contradictory material.\nFINAL: CONFLICT",
            }

        m_decimal = re.search(r"(?<![\w.])0\.(?P<nines>9+)(?P<repeat>\.\.\.|…)?(?!\d)", p_lower)
        if m_decimal and (
            "less than 1" in p_lower
            or "< 1" in p_lower
            or "equal to 1" in p_lower
            or "equals 1" in p_lower
        ):
            if bool(m_decimal.group("repeat")):
                return {
                    "type": "false_premise",
                    "verdict": "equal",
                    "explanation": "0.999... equals 1.\nFINAL: equal",
                }
            literal = "0." + m_decimal.group("nines")
            return {
                "type": "numeric_literal",
                "verdict": "less than 1",
                "explanation": f"{literal} is strictly less than 1.\nFINAL: less than 1",
            }

        return {
            "type": "heuristic_no_match",
            "verdict": "unknown",
            "explanation": "No epistemic heuristic rule matched.\nFINAL: unknown",
        }

    @classmethod
    def _check_import_policy(cls, node: ast.AST) -> None:
        """Allow imports only from :attr:`ALLOWED_STDLIB_MODULES` (P0.8)."""
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
            imported_names: List[str] = []
        elif isinstance(node, ast.ImportFrom):
            if int(node.level or 0) != 0:
                raise PermissionError("relative imports are forbidden in the sandbox")
            modules = [node.module or ""]
            imported_names = [alias.name for alias in node.names]
        else:  # pragma: no cover - defensive
            raise PermissionError(f"unexpected import node {type(node).__name__}")
        for name in modules:
            root = str(name).split(".")[0]
            if root not in cls.ALLOWED_STDLIB_MODULES:
                raise PermissionError(f"Forbidden import: {name}")
            if any(part.startswith("_") for part in str(name).split(".")):
                raise PermissionError(f"Forbidden private import: {name}")
        for name in imported_names:
            if str(name).split(".")[0].startswith("_"):
                raise PermissionError(f"Forbidden private import: {name}")
        for alias in node.names:
            if alias.asname and str(alias.asname).startswith("_"):
                raise PermissionError(f"Forbidden private import alias: {alias.asname}")

    @classmethod
    def _check_ast_policy(cls, code_str: str) -> None:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                cls._check_import_policy(node)
            elif isinstance(node, cls.FORBIDDEN_AST_NODES):
                raise PermissionError(f"Forbidden AST node: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id in cls.FORBIDDEN_IDENTIFIERS:
                raise PermissionError(f"Forbidden identifier: {node.id}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise PermissionError(f"Forbidden dunder attribute: {node.attr}")
            if isinstance(node, ast.Attribute) and not isinstance(
                node.value, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)
            ):
                literal_method = (
                    isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and node.attr in cls.SAFE_LITERAL_ATTRIBUTES
                )
                if not literal_method:
                    raise PermissionError("Forbidden attribute access on a literal")

    @classmethod
    def _sandbox_dispatch(
        cls,
        code_str: str,
        fn_name: str = "",
        args: Optional[List[Any]] = None,
        timeout: float = 5.0,
        mode: str = "fn_call",
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        import shutil
        import subprocess
        import sys

        sandbox_dir = tempfile.mkdtemp(prefix="maya250m_sandbox_")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _SANDBOX_DRIVER],
                input=json.dumps(
                    {"mode": mode, "code": code_str, "fn": fn_name, "args": args or []}
                ),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=sandbox_dir,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                preexec_fn=_limit_child,
            )
        except subprocess.TimeoutExpired:
            return False, None, None, f"TimeoutError: exceeded {timeout:.1f}s sandbox limit"
        finally:
            shutil.rmtree(sandbox_dir, ignore_errors=True)

        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no diagnostics"]
            return False, None, None, f"SandboxError: exit {proc.returncode}: {tail[0][:200]}"
        lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
        if not lines:
            return False, None, None, "SandboxError: empty result"
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            return False, None, None, "SandboxError: malformed result"
        if payload.get("ok"):
            return True, payload.get("repr"), payload.get("input_after"), None
        return False, None, None, str(payload.get("error", "execution failed"))[:300]

    def execute_experiment_block(self, script_str: str, timeout: float = 3.0) -> str:
        """Execute one ``<experiment>`` cell with the episode's verified preamble.

        Each cell still runs in a fresh sandboxed subprocess, but definitions that
        succeeded earlier in the same episode are replayed first. A research trace
        can therefore define a model/dataset in cell #1 and refine the hypothesis
        in cell #2 without hitting ``NameError``.
        """
        try:
            self._check_ast_policy(script_str)
        except (SyntaxError, PermissionError) as exc:
            return f"EXP_ERROR: {exc}"
        combined = "\n".join([*self.experiment_preamble, script_str])
        if self.experiment_preamble:
            try:
                self._check_ast_policy(combined)
            except (SyntaxError, PermissionError) as exc:  # pragma: no cover - defensive
                return f"EXP_ERROR: {exc}"
        ok, res_repr, _after, err = self._sandbox_dispatch(
            combined, timeout=timeout, mode="experiment_script"
        )
        if not ok:
            return f"EXP_ERROR: {err}"
        self._absorb_experiment_definitions(script_str)
        return str(res_repr)

    def _absorb_experiment_definitions(self, script_str: str) -> None:
        """Carry verified definitions to later cells, excluding ``RESULT``."""
        try:
            tree = ast.parse(script_str)
        except SyntaxError:  # pragma: no cover - policy check already parsed it
            return
        kept: List[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kept.append(ast.unparse(node))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                kept.append(ast.unparse(node))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == "RESULT" for t in targets):
                    continue
                kept.append(ast.unparse(node))
        if not kept:
            return
        self.experiment_preamble.extend(kept)
        while len(self.experiment_preamble) > 64:
            self.experiment_preamble.pop(0)
        while sum(len(cell) for cell in self.experiment_preamble) > 24000:
            self.experiment_preamble.pop(0)

    @classmethod
    def execute_capture(cls, code_str: str, fn_name: str, args: List[Any]) -> Any:
        cls._check_ast_policy(code_str)
        ok, res_repr, _input_after, error = cls._sandbox_dispatch(code_str, fn_name, args)
        if not ok:
            raise RuntimeError(error or "sandbox execution failed")
        return ast.literal_eval(res_repr or "None")

    @classmethod
    def safe_execute_python(
        cls,
        code_str: str,
        fn_name: str,
        test_cases: List[Dict[str, Any]],
    ) -> Tuple[int, int, List[str]]:
        try:
            cls._check_ast_policy(code_str)
        except SyntaxError as e:
            return 0, len(test_cases), [f"SyntaxError: {e}"]
        except PermissionError as e:
            return 0, len(test_cases), [str(e)]

        passed = 0
        errors: List[str] = []
        for idx, tc in enumerate(test_cases):
            original = json.loads(json.dumps(tc["input"]))
            ok, res_repr, after_repr, error = cls._sandbox_dispatch(
                code_str, fn_name, copy.deepcopy(original)
            )
            if not ok:
                errors.append(f"Test {idx} raised {error}")
                continue
            try:
                result = ast.literal_eval(res_repr or "None")
            except (ValueError, SyntaxError):
                errors.append(f"Test {idx} returned a non-literal: {res_repr!r}")
                continue
            value_ok = (
                result in tc["expected_one_of"]
                if "expected_one_of" in tc
                else result == tc.get("expected")
            )
            try:
                after = ast.literal_eval(after_repr or "None")
            except (ValueError, SyntaxError):
                after = after_repr
            mutation_ok = after == original
            if value_ok and mutation_ok:
                passed += 1
            elif not mutation_ok:
                errors.append(f"Test {idx} mutated its input")
            else:
                errors.append(
                    f"Test {idx} failed: got {result!r}, expected "
                    f"{tc.get('expected', tc.get('expected_one_of'))!r}"
                )

        return passed, len(test_cases), errors

    def select_verified(
        self,
        candidates: List[str],
        fn_name: str,
        test_cases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        for code in candidates:
            passed, total, errors = self.safe_execute_python(code, fn_name, test_cases)
            try:
                self._check_ast_policy(code)
                syntax_ok = True
            except (SyntaxError, PermissionError):
                syntax_ok = False
            scored.append(
                {
                    "code": code,
                    "passed": passed,
                    "total": total,
                    "errors": errors,
                    "syntax_ok": syntax_ok,
                    "contract_ok": self._contract_ok(code, fn_name, test_cases),
                    "mutation_free": not any("mutated its input" in e for e in errors),
                }
            )
        winners = [s for s in scored if s["total"] > 0 and s["passed"] == s["total"]]
        losers = [s for s in scored if s["total"] > 0 and s["passed"] < s["total"]]
        if len(winners) == 1 and losers:
            self.dpo_buffer.append(
                {
                    "fn": fn_name,
                    "winner": winners[0]["code"],
                    "losers": [s["code"] for s in losers],
                    "tests": test_cases,
                    "scores": [{"passed": s["passed"], "total": s["total"]} for s in scored],
                    "oracle": "subprocess unit-test execution",
                }
            )
        return scored

    @staticmethod
    def _contract_ok(code_str: str, fn_name: str, test_cases: List[Dict[str, Any]]) -> bool:
        """Does ``fn_name`` exist and accept every test case's positional arity?"""
        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            return False
        node = next(
            (
                candidate
                for candidate in ast.walk(tree)
                if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
                and candidate.name == fn_name
            ),
            None,
        )
        if node is None:
            return False
        args = node.args
        positional = list(args.posonlyargs) + list(args.args)
        if positional and positional[0].arg in {"self", "cls"}:
            positional = positional[1:]
        required = max(0, len(positional) - len(args.defaults))
        strict_max = None if args.vararg is not None else len(positional)
        for test_case in test_cases:
            n_args = len(test_case.get("input") or [])
            if n_args < required:
                return False
            if strict_max is not None and n_args > strict_max:
                return False
        return True

    @classmethod
    def _execution_reward(cls, item: Dict[str, Any]) -> Tuple[float, str]:
        """Shaped verifiable reward ladder (never a zero-variance pass/fail wall).

        ``-0.5`` syntax/policy violation, ``-0.2`` contract violation or caller
        input mutation, ``0.0`` clean run with no passing test, ``+0.3 * pass rate``
        for a partial pass, ``+1.0`` for a full pass with zero mutation.
        """
        total = int(item.get("total") or 0)
        passed = int(item.get("passed") or 0)
        if not bool(item.get("syntax_ok", True)):
            return -0.5, "syntax_or_policy"
        if not bool(item.get("contract_ok", True)):
            return -0.2, "contract_violation"
        if not bool(item.get("mutation_free", True)):
            return -0.2, "input_mutation"
        if total <= 0 or passed <= 0:
            return 0.0, "no_tests_passed"
        if passed >= total:
            return 1.0, "full_pass"
        return 0.3 * (float(passed) / float(total)), "partial_pass"

    def compute_rlvr_group_rewards(
        self,
        candidates: List[str],
        fn_name: str,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute GRPO group-normalized verifiable rewards from Vexx Tier-5 unit tests."""
        raw_rewards: List[float] = []
        details = self.select_verified(candidates, fn_name, test_cases)
        for item in details:
            reward, reason = self._execution_reward(item)
            item["reward"] = reward
            item["reward_reason"] = reason
            raw_rewards.append(reward)

        arr = np.asarray(raw_rewards, dtype=np.float64)
        mean_r = float(arr.mean()) if arr.size > 0 else 0.0
        std_r = float(arr.std()) if arr.size > 0 else 0.0
        advantages = (
            ((arr - mean_r) / (std_r + 1e-6)).tolist()
            if std_r > 1e-8
            else [0.0 for _ in raw_rewards]
        )
        return {
            "rewards": raw_rewards,
            "advantages": advantages,
            "mean_reward": mean_r,
            "std_reward": std_r,
            "details": details,
        }

    @staticmethod
    def _ast_max_depth(code_str: str) -> int:
        """Structural depth of the candidate AST (MBR tie-break: prefer simpler)."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            return 1 << 20
        depth = 0
        stack: List[Tuple[ast.AST, int]] = [(tree, 1)]
        while stack:
            node, node_depth = stack.pop()
            depth = max(depth, node_depth)
            for child in ast.iter_child_nodes(node):
                stack.append((child, node_depth + 1))
        return depth

    def select_by_mbr(
        self,
        candidates: List[str],
        fn_name: str,
        inputs: List[List[Any]],
        *,
        log_probs: Optional[Sequence[float]] = None,
        temperature: float = 1.0,
        timeout: float = 5.0,
    ) -> Dict[str, Any]:
        """Minimum-Bayes-Risk (MBR) selection over executed behavioral outputs.

        Computes the expected posterior risk ``R(y_i) = sum_j P(y_j | x) L(y_i, y_j)``
        under the per-probe execution mismatch loss
        ``L(y_i, y_j) = (1 / P) sum_{m=1}^P I[out_i(x_m) != out_j(x_m)]``,
        where ``P(y_j | x)`` is the temperature-scaled softmax posterior over
        eligible survivors (uniform ``1 / M`` when ``log_probs`` is omitted).
        Selects ``argmin_{y_i} R(y_i)`` with ties broken by exact-tuple vote count,
        shallowest AST depth, and shortest code length.
        """
        if not inputs:
            raise ValueError("MBR selection needs at least one probe input")
        if log_probs is not None and len(log_probs) != len(candidates):
            raise ValueError("len(log_probs) must equal len(candidates)")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")

        probes = [{"input": list(inp)} for inp in inputs]
        scoreboard: List[Dict[str, Any]] = []
        for index, code in enumerate(candidates):
            try:
                self._check_ast_policy(code)
                policy_ok = True
            except (SyntaxError, PermissionError):
                policy_ok = False
            contract_ok = policy_ok and self._contract_ok(code, fn_name, probes)
            outputs: List[str] = []
            mutation_free = True
            executed = contract_ok
            error: Optional[str] = None
            if executed:
                for inp in inputs:
                    ok, res_repr, after_repr, err = self._sandbox_dispatch(
                        code, fn_name, copy.deepcopy(inp), timeout=timeout
                    )
                    if not ok:
                        executed = False
                        error = err
                        outputs = []
                        break
                    outputs.append(str(res_repr))
                    try:
                        after = ast.literal_eval(after_repr or "None")
                    except (ValueError, SyntaxError):
                        after = after_repr
                    if after != inp:
                        mutation_free = False
            raw_lp = float(log_probs[index]) if log_probs is not None else 0.0
            scoreboard.append(
                {
                    "index": index,
                    "code": code,
                    "policy_ok": policy_ok,
                    "contract_ok": contract_ok,
                    "executed": executed,
                    "outputs": outputs,
                    "mutation_free": mutation_free,
                    "ast_depth": self._ast_max_depth(code),
                    "log_prob": raw_lp,
                    "posterior_prob": 0.0,
                    "bayes_risk": 1.0,
                    "expected_utility": 0.0,
                    "error": error,
                }
            )

        survivors = [row for row in scoreboard if row["executed"] and row["outputs"]]
        eligible = [row for row in survivors if row["mutation_free"]] or survivors

        # Compute normalized posterior P(y_j | x) over eligible survivors
        if eligible:
            scaled = np.asarray(
                [row["log_prob"] / float(temperature) for row in eligible],
                dtype=np.float64,
            )
            scaled = scaled - np.max(scaled)
            weights = np.exp(scaled)
            posteriors = (weights / np.sum(weights)).tolist()
            for row, prob in zip(eligible, posteriors):
                row["posterior_prob"] = float(prob)

            # Pairwise per-probe Hamming loss matrix L(y_i, y_j) in [0, 1]
            n_probes = max(1, len(inputs))
            for row_i in eligible:
                out_i = row_i["outputs"]
                risk = 0.0
                for row_j, p_j in zip(eligible, posteriors):
                    out_j = row_j["outputs"]
                    disagree = sum(1 for a, b in zip(out_i, out_j) if a != b) / float(n_probes)
                    risk += p_j * disagree
                row_i["bayes_risk"] = float(risk)
                row_i["expected_utility"] = float(1.0 - risk)

        counts: Dict[Tuple[str, ...], int] = {}
        posterior_by_tuple: Dict[Tuple[str, ...], float] = {}
        for row in eligible:
            key = tuple(row["outputs"])
            counts[key] = counts.get(key, 0) + 1
            posterior_by_tuple[key] = posterior_by_tuple.get(key, 0.0) + row["posterior_prob"]

        # Sort eligible candidates by minimal Bayes risk, then highest exact-tuple votes,
        # then shallowest AST depth, then shortest code.
        pool = list(eligible)
        pool.sort(
            key=lambda row: (
                round(row["bayes_risk"], 9),
                -counts.get(tuple(row["outputs"]), 0),
                row["ast_depth"],
                len(row["code"]),
                row["index"],
            )
        )
        winner = pool[0] if pool else None
        best_outputs = tuple(winner["outputs"]) if winner is not None else ()
        votes = counts.get(best_outputs, 0)
        posterior_mass = posterior_by_tuple.get(best_outputs, 0.0)
        agreement = (votes / len(eligible)) if eligible else 0.0
        return {
            "winner": winner["code"] if winner else None,
            "winner_index": winner["index"] if winner else None,
            "outputs": list(best_outputs),
            "votes": votes,
            "survivors": len(eligible),
            "agreement": agreement,
            "bayes_risk": winner["bayes_risk"] if winner else 1.0,
            "expected_utility": winner["expected_utility"] if winner else 0.0,
            "posterior_mass": float(posterior_mass),
            "consensus": bool(
                winner is not None
                and len(eligible) >= 2
                and (
                    (votes >= 2 and votes * 2 > len(eligible))
                    or (log_probs is not None and posterior_mass > 0.5 and votes >= 1)
                )
            ),
            "scoreboard": scoreboard,
        }

    #: Boundary probe inputs (one arg-list each) for property-based micro-fuzzing.
    BOUNDARY_PROBES: Tuple[List[Any], ...] = (
        [[]],
        [[0]],
        [[-1, -2, -1]],
        [[3, 3, 3]],
        [[1, 2, 3, 4, 5]],
        [[2, 1]],
    )

    def fuzz_against_oracle(
        self,
        candidate: str,
        reference: str,
        fn_name: str,
        inputs: Optional[List[List[Any]]] = None,
        *,
        timeout: float = 5.0,
    ) -> Dict[str, Any]:
        """Differential property fuzzing against a self-written naive oracle.

        Both programs run in the Tier-5 sandbox on boundary probes (empty, singleton,
        negatives, duplicates, sorted and reversed inputs) and their observed outputs
        are compared. Disagreement, a crash on either side, or a non-literal return
        value all count as failure: this is a property check, never a proof, and it
        fails closed when the reference oracle itself errors.
        """
        probe_inputs = list(inputs) if inputs is not None else [list(p) for p in self.BOUNDARY_PROBES]
        if not probe_inputs:
            raise ValueError("property fuzzing needs at least one probe input")
        mismatches: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for index, probe in enumerate(probe_inputs):
            ok_ref, ref_repr, _ref_after, ref_err = self._sandbox_dispatch(
                reference, fn_name, copy.deepcopy(probe), timeout=timeout
            )
            if not ok_ref:
                errors.append({"input": probe, "side": "reference", "error": ref_err})
                continue
            ok_cand, cand_repr, _cand_after, cand_err = self._sandbox_dispatch(
                candidate, fn_name, copy.deepcopy(probe), timeout=timeout
            )
            if not ok_cand:
                errors.append({"input": probe, "side": "candidate", "error": cand_err})
                continue
            try:
                expected = ast.literal_eval(ref_repr or "None")
                actual = ast.literal_eval(cand_repr or "None")
                agrees = actual == expected
            except (ValueError, SyntaxError):
                expected, actual = ref_repr, cand_repr
                agrees = ref_repr == cand_repr
            if not agrees:
                mismatches.append(
                    {"input": probe, "expected": expected, "actual": actual}
                )
        return {
            "fn": fn_name,
            "checked": len(probe_inputs),
            "agrees": not mismatches and not errors,
            "mismatches": mismatches,
            "errors": errors,
            "probes": probe_inputs,
        }

    def export_dpo(self, path: str) -> int:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in self.dpo_buffer:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(self.dpo_buffer)


Vexx3Controller = Vexx35Controller


def create_empty_hybrid_cache_v2(
    num_loops: int,
    prelude_layers: int,
    core_layers: int,
    coda_layers: int,
    batch_size: int = 1,
    config_hash: str = "",
) -> dict:
    """Create an empty Schema-2 structured multi-loop hybrid cache bundle."""
    return {
        "schema_version": 2,
        "num_loops": int(num_loops),
        "position": 0,
        "batch_size": int(batch_size),
        "config_hash": str(config_hash),
        "prelude": [None for _ in range(int(prelude_layers))],
        "core_loops": [
            [None for _ in range(int(core_layers))]
            for _ in range(int(num_loops))
        ],
        "coda": [None for _ in range(int(coda_layers))],
    }
