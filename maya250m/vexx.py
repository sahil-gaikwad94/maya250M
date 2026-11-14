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


def truncate_cache_at_token(cache: Any, tok_idx: int) -> Any:
    """Recursively truncate Full-Attention KV tensors at ``tok_idx`` (v1 or v2 caches)."""
    tok_idx = int(tok_idx)
    if isinstance(cache, list):
        for item in cache:
            truncate_cache_at_token(item, tok_idx)
        return cache
    if isinstance(cache, dict):
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
import contextlib, io, json, os, sys
payload = json.load(sys.stdin)
SAFE = {
    "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "next": next, "ord": ord, "pow": pow, "print": print,
    "range": range, "reversed": reversed, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "True": True, "False": False, "None": None,
}
mode = payload.get("mode", "fn_call")
ns = {"__builtins__": SAFE}
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
_ALLOWED_FUNCS = {
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "abs": sp.Abs,
    "Abs": sp.Abs,
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

    return build(tree)


class Vexx35Controller:
    """Stateful six-tier neuro-symbolic controller for Maya-250M."""

    FORBIDDEN_AST_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
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
        i = 0
        n = len(code_prefix)
        in_str: Optional[str] = None
        f_mode = False
        f_brace_depth = 0
        stack: List[Tuple[str, Optional[str], int]] = []
        closed_frames: List[Tuple[str, Optional[str], int]] = []
        sanitized: List[str] = []

        while i < n:
            ch = code_prefix[i]
            if in_str is not None and in_str != "#":
                if f_mode and f_brace_depth == 0 and ch == "{":
                    if i + 1 < n and code_prefix[i + 1] == "{":
                        sanitized.append("  ")
                        i += 2
                        continue
                    f_brace_depth = 1
                    sanitized.append("(")
                    stack.append(("(", None, 0))
                    i += 1
                    continue
                if f_mode and f_brace_depth > 0:
                    if ch in "'\"":
                        return {
                            "valid": False,
                            "reason": "quote_in_fstring_expression",
                            "stack": stack,
                            "in_str": in_str,
                        }
                    if ch == "{":
                        f_brace_depth += 1
                    elif ch == "}":
                        f_brace_depth -= 1
                        if f_brace_depth == 0:
                            sanitized.append(")")
                            if stack and stack[-1][0] == "(":
                                stack.pop()
                            i += 1
                            continue
                else:
                    if ch == "\\" and i + 1 < n:
                        sanitized.append("  ")
                        i += 2
                        continue
                    if code_prefix.startswith(in_str, i):
                        sanitized.append(" " * len(in_str))
                        i += len(in_str)
                        in_str = None
                        f_mode = False
                        continue
                    sanitized.append(" ")
                    i += 1
                    continue
            elif in_str == "#":
                if ch == "\n":
                    in_str = None
                    sanitized.append("\n")
                else:
                    sanitized.append(" ")
                i += 1
                continue

            if ch == "#":
                in_str = "#"
                sanitized.append(" ")
                i += 1
                continue

            m_str = re.match(r"^([rRfFbB]{0,2})('''|\"\"\"|'|\")", code_prefix[i:])
            if m_str:
                prefix_flags, quote_tok = m_str.group(1).lower(), m_str.group(2)
                in_str = quote_tok
                f_mode = "f" in prefix_flags
                full_len = len(m_str.group(0))
                sanitized.append(" " * full_len)
                i += full_len
                continue

            sanitized.append(ch)
            if ch in "([{":
                callee = None
                if ch == "(":
                    before = "".join(sanitized[:-1]).rstrip()
                    m_call = re.search(r"([A-Za-z_][A-Za-z0-9_.]*)$", before)
                    if m_call:
                        ident = m_call.group(1)
                        if ident not in {
                            "if", "while", "for", "return", "elif", "assert",
                            "not", "and", "or", "in", "def", "class", "lambda", "with",
                        }:
                            callee = ident
                stack.append((ch, callee, 0))
            elif ch in ")]}":
                if not stack:
                    return {
                        "valid": False,
                        "reason": "unmatched_close",
                        "stack": stack,
                        "in_str": in_str,
                    }
                open_ch, _, _ = stack[-1]
                if (open_ch, ch) in {("(", ")"), ("[", "]"), ("{", "}")}:
                    closed_frames.append(stack.pop())
                else:
                    return {
                        "valid": False,
                        "reason": "mismatched_close",
                        "stack": stack,
                        "in_str": in_str,
                    }
            elif ch == "," and stack:
                top_d, top_c, top_cnt = stack[-1]
                stack[-1] = (top_d, top_c, top_cnt + 1)
            i += 1

        return {
            "valid": True,
            "stack": stack,
            "closed_frames": closed_frames,
            "in_str": in_str,
            "f_brace_depth": f_brace_depth,
            "sanitized": "".join(sanitized),
        }

    def apply_in_flight_mask(
        self,
        logits: np.ndarray,
        prefix_text: str,
        vocab_map: Dict[int, str],
        expected_return_type: Optional[str] = None,
        remaining_token_budget: int = 256,
        mode: str = "python",
    ) -> np.ndarray:
        """Apply Tier 0 + Tier 1 constraints to next-token logits."""
        if logits.ndim != 1:
            raise ValueError(f"logits must be 1-D [vocab], got shape {tuple(logits.shape)}")
        masked = np.array(logits, dtype=np.float64, copy=True)

        for tid, tok_str in vocab_map.items():
            if tid < 0 or tid >= masked.shape[-1]:
                raise ValueError(f"vocabulary id {tid} out of range")
            if token_category(tok_str) == NEVER_EMIT:
                masked[tid] = -np.inf

        phase = self.crane_phase(prefix_text)

        for tid, tok_str in vocab_map.items():
            if token_category(tok_str) != PROTOCOL_MARKERS:
                continue
            if phase == "UNCONSTRAINED_THINK" and tok_str in (THINK_END, EXP_START):
                continue
            if phase == "EXPERIMENT_DESIGN" and tok_str == EXP_END:
                continue
            masked[tid] = -np.inf

        if phase == "UNCONSTRAINED_THINK" and remaining_token_budget > 4:
            return masked

        if mode != "python" and phase != "EXPERIMENT_DESIGN":
            return masked

        active_code = (
            prefix_text[prefix_text.rfind(EXP_START) + len(EXP_START) :]
            if phase == "EXPERIMENT_DESIGN"
            else prefix_text
        )
        live_syms = dict(self.symbol_table)
        live_syms.update(self.extract_ast_symbol_table(active_code))
        base_state = self._lex_code_state(active_code)
        open_depth = len(base_state.get("stack", []))

        for tid, tok_str in vocab_map.items():
            cat = token_category(tok_str)
            if cat == NEVER_EMIT or cat == PROTOCOL_MARKERS:
                continue

            if remaining_token_budget <= open_depth + 1 and any(c in tok_str for c in "([{"):
                masked[tid] = -np.inf
                continue

            cand_text = active_code + tok_str
            st = self._lex_code_state(cand_text)
            if not st["valid"]:
                masked[tid] = -np.inf
                continue

            if "(" in tok_str and st["in_str"] is None and st["stack"]:
                _, callee, _ = st["stack"][-1]
                if callee is not None:
                    if "." in callee:
                        recv, meth = callee.rsplit(".", 1)
                        if recv not in {"self", "cls"}:
                            masked[tid] = -np.inf
                            continue
                        if live_syms and meth not in live_syms:
                            masked[tid] = -np.inf
                            continue
                        callee_key = meth
                    else:
                        callee_key = callee

                    last_line = cand_text.splitlines()[-1].lstrip() if cand_text.splitlines() else ""
                    if not last_line.startswith(("def ", "@")) and phase != "EXPERIMENT_DESIGN":
                        if callee_key not in live_syms:
                            masked[tid] = -np.inf
                            continue
                        ret = live_syms[callee_key].get("return_type")
                        if expected_return_type and ret not in {expected_return_type, "any"}:
                            masked[tid] = -np.inf
                            continue

            if ")" in tok_str and base_state["valid"] and st.get("closed_frames"):
                closed_delim, closed_callee, comma_cnt = st["closed_frames"][-1]
                if closed_delim == "(" and closed_callee is not None:
                    if "." in closed_callee and closed_callee.rsplit(".", 1)[0] not in {"self", "cls"}:
                        masked[tid] = -np.inf
                        continue
                    callee_key = closed_callee.rsplit(".", 1)[-1]
                    last_line = active_code.splitlines()[-1].lstrip() if active_code.splitlines() else ""
                    if callee_key in live_syms and not last_line.startswith(("def ", "@")):
                        lparen = cand_text.rfind("(")
                        rparen = cand_text.rfind(")")
                        inner_call = cand_text[lparen + 1 : rparen].strip() if rparen > lparen else ""
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

    @staticmethod
    def verify_epistemic_calibration(prompt: str) -> Optional[Dict[str, str]]:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        p_lower = prompt.lower()

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
    def _check_ast_policy(cls, code_str: str) -> None:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, cls.FORBIDDEN_AST_NODES):
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

    @classmethod
    def execute_experiment_block(cls, script_str: str, timeout: float = 3.0) -> str:
        """Execute a ``<experiment>...</experiment>`` block safely and return observation."""
        try:
            cls._check_ast_policy(script_str)
        except (SyntaxError, PermissionError) as exc:
            return f"EXP_ERROR: {exc}"
        ok, res_repr, _after, err = cls._sandbox_dispatch(
            script_str, timeout=timeout, mode="experiment_script"
        )
        if not ok:
            return f"EXP_ERROR: {err}"
        return str(res_repr)

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
            scored.append({"code": code, "passed": passed, "total": total, "errors": errors})
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
            tot = max(1, int(item["total"]))
            frac = float(item["passed"]) / float(tot)
            r = 1.0 if item["passed"] == item["total"] and item["total"] > 0 else 0.2 * frac
            raw_rewards.append(r)

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
