"""The seven deterministic, resource-bounded agent tools.

Every tool is fast, local and returns a structured result. None of them calls a
network service, an external LLM, or an unbounded subprocess.

======================  ==========================================================
tool                    bound
======================  ==========================================================
``get_repo_map``        PPR-ranked skeleton map, ``max_tokens`` default 1500
``search_repo``         regex over ``*.py``, ``max_hits`` default 20
``view_file_span``      at most 160 numbered lines per call
``inspect_symbol_graph`` one symbol: definition, contract, callers, callees
``apply_patch``         AST + cross-file verified SEARCH/REPLACE, atomic write
``run_tests``           isolated subprocess, 512 MiB, wall-clock timeout
``git_revert_file``     restore the task-start snapshot of one file
======================  ==========================================================
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..codebase.codegen import PatchError, SearchReplacePatchEngine, parse_patch_blocks
from ..codebase.indexer import RepoGraphIndexer
from ..codebase.multifile import build_cross_file_symbol_table
from ..codebase.repomap import RepoMapBuilder
from ..verification import VerificationResult, failure, ok
from .sandbox import Sandbox, SandboxLimits, SandboxPolicy, SandboxViolation

__all__ = ["RepoTools", "TOOL_SCHEMAS", "ToolCallError"]

MAX_SPAN_LINES = 160
MAX_SEARCH_FILE_BYTES = 400_000

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "get_repo_map": {
        "description": "Return a PPR-ranked repository skeleton map for a query.",
        "parameters": {
            "query": {"type": "string", "required": True},
            "max_tokens": {"type": "integer", "default": 1500},
        },
    },
    "search_repo": {
        "description": "Regex search across repository files; returns file/line/snippet hits.",
        "parameters": {
            "pattern": {"type": "string", "required": True},
            "file_glob": {"type": "string", "default": "*.py"},
            "max_hits": {"type": "integer", "default": 20},
        },
    },
    "view_file_span": {
        "description": "Read numbered lines from one file (<= 160 lines).",
        "parameters": {
            "file_path": {"type": "string", "required": True},
            "start_line": {"type": "integer", "required": True},
            "end_line": {"type": "integer", "required": True},
        },
    },
    "inspect_symbol_graph": {
        "description": "Definition, contract, callers and callees for one symbol.",
        "parameters": {"symbol_name": {"type": "string", "required": True}},
    },
    "apply_patch": {
        "description": "Apply an AST-verified SEARCH/REPLACE patch to one file.",
        "parameters": {
            "file_path": {"type": "string", "required": True},
            "search_block": {"type": "string", "required": True},
            "replace_block": {"type": "string", "required": True},
        },
    },
    "run_tests": {
        "description": "Run pytest in a sandboxed subprocess and summarize failures.",
        "parameters": {
            "test_target": {"type": "string", "default": "tests/"},
            "timeout_s": {"type": "number", "default": 10.0},
        },
    },
    "apply_patch_set": {
        "description": "Apply a multi-file SEARCH/REPLACE set atomically (all or nothing).",
        "parameters": {"patch_text": {"type": "string", "required": True}},
    },
    "git_revert_file": {
        "description": "Restore one file to its task-start snapshot.",
        "parameters": {"file_path": {"type": "string", "required": True}},
    },
}


class ToolCallError(ValueError):
    """Raised when a tool is called with invalid arguments."""


def _limit_child_factory(memory_limit_mb: int = 512, cpu_seconds: int = 30):
    """Build a ``preexec_fn`` that applies CPU/address-space/file limits."""

    def _limit_child() -> None:  # pragma: no cover - POSIX only
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        limit = int(memory_limit_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return _limit_child


@dataclass
class RepoTools:
    """Deterministic repository and verification tools bound to one repository."""

    indexer: RepoGraphIndexer
    root: Optional[str] = None
    vexx: Any = None
    pytest_target: str = "tests/"
    snapshot_dir: Optional[str] = None
    memory_limit_mb: int = 512
    sandbox: Optional[Sandbox] = None
    allow_test_edits: bool = False
    _snapshot: Dict[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = os.path.abspath(self.root or self.indexer.root)
        self.engine = SearchReplacePatchEngine(
            indexer=self.indexer,
            vexx=self.vexx,
            root=self.root,
            allow_test_edits=self.allow_test_edits,
        )
        self.repomap = RepoMapBuilder(self.indexer)
        if self.sandbox is None:
            self.sandbox = Sandbox(
                SandboxPolicy(
                    root=self.root,
                    limits=SandboxLimits(memory_mb=self.memory_limit_mb),
                )
            )
        if not self._snapshot:
            self.take_snapshot()

    # ------------------------------------------------------------------ snapshot

    def take_snapshot(self, files: Optional[Iterable[str]] = None) -> int:
        """Record the current bytes of every indexed file for later revert."""
        targets = list(files) if files is not None else sorted(self.indexer.files)
        self.snapshot_dir = self.snapshot_dir or tempfile.mkdtemp(prefix="maya_task_snapshot_")
        count = 0
        for rel in targets:
            full = os.path.join(self.root, rel)
            try:
                with open(full, "rb") as handle:
                    raw = handle.read()
            except OSError:
                continue
            self._snapshot[rel] = raw
            dest = os.path.join(self.snapshot_dir, rel)
            os.makedirs(os.path.dirname(dest) or self.snapshot_dir, exist_ok=True)
            with open(dest, "wb") as handle:
                handle.write(raw)
            count += 1
        return count

    # --------------------------------------------------------------------- tools

    def get_repo_map(self, query: str, max_tokens: int = 1500) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ToolCallError("get_repo_map requires a non-empty query")
        return self.repomap.build(query, token_budget=int(max_tokens)).text

    def search_repo(
        self, pattern: str, file_glob: str = "*.py", max_hits: int = 20
    ) -> List[Dict[str, Any]]:
        if not pattern:
            raise ToolCallError("search_repo requires a pattern")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolCallError(f"invalid regular expression: {exc}") from exc

        hits: List[Dict[str, Any]] = []
        for rel in sorted(self.indexer.files):
            if not fnmatch.fnmatch(os.path.basename(rel), file_glob) and not fnmatch.fnmatch(
                rel, file_glob
            ):
                continue
            full = os.path.join(self.root, rel)
            try:
                if os.path.getsize(full) > MAX_SEARCH_FILE_BYTES:
                    continue
                with open(full, "r", encoding="utf-8", errors="replace") as handle:
                    for line_no, line in enumerate(handle, 1):
                        if regex.search(line):
                            hits.append(
                                {
                                    "file": rel,
                                    "line": line_no,
                                    "snippet": line.strip()[:200],
                                }
                            )
                            if len(hits) >= int(max_hits):
                                return hits
            except OSError:
                continue
        return hits

    def view_file_span(self, file_path: str, start_line: int, end_line: int) -> str:
        rel = file_path.replace("\\", "/")
        if rel not in self.indexer.files:
            raise ToolCallError(f"file not indexed: {file_path}")
        start = max(1, int(start_line))
        end = max(start, int(end_line))
        if end - start + 1 > MAX_SPAN_LINES:
            end = start + MAX_SPAN_LINES - 1
        full = os.path.join(self.root, rel)
        with open(full, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        end = min(end, len(lines))
        rendered = [f"{i:>6}| {lines[i - 1].rstrip()}" for i in range(start, end + 1)]
        return "\n".join(rendered)

    def inspect_symbol_graph(self, symbol_name: str) -> Dict[str, Any]:
        if not symbol_name:
            raise ToolCallError("inspect_symbol_graph requires a symbol name")
        basename = symbol_name.rsplit(".", 1)[-1]
        definitions = self.indexer.symbols_named(basename)
        callers = self.indexer.callers(basename)
        result: Dict[str, Any] = {
            "symbol": symbol_name,
            "found": bool(definitions),
            "definitions": [
                {
                    "qualname": d.qualname,
                    "file": d.file_path,
                    "kind": d.kind,
                    "start_line": d.start_line,
                    "end_line": d.end_line,
                    "signature": d.signature,
                    "contract": {
                        "min_arity": d.min_arity,
                        "max_arity": d.max_arity,
                        "return_type": d.return_type,
                    },
                }
                for d in definitions
            ],
            "callers": [
                {
                    "file": c.file_path,
                    "line": c.line,
                    "n_positional": c.n_positional,
                    "caller": c.caller_qualname,
                }
                for c in callers
            ],
            "callees": self.indexer.callees(definitions[0].qualname) if definitions else [],
        }
        if definitions:
            result["cross_file_contracts"] = build_cross_file_symbol_table(
                self.indexer, definitions[0].file_path
            )
        return result

    def apply_patch(
        self, file_path: str, search_block: str, replace_block: str
    ) -> VerificationResult:
        return self.engine.apply_and_verify(
            file_path, search_block, replace_block, commit=True
        )

    def apply_patch_set(self, patch_text: str) -> VerificationResult:
        """Apply a multi-file patch as one atomic, test-tamper-protected transaction."""
        return self.engine.apply_patch_text_transactional(patch_text, commit=True)

    def run_tests(
        self, test_target: str = "", timeout_s: float = 10.0, memory_limit_mb: Optional[int] = None
    ) -> VerificationResult:
        if self.sandbox is not None:
            return self._run_tests_sandboxed(
                test_target or self.pytest_target, timeout_s, memory_limit_mb
            )
        target = test_target or self.pytest_target
        env = dict(os.environ)
        env["PYTHONPATH"] = self.root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONHASHSEED"] = "0"

        commands = [
            [sys.executable, "-B", "-m", "pytest", "-q", target, "--no-header"],
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", target],
        ]
        limiter = _limit_child_factory(
            self.memory_limit_mb if memory_limit_mb is None else int(memory_limit_mb),
            max(2, int(timeout_s) + 5),
        )
        last_output = ""
        for command in commands:
            try:
                proc = subprocess.run(
                    command,
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=float(timeout_s),
                    env=env,
                    preexec_fn=limiter if os.name == "posix" else None,
                )
            except subprocess.TimeoutExpired:
                return failure(
                    "test_timeout",
                    f"test run exceeded {timeout_s:.1f}s",
                    command=" ".join(command[:4]),
                )
            except OSError as exc:
                last_output = str(exc)
                continue

            output = (proc.stdout or "") + (proc.stderr or "")
            last_output = output
            if proc.returncode == 0:
                return ok(
                    "tests passed",
                    command=" ".join(command[:4]),
                    output_tail=output.strip().splitlines()[-5:],
                )
            if "No module named pytest" in output:
                continue  # fall back to unittest
            break

        summary = self._summarize_test_output(last_output)
        return failure(
            "tests_failed",
            summary["headline"],
            output_tail=summary["tail"],
            failing_tests=summary["failing"],
        )

    def _run_tests_sandboxed(
        self,
        target: str,
        timeout_s: float,
        memory_limit_mb: Optional[int],
    ) -> VerificationResult:
        """Run pytest inside the sandbox: scrubbed env, limits, write audit."""
        command = [sys.executable, "-B", "-m", "pytest", "-q", target, "--no-header"]
        if memory_limit_mb is not None:
            self.sandbox.policy = SandboxPolicy(
                root=self.root,
                writable_roots=self.sandbox.policy.writable_roots,
                limits=SandboxLimits(
                    memory_mb=int(memory_limit_mb),
                    cpu_seconds=max(2, int(timeout_s) + 5),
                    wall_seconds=max(5.0, float(timeout_s) + 5.0),
                ),
            )
        try:
            run = self.sandbox.run(
                command, cwd=self.root, timeout=float(timeout_s) + 5.0
            )
        except SandboxViolation as exc:
            return failure("sandbox_error", exc.message, **exc.details)

        if "No module named pytest" in run.output:
            command = [
                sys.executable,
                "-B",
                "-m",
                "unittest",
                "discover",
                "-s",
                target,
            ]
            try:
                run = self.sandbox.run(
                    command, cwd=self.root, timeout=float(timeout_s) + 5.0
                )
            except SandboxViolation as exc:
                return failure("sandbox_error", exc.message, **exc.details)

        details = {
            "command": " ".join(command[:4]),
            "sandbox": run.to_dict(),
        }
        if run.timed_out:
            return failure(
                "test_timeout", f"test run exceeded {timeout_s:.1f}s", **details
            )
        if run.returncode == 0:
            return ok(
                "tests passed",
                output_tail=run.output.strip().splitlines()[-5:],
                **details,
            )
        summary = self._summarize_test_output(run.output)
        return failure(
            "tests_failed",
            summary["headline"],
            output_tail=summary["tail"],
            failing_tests=summary["failing"],
            **details,
        )

    @staticmethod
    def _summarize_test_output(output: str) -> Dict[str, Any]:
        lines = [line.rstrip() for line in (output or "").splitlines() if line.strip()]
        failing = [line.strip() for line in lines if line.startswith("FAILED") or line.startswith("ERROR")]
        headline = failing[0] if failing else (lines[-1] if lines else "tests failed")
        return {"headline": headline[:400], "tail": lines[-8:], "failing": failing[:10]}

    def git_revert_file(self, file_path: str) -> str:
        rel = file_path.replace("\\", "/")
        if rel not in self._snapshot:
            return f"no snapshot recorded for {rel}"
        full = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(full) or self.root, exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(self._snapshot[rel])
        try:
            self.indexer.index_file(rel)
        except Exception:  # pragma: no cover - best effort refresh
            pass
        return f"reverted {rel} to its task-start snapshot"

    # ------------------------------------------------------------------ dispatch

    def call(self, name: str, **kwargs: Any) -> Any:
        """Validated JSON-style dispatch used by the runtime."""
        if name not in TOOL_SCHEMAS:
            raise ToolCallError(f"unknown tool {name!r}; valid: {sorted(TOOL_SCHEMAS)}")
        schema = TOOL_SCHEMAS[name]["parameters"]
        unknown = set(kwargs) - set(schema)
        if unknown:
            raise ToolCallError(
                f"{name} received unknown parameter(s) {sorted(unknown)}; "
                f"valid: {sorted(schema)}"
            )
        for param, spec in schema.items():
            if spec.get("required") and param not in kwargs:
                raise ToolCallError(f"{name} missing required parameter {param!r}")
        method = getattr(self, name)
        return method(**kwargs)

    def cleanup(self) -> None:
        if self.snapshot_dir and os.path.isdir(self.snapshot_dir):
            shutil.rmtree(self.snapshot_dir, ignore_errors=True)
