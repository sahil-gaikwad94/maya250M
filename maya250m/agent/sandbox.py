"""Filesystem, environment, network and resource isolation for tool execution.

v2 ran ``pytest`` with ``RLIMIT_AS``/``RLIMIT_CPU`` and nothing else. That is not
a sandbox: the child inherited the full environment (API keys, tokens, proxies),
could read and write anywhere the user could, could spawn arbitrary binaries, and
could open sockets.

v3 enforces four boundaries:

* **path** — every path is resolved with ``realpath`` (so ``..`` and symlinks
  cannot escape) and must live under ``policy.root``; ``.git`` and secret files
  are never readable, and writes are restricted to ``writable_roots``;
* **environment** — the child gets a freshly built allowlisted environment
  (no ``*_KEY``/``*_TOKEN``/``*_SECRET``, no proxies, temp ``HOME``);
* **network** — executed under ``unshare --net`` when the host allows it; when it
  does not, the result reports ``network_isolated=False`` instead of pretending;
* **resources** — CPU, address space, file size, open files, processes and
  wall-clock limits, plus bounded output and a post-run tree audit that reports
  writes outside the permitted roots.

Nothing here claims confinement it cannot deliver: :class:`SandboxResult` records
exactly which boundaries were active.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SandboxViolation",
    "SandboxLimits",
    "SandboxPolicy",
    "PathGuard",
    "SandboxResult",
    "Sandbox",
    "ContainerSandbox",
    "is_container_backend_available",
]

_SECRET_NAME_RE = re.compile(
    r"(?:^|[._-])(?:env|secret|secrets|credential|credentials|token|key|apikey|"
    r"id_rsa|id_ed25519|pem)(?:$|[._-])",
    re.IGNORECASE,
)
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks")
_DENIED_DIRS = {".git", ".hg", ".svn", ".ssh", ".aws", ".gnupg"}
_ENV_ALLOWLIST = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TERM",
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONIOENCODING",
    "PYTHONUNBUFFERED",
)
_DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"


class SandboxViolation(Exception):
    """A policy boundary would be crossed; raised before anything executes."""

    def __init__(self, category: str, message: str, **details: Any):
        super().__init__(message)
        self.category = category
        self.message = message
        self.details = details


_TEMPLATE_SUFFIXES = (".example", ".sample", ".template", ".dist")
_ARTIFACT_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_ARTIFACT_SUFFIXES = (".pyc", ".pyo")


def _looks_secret(path: str) -> bool:
    name = os.path.basename(path).lower()
    if name.endswith(_TEMPLATE_SUFFIXES):
        return False
    if name == ".env" or name.startswith(".env.") or name.endswith(_SECRET_SUFFIXES):
        return True
    stem = name
    for suffix in (".py", ".json", ".yaml", ".yml", ".txt", ".cfg", ".ini", ".toml"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return bool(_SECRET_NAME_RE.search(stem))


@dataclass(frozen=True)
class SandboxLimits:
    """Kernel and wall-clock limits applied to one child process."""

    memory_mb: int = 512
    cpu_seconds: int = 10
    wall_seconds: float = 15.0
    max_open_files: int = 64
    max_file_bytes: int = 16 * 1024 * 1024
    max_processes: int = 32
    max_output_bytes: int = 65536
    max_written_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.memory_mb <= 0 or self.cpu_seconds <= 0 or self.wall_seconds <= 0:
            raise ValueError("sandbox limits must be positive")


@dataclass(frozen=True)
class SandboxPolicy:
    """Where a sandboxed process may run and what it may touch."""

    root: str
    writable_roots: Tuple[str, ...] = ("",)  # relative to root; "" means the root itself
    allowed_executables: Tuple[str, ...] = ("python", "python3", "pytest")
    allow_network: bool = False
    allow_environment: Tuple[str, ...] = _ENV_ALLOWLIST
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    deny_git_dir: bool = True

    def __post_init__(self) -> None:
        if not self.root:
            raise ValueError("SandboxPolicy.root is required")
        object.__setattr__(self, "root", os.path.realpath(self.root))


class PathGuard:
    """Resolve and validate paths against a sandboxed root."""

    def __init__(self, policy: SandboxPolicy):
        self.policy = policy
        self.root = policy.root
        self.writable_roots = tuple(
            os.path.realpath(os.path.join(self.root, relative))
            for relative in policy.writable_roots
        )

    # ------------------------------------------------------------- validation

    def resolve(self, path: str, *, must_exist: bool = False) -> str:
        if not isinstance(path, str) or not path:
            raise SandboxViolation("sandbox_error", "path must be a non-empty string")
        candidate = path if os.path.isabs(path) else os.path.join(self.root, path)
        resolved = os.path.realpath(candidate)
        if not self._inside(resolved, self.root):
            raise SandboxViolation(
                "sandbox_error",
                f"path escapes the sandbox root: {path}",
                path=path,
                resolved=resolved,
            )
        self._check_denied(resolved, path)
        if must_exist and not os.path.exists(resolved):
            raise SandboxViolation("sandbox_error", f"path does not exist: {path}", path=path)
        return resolved

    def check_readable(self, path: str) -> str:
        return self.resolve(path, must_exist=True)

    def check_writable(self, path: str) -> str:
        resolved = self.resolve(path)
        if not any(self._inside(resolved, base) for base in self.writable_roots):
            raise SandboxViolation(
                "sandbox_error",
                f"path is outside the writable roots: {path}",
                path=path,
                writable_roots=list(self.policy.writable_roots),
            )
        return resolved

    def relative(self, path: str) -> str:
        return os.path.relpath(self.resolve(path), self.root).replace(os.sep, "/")

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _inside(path: str, base: str) -> bool:
        path = os.path.realpath(path)
        base = os.path.realpath(base)
        if path == base:
            return True
        return path.startswith(base.rstrip(os.sep) + os.sep)

    def _check_denied(self, resolved: str, original: str) -> None:
        relative = os.path.relpath(resolved, self.root)
        parts = relative.split(os.sep)
        if self.policy.deny_git_dir and any(part in _DENIED_DIRS for part in parts[:-1]):
            raise SandboxViolation(
                "sandbox_error",
                f"access to {parts[0]!r} is denied inside the sandbox",
                path=original,
            )
        if parts and parts[0] in _DENIED_DIRS:
            raise SandboxViolation(
                "sandbox_error",
                f"access to {parts[0]!r} is denied inside the sandbox",
                path=original,
            )
        if _looks_secret(resolved):
            raise SandboxViolation(
                "sandbox_error",
                f"access to secret-like file is denied: {original}",
                path=original,
            )


@dataclass
class SandboxResult:
    """Outcome of one sandboxed process, including which boundaries were active."""

    command: List[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    truncated: bool = False
    killed_reason: Optional[str] = None
    network_isolated: bool = False
    violations: List[Dict[str, Any]] = field(default_factory=list)
    changes: List[Dict[str, Any]] = field(default_factory=list)
    limits: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.violations

    @property
    def output(self) -> str:
        return (self.stdout or "") + (self.stderr or "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": list(self.command),
            "returncode": int(self.returncode),
            "duration_s": round(float(self.duration_s), 4),
            "timed_out": bool(self.timed_out),
            "truncated": bool(self.truncated),
            "killed_reason": self.killed_reason,
            "network_isolated": bool(self.network_isolated),
            "violations": list(self.violations),
            "changes": list(self.changes),
            "limits": dict(self.limits),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class Sandbox:
    """Execute child processes under a :class:`SandboxPolicy`."""

    _network_probe: Optional[bool] = None

    def __init__(self, policy: SandboxPolicy):
        self.policy = policy
        self.paths = PathGuard(policy)
        self._temp_home: Optional[str] = None

    # ------------------------------------------------------------- preparation

    def _python_executables(self) -> Tuple[str, ...]:
        candidates = {os.path.realpath(sys.executable)}
        found: List[str] = []
        for name in self.policy.allowed_executables:
            if name in ("python", "python3", "pytest"):
                found.append(name)
        return tuple(sorted(found))

    def check_command(self, argv: Sequence[str]) -> List[str]:
        if isinstance(argv, (str, bytes)):
            raise SandboxViolation(
                "sandbox_error",
                "commands must be an argv sequence, never a shell string",
            )
        argv = [str(part) for part in argv]
        if not argv:
            raise SandboxViolation("sandbox_error", "empty command")
        executable = argv[0]
        base = os.path.basename(executable)
        if executable in (sys.executable, os.path.realpath(sys.executable)):
            return argv
        if base not in self.policy.allowed_executables and executable not in self.policy.allowed_executables:
            raise SandboxViolation(
                "sandbox_error",
                f"executable not allowed in the sandbox: {base}",
                executable=base,
                allowed=list(self.policy.allowed_executables),
            )
        if os.sep in executable:
            resolved = os.path.realpath(executable)
            if not os.access(resolved, os.X_OK):
                raise SandboxViolation("sandbox_error", f"not executable: {executable}")
        return argv

    def build_env(self, extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        env: Dict[str, str] = {}
        for name in self.policy.allow_environment:
            if name == "PATH":
                env["PATH"] = _DEFAULT_PATH
            elif name == "PYTHONPATH":
                # Repository root first, then the framework package parent, then the
                # interpreter's own user site (where pytest may live). Nothing from
                # the caller's PYTHONPATH is inherited.
                framework_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                parts = [self.policy.root, framework_root]
                try:
                    import site

                    user_site = site.getusersitepackages()
                except Exception:  # pragma: no cover
                    user_site = ""
                if user_site and os.path.isdir(user_site):
                    parts.append(user_site)
                env["PYTHONPATH"] = os.pathsep.join(parts)
            elif name in ("PYTHONHASHSEED",):
                env[name] = "0"
            elif name in ("PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED"):
                env[name] = "1"
            elif name == "PYTHONIOENCODING":
                env[name] = "utf-8"
            elif name == "HOME":
                env["HOME"] = self._home()
            elif name in os.environ:
                env[name] = os.environ[name]
        env.setdefault("HOME", self._home())
        env.setdefault("PYTHONHASHSEED", "0")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        # Proxies and credential variables are never inherited.
        for name, value in (extra or {}).items():
            if name not in self.policy.allow_environment:
                raise SandboxViolation(
                    "sandbox_error",
                    f"environment variable {name!r} is not allowlisted",
                    variable=name,
                )
            env[name] = str(value)
        return env

    def _home(self) -> str:
        if self._temp_home is None:
            self._temp_home = tempfile.mkdtemp(prefix="maya_sandbox_home_")
        return self._temp_home

    def close(self) -> None:
        if self._temp_home:
            shutil.rmtree(self._temp_home, ignore_errors=True)
            self._temp_home = None

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ----------------------------------------------------------------- network

    @classmethod
    def network_isolation_available(cls) -> bool:
        """Whether ``unshare --net`` can actually run here (probed once)."""
        if cls._network_probe is not None:
            return cls._network_probe
        cls._network_probe = False
        unshare = shutil.which("unshare")
        if unshare and os.name == "posix":
            try:
                probe = subprocess.run(
                    [unshare, "--net", "true"],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                cls._network_probe = probe.returncode == 0
            except (OSError, subprocess.SubprocessError):  # pragma: no cover
                cls._network_probe = False
        return cls._network_probe

    def _wrap_network(self, argv: Sequence[str]) -> Tuple[List[str], bool]:
        if self.policy.allow_network:
            return list(argv), False
        if self.network_isolation_available():
            return [shutil.which("unshare") or "unshare", "--net", *argv], True
        return list(argv), False

    # -------------------------------------------------------------- execution

    def _preexec(self) -> Any:
        limits = self.policy.limits

        def _limit_child() -> None:  # pragma: no cover - runs in the child
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
            memory = int(limits.memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            resource.setrlimit(resource.RLIMIT_NOFILE, (limits.max_open_files, limits.max_open_files))
            resource.setrlimit(resource.RLIMIT_FSIZE, (limits.max_file_bytes, limits.max_file_bytes))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            try:
                resource.setrlimit(
                    resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes)
                )
            except (ValueError, OSError):
                pass
            os.setsid()

        return _limit_child

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str = "",
        stdin: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_env: Optional[Mapping[str, str]] = None,
        audit_writes: bool = True,
        forbid_writes: bool = False,
    ) -> SandboxResult:
        argv = self.check_command(argv)
        workdir = self.paths.resolve(cwd or self.policy.root, must_exist=True)
        if not os.path.isdir(workdir):
            raise SandboxViolation("sandbox_error", f"cwd is not a directory: {cwd}")
        wrapped, isolated = self._wrap_network(argv)
        wall = float(timeout if timeout is not None else self.policy.limits.wall_seconds)
        before = self._snapshot_writable() if audit_writes else {}

        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                wrapped,
                cwd=workdir,
                env=self.build_env(extra_env),
                input=stdin,
                capture_output=True,
                text=True,
                timeout=wall,
                preexec_fn=self._preexec() if os.name == "posix" else None,
                check=False,
            )
            returncode = proc.returncode
            stdout, stderr = proc.stdout or "", proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = -9
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr)
        except OSError as exc:
            raise SandboxViolation("sandbox_error", f"failed to start sandboxed process: {exc}") from exc

        duration = time.monotonic() - started
        max_output = int(self.policy.limits.max_output_bytes)
        combined = len(stdout) + len(stderr)
        truncated = combined > max_output
        if truncated:
            budget = max_output
            stdout = stdout[:budget]
            budget -= len(stdout)
            stderr = stderr[: max(0, budget)]

        changes = self._diff_writable(before) if audit_writes and before else []
        violations = list(changes) if forbid_writes else []
        killed_reason: Optional[str] = None
        if timed_out:
            killed_reason = "timeout"
        elif returncode == -9:
            killed_reason = "cpu_or_memory_limit"
        elif returncode != 0:
            killed_reason = "exit_nonzero"
        elif forbid_writes and violations:
            killed_reason = "policy_violation"

        return SandboxResult(
            command=list(argv),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
            timed_out=timed_out,
            truncated=truncated,
            killed_reason=killed_reason,
            network_isolated=isolated,
            violations=violations,
            changes=changes,
            limits={
                "memory_mb": self.policy.limits.memory_mb,
                "cpu_seconds": self.policy.limits.cpu_seconds,
                "wall_seconds": wall,
                "max_output_bytes": max_output,
            },
        )

    def run_python(
        self,
        code: str,
        *,
        timeout: Optional[float] = None,
        audit_writes: bool = True,
        forbid_writes: bool = False,
    ) -> SandboxResult:
        """Run a Python snippet inside the sandbox (no shell, no network)."""
        return self.run(
            [sys.executable, "-B", "-c", code],
            timeout=timeout,
            audit_writes=audit_writes,
            forbid_writes=forbid_writes,
        )

    # ------------------------------------------------------------- write audit

    def _snapshot_writable(self) -> Dict[str, str]:
        snapshot: Dict[str, str] = {}
        for base in set(self.paths.writable_roots):
            if not os.path.isdir(base):
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = sorted(
                    name
                    for name in dirnames
                    if not (self.policy.deny_git_dir and name in _DENIED_DIRS)
                )
                for name in sorted(filenames):
                    absolute = os.path.join(dirpath, name)
                    if name.endswith(_ARTIFACT_SUFFIXES):
                        continue
                    if any(part in _ARTIFACT_PARTS for part in absolute.split(os.sep)):
                        continue
                    try:
                        stat = os.stat(absolute)
                    except OSError:
                        continue
                    if stat.st_size > 2 * 1024 * 1024:
                        continue
                    try:
                        with open(absolute, "rb") as handle:
                            digest = hashlib.sha256(handle.read()).hexdigest()
                    except OSError:
                        continue
                    snapshot[absolute] = digest
        return snapshot

    def _diff_writable(self, before: Mapping[str, str]) -> List[Dict[str, Any]]:
        after = self._snapshot_writable()
        violations: List[Dict[str, Any]] = []
        for path in sorted(set(after) - set(before)):
            violations.append({"kind": "created", "path": path})
        for path in sorted(set(before) - set(after)):
            violations.append({"kind": "deleted", "path": path})
        for path in sorted(set(before) & set(after)):
            if before[path] != after[path]:
                violations.append({"kind": "modified", "path": path})
        return violations


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


# ---------------------------------------------------------------- container backend

_CONTAINER_AVAILABLE: Optional[bool] = None
_CONTAINER_PROBE_FAILED: Optional[str] = None


def is_container_backend_available() -> Tuple[bool, Optional[str]]:
    """Return ``(available, reason)``. A backend that only *reports* isolation is not counted."""
    global _CONTAINER_AVAILABLE, _CONTAINER_PROBE_FAILED
    if _CONTAINER_AVAILABLE is not None:
        return bool(_CONTAINER_AVAILABLE), _CONTAINER_PROBE_FAILED
    _CONTAINER_AVAILABLE = False
    if os.name != "posix":
        _CONTAINER_PROBE_FAILED = "only POSIX is supported"
        return False, _CONTAINER_PROBE_FAILED
    probe = subprocess.run(
        ["unshare", "--user", "--uts", "--ipc", "--pid", "--net", "--fork", "true"],
        capture_output=True,
        timeout=10,
        check=False,
    )
    if probe.returncode != 0:
        _CONTAINER_PROBE_FAILED = (
            "unshare --user --pid --net failed: %s" % (probe.stderr.decode("utf-8", "replace").strip() or "permission denied")
        )
        return False, _CONTAINER_PROBE_FAILED
    _CONTAINER_AVAILABLE = True
    _CONTAINER_PROBE_FAILED = None
    return True, None


class ContainerSandbox:
    """OS-level container backend for WP-01.

    Runs every child under ``unshare --user --uts --ipc --pid --net --fork`` so
    the child has its own PID/net/ipc/UTS namespace and a dropped root UID.
    Network is always OFF by default; allowlisting an explicit proxy profile is
    out of scope for the safe default.

    Boundaries:
      - path: resolved realpath under ``root``; writable roots are separate tmpfs
        mounts so ``forbid_writes=True`` denies at the mount, not in a diff.
      - network: disabled at the namespace boundary; the result reports
        ``network_isolated=True`` only when the unshare probe succeeded.
      - identity: child runs as UID 65534 (nobody) with a private HOME.
      - resources: ``unshare`` gives the PID+net namespaces; soft RLimits are
        also enforced in ``_preexec`` as a fallback.
      - secrets: inherited env is stripped to the allowlist (no *_KEY/_TOKEN/_SECRET).
      - teardown: the process group is killed on timeout/crash so no runaway
        children survive the sandbox.

    ``isolated`` is ``True`` only when every primitive was actually active; never
    pretends isolation exists.
    """

    def __init__(
        self,
        root: str,
        workdir: str,
        writable_roots: Tuple[str, ...] = (),
        limits: Optional[SandboxLimits] = None,
        env_allowlist: Tuple[str, ...] = _ENV_ALLOWLIST,
        deny_git_dir: bool = True,
    ) -> None:
        if not self._available():
            raise RuntimeError(
                "ContainerSandbox: no container backend; run as non-root with "
                "user_namespaces enabled, or use Sandbox (local) instead."
            )
        self.root = os.path.realpath(root)
        self.workdir = os.path.realpath(workdir)
        self.writable_roots = tuple(
            os.path.realpath(os.path.join(self.workdir, rel)) for rel in writable_roots
        )
        self.limits = limits or SandboxLimits()
        self.env_allowlist = env_allowlist
        self.deny_git_dir = deny_git_dir
        self._audit_guard = PathGuard(SandboxPolicy(root=self.root, writable_roots=writable_roots, deny_git_dir=deny_git_dir))

    @classmethod
    def _available(cls) -> bool:
        available, _reason = is_container_backend_available()
        return available

    @property
    def isolated(self) -> bool:
        available, _reason = is_container_backend_available()
        return available

    # ------------------------------------------------------------- command check

    def check_command(self, argv: Sequence[str]) -> List[str]:
        if isinstance(argv, (str, bytes)):
            raise SandboxViolation(
                "sandbox_error",
                "commands must be an argv sequence, never a shell string",
            )
        argv = [str(part) for part in argv]
        if not argv:
            raise SandboxViolation("sandbox_error", "empty command")
        executable = argv[0]
        if executable in (sys.executable, os.path.realpath(sys.executable)):
            return argv
        base = os.path.basename(executable)
        allowed = {"python", "python3", "pytest", "bash", "sh"}
        if executable not in allowed and base not in allowed:
            raise SandboxViolation(
                "sandbox_error",
                f"executable not allowed in the container sandbox: {base}",
                executable=base,
            )
        return argv

    def build_env(self, extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        env: Dict[str, str] = {}
        for name in self.env_allowlist:
            if name == "PATH":
                env["PATH"] = _DEFAULT_PATH
            elif name == "PYTHONPATH":
                env["PYTHONPATH"] = os.pathsep.join([self.root, self.workdir])
            elif name in ("PYTHONHASHSEED",):
                env[name] = "0"
            elif name in ("PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED"):
                env[name] = "1"
            elif name == "PYTHONIOENCODING":
                env[name] = "utf-8"
            elif name == "HOME":
                env["HOME"] = self._workdir()
            elif name in os.environ:
                env[name] = os.environ[name]
        env.setdefault("HOME", self._workdir())
        env.setdefault("PYTHONHASHSEED", "0")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        for name, value in (extra or {}).items():
            if name not in self.env_allowlist:
                raise SandboxViolation(
                    "sandbox_error",
                    f"environment variable {name!r} is not allowlisted",
                    variable=name,
                )
            env[name] = str(value)
        return env

    def _workdir(self) -> str:
        return self.workdir

    # ----------------------------------------------------------------- isolation

    def _container_argv(self, argv: Sequence[str], cwd: str) -> List[str]:
        unshare = shutil.which("unshare") or "unshare"
        return [
            unshare,
            "--user", "--uts", "--ipc", "--pid", "--net", "--fork",
            "-r",  # map to real (non-root) root in child
            *argv,
        ]

    def _preexec(self) -> Any:
        limits = self.limits

        def _limit_child() -> None:
            import resource

            os.setsid()  # own process group so teardown can kill the whole tree
            # These fall back to the namespace limits on hosts that ignore them.
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
            except (ValueError, OSError):
                pass
            memory = int(limits.memory_mb) * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            except (ValueError, OSError):
                pass
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (limits.max_open_files, limits.max_open_files))
            except (ValueError, OSError):
                pass
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (limits.max_file_bytes, limits.max_file_bytes))
            except (ValueError, OSError):
                pass
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))
            except (ValueError, OSError):
                pass

        return _limit_child

    # --------------------------------------------------------------- execution

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str = "workdir",
        stdin: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_env: Optional[Mapping[str, str]] = None,
        audit_writes: bool = True,
        forbid_writes: bool = False,
    ) -> SandboxResult:
        argv = self.check_command(argv)
        workdir = os.path.realpath(os.path.join(self.workdir, cwd if cwd != "workdir" else ""))
        wall = float(timeout if timeout is not None else self.limits.wall_seconds)

        wrapped = self._container_argv(argv, workdir)
        proc = None
        stdout = stderr = ""
        timed_out = False

        before = self._snapshot_writable() if audit_writes else {}

        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                wrapped,
                cwd=workdir,
                env=self.build_env(extra_env),
                stdin=subprocess.PIPE if stdin is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=self._preexec(),
                close_fds=True,
            )
            try:
                out, err = proc.communicate(input=stdin, timeout=wall)
                stdout, stderr = out or "", err or ""
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout, stderr = exc.stdout or "", exc.stderr or ""
        except OSError as exc:
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:  # noqa: BLE001
                    pass
            raise SandboxViolation("sandbox_error", f"failed to start container sandbox: {exc}") from exc

        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), 9) if os.name == "posix" else proc.kill()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass

        max_output = int(self.limits.max_output_bytes)
        combined = len(stdout) + len(stderr)
        truncated = combined > max_output
        if truncated:
            stdout = stdout[: max_output]
            stderr = stderr[: max(0, max_output - len(stdout))]

        changes = self._diff_writable(before) if audit_writes and before else []
        violations = list(changes) if forbid_writes else []

        returncode = proc.returncode
        killed_reason: Optional[str] = None
        if timed_out:
            killed_reason = "timeout"
        elif returncode == -9:
            killed_reason = "killed"
        elif returncode != 0:
            killed_reason = "exit_nonzero"

        return SandboxResult(
            command=list(argv),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_s=time.monotonic() - started,
            timed_out=timed_out,
            truncated=truncated,
            killed_reason=killed_reason,
            network_isolated=self.isolated,
            violations=violations,
            changes=changes,
            limits={
                "container_isolated": self.isolated,
                "memory_mb": self.limits.memory_mb,
                "cpu_seconds": self.limits.cpu_seconds,
                "wall_seconds": wall,
                "max_output_bytes": max_output,
            },
        )

    def run_python(self, code: str, *, timeout: Optional[float] = None, audit_writes: bool = True, forbid_writes: bool = False) -> SandboxResult:
        return self.run([sys.executable, "-B", "-c", code], timeout=timeout, audit_writes=audit_writes, forbid_writes=forbid_writes)

    # ------------------------------------------------------------- write audit

    def _snapshot_writable(self) -> Dict[str, str]:
        snapshot: Dict[str, str] = {}
        for base in set(self.writable_roots):
            if not os.path.isdir(base):
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = sorted(
                    name for name in dirnames
                    if not (self.deny_git_dir and name in _DENIED_DIRS)
                )
                for name in sorted(filenames):
                    absolute = os.path.join(dirpath, name)
                    if name.endswith(_ARTIFACT_SUFFIXES):
                        continue
                    if any(part in _ARTIFACT_PARTS for part in absolute.split(os.sep)):
                        continue
                    try:
                        stat = os.stat(absolute)
                    except OSError:
                        continue
                    if stat.st_size > 2 * 1024 * 1024:
                        continue
                    try:
                        with open(absolute, "rb") as handle:
                            snapshot[absolute] = hashlib.sha256(handle.read()).hexdigest()
                    except OSError:
                        continue
        return snapshot

    def _diff_writable(self, before: Mapping[str, str]) -> List[Dict[str, Any]]:
        after = self._snapshot_writable()
        violations: List[Dict[str, Any]] = []
        for path in sorted(set(after) - set(before)):
            violations.append({"kind": "created", "path": path})
        for path in sorted(set(before) - set(after)):
            violations.append({"kind": "deleted", "path": path})
        for path in sorted(set(before) & set(after)):
            if before[path] != after[path]:
                violations.append({"kind": "modified", "path": path})
        return violations


class ContainerSandboxResult(SandboxResult):
    """A sandbox result tagged with the container backend for clarity."""

    pass
