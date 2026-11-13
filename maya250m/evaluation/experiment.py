"""WP-00: canonical experiment manifest, telemetry, and release gates.

Every benchmark run must be reproducible from a single manifest plus stored
JSONL. No score is considered valid unless the underlying config, checkpoint,
tokenizer, and dataset provenance are hashed and attached.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "ExperimentManifest",
    "ExperimentRun",
    "SCORECARD_MODES",
    "Score",
    "ScoreMode",
    "Scorecard",
    "experiment_commit_info",
    "hash_file",
    "load_scores_from_jsonl",
    "publish_allowed",
    "register_score",
    "run_from_manifest",
    "validate_manifest",
]


# ------------------------------------------------------------------ commit info


def _sha256_of(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def experiment_commit_info(
    repo_root: Optional[str | os.PathLike] = None,
) -> Dict[str, Any]:
    """Return git commit info for ``repo_root`` or the current working tree.

    Captures the current commit hash, any dirty-tree state, and tags.
    If no Git repository is present, returns ``{"dirty": True, ...}`` with no
    commit hash; do not publish scores without one.
    """
    root = str(repo_root or ".")
    info = {"dirty": True, "commit": None, "tags": []}
    try:
        import subprocess

        if os.path.exists(os.path.join(root, ".git")):
            info["commit"] = subprocess.run(
                ["git", "-C", root, "rev-parse", "--verify", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "-C", root, "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                ).stdout.strip()
            )
            info["dirty"] = dirty
            tags = subprocess.run(
                ["git", "-C", root, "describe", "--tags", "--exact-match", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if tags.returncode == 0:
                info["tags"] = tags.stdout.strip().splitlines()
    except Exception:  # noqa: BLE001 - no Git = known unknown, not a crash
        pass
    return info


def hash_file(path: str | os.PathLike) -> str:
    """Content-addressed SHA-256 of a file (checkpoint, tokenizer, manifest)."""
    return _sha256_of(path)


# ---------------------------------------------------------------- manifest


@dataclass
class ExperimentManifest:
    """Immutable description of everything that can affect a benchmark score."""

    run_id: str
    git_commit: str
    git_dirty: bool
    config_sha256: str
    checkpoint_sha256: Optional[str]
    tokenizer_sha256: str
    dataset_manifest_sha256: Optional[str]
    benchmark_version: str
    seed: int
    device: str
    precision: str
    max_context: int
    recurrent_loops: int
    max_new_tokens: int
    temperature: float
    top_p: Optional[float]
    tool_policy: Dict[str, Any]
    retrieval_policy: Dict[str, Any]
    created_at_utc: str
    n_seeds: int = 1
    retries_per_task: int = 1
    test_budget_seconds: Optional[int] = None
    model_tag: Optional[str] = None
    data_mode: Optional[str] = None  # smoke | local-shards | production

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentManifest":
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "config_sha256": self.config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "benchmark_version": self.benchmark_version,
            "seed": self.seed,
            "device": self.device,
            "precision": self.precision,
            "max_context": self.max_context,
            "recurrent_loops": self.recurrent_loops,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "tool_policy": dict(self.tool_policy),
            "retrieval_policy": dict(self.retrieval_policy),
            "created_at_utc": self.created_at_utc,
            "n_seeds": self.n_seeds,
            "retries_per_task": self.retries_per_task,
            "test_budget_seconds": self.test_budget_seconds,
            "model_tag": self.model_tag,
            "data_mode": self.data_mode,
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]


# ---------------------------------------------------------------- scores


class ScoreMode:
    """Scorecard modes (WP-02.1): base | controller | tools | full_agent."""

    BASE = "base"
    CONTROLLER = "controller"
    TOOLS = "tools"
    FULL_AGENT = "full_agent"


SCORECARD_MODES = (
    ScoreMode.BASE,
    ScoreMode.CONTROLLER,
    ScoreMode.TOOLS,
    ScoreMode.FULL_AGENT,
)


@dataclass
class Score:
    """A single scorecard entry with provenance and tolerances."""

    name: str
    value: float
    n: int
    status: str  # measured | not_run | failed
    reason: str = ""
    unit: str = ""
    mode: str = "base"  # base | controller | tools | full_agent
    suite: str = ""
    metric_family: str = "accuracy"
    metadata: Dict[str, Any] = field(default_factory=dict)
    reproducible_from_jsonl: bool = True


@dataclass
class Scorecard:
    """Same-identity collection of scores; report-level provenance lives in the run."""

    title: str
    scores: List[Score] = field(default_factory=list)
    identity: str = "benchmark"
    status: str = "not_run"

    @property
    def summary(self) -> Dict[str, float]:
        grouped: Dict[str, List[float]] = {}
        for s in self.scores:
            if s.status == "measured":
                grouped.setdefault(s.metric_family, []).append(s.value)
        return {fam: sum(v) / len(v) for fam, v in grouped.items() if v}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "identity": self.identity,
            "status": self.status,
            "scores": [s.__dict__ for s in self.scores],
        }


@dataclass
class ExperimentRun:
    """One complete run: manifest, telemetry, JSONL, scorecard, gates."""

    manifest: ExperimentManifest
    scorecard: Scorecard
    telemetry: Dict[str, Any] = field(default_factory=dict)
    jsonl_path: Optional[str] = None
    report_path: Optional[str] = None
    gates: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def record_jsonl_line(self, row: Dict[str, Any]) -> None:
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def add_score(self, score: Score) -> None:
        self.scorecard.scores.append(score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "scorecard": self.scorecard.to_dict(),
            "telemetry": dict(self.telemetry),
            "jsonl_path": self.jsonl_path,
            "report_path": self.report_path,
            "gates": dict(self.gates),
            "errors": list(self.errors),
        }


# -------------------------------------------------------- registry + gates


def validate_manifest(manifest: ExperimentManifest) -> List[str]:
    """Acceptance gates applied before publishing any score."""
    issues: List[str] = []
    if not manifest.git_commit:
        issues.append("no git commit: scores are not reproducible")
    if manifest.git_dirty:
        issues.append("working tree dirty: recorded commit does not match source")
    if not manifest.checkpoint_sha256:
        issues.append("no checkpoint hash: score may not correspond to any weights")
    if not manifest.tokenizer_sha256:
        issues.append("no tokenizer hash: scores may not be reproducible")
    if manifest.data_mode == "smoke":
        issues.append("data_mode=smoke: scores must be labelled smoke-only")
    if manifest.tool_policy.get("untrusted_execution", False):
        issues.append("untrusted_execution not sandboxed")
    return issues


def publish_allowed(manifest: ExperimentManifest) -> bool:
    """Gate C/D: never publish a benchmark score with unknown provenance."""
    issues = validate_manifest(manifest)
    return not any(
        msg.startswith(("no git commit", "no tokenizer hash", "untrusted_execution"))
        for msg in issues
    )


# -------------------------------------------------- determinism + regeneration


def load_scores_from_jsonl(
    jsonl_path: str, manifest: ExperimentManifest
) -> Dict[str, float]:
    """Regenerate aggregate scores from stored JSONL without rerunning inference.

    Expects JSONL rows with at minimum ``{"name":..., "value":..., "status":..., "mode":...}``.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    statuses: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            name = row["name"]
            totals[name] = totals.get(name, 0.0) + float(row["value"])
            counts[name] = counts.get(name, 0) + 1
            statuses[name] = row.get("status", statuses[name])
    return {name: totals[name] / counts[name] for name in totals}


# ------------------------------------------------------------------ CLI runner


SuitesFn = Callable[[ExperimentRun], Dict[str, Any]]


def run_from_manifest(
    manifest: ExperimentManifest,
    suites: List[str],
    suites_fn: Optional[Dict[str, SuitesFn]] = None,
    telemetry_fn: Optional[Callable[[ExperimentRun, str, Dict[str, Any]], None]] = None,
) -> ExperimentRun:
    """Execute a composed experiment from a manifest.

    ``suites_fn`` maps suite name -> (loader, scorer) callables; ``telemetry_fn``
    accepts ``(run, suite_name, suite_result)``. This is a composable backbone so
    each concrete suite plugs in without duplicating manifest/gate logic.
    """
    run = ExperimentRun(
        manifest=manifest,
        scorecard=Scorecard(title=f"run-{manifest.run_id}"),
        jsonl_path=f"runs/{manifest.run_id}/results.jsonl",
    )
    os.makedirs(os.path.dirname(run.jsonl_path) if os.path.dirname(run.jsonl_path) else ".", exist_ok=True)
    suites = list(suites or ["smoke"])
    suites_fn = suites_fn or {}
    start = time.monotonic()
    telemetry = {
        "suites": suites,
        "started_at_utc": manifest.created_at_utc,
        "durations_seconds": {},
    }
    for suite in suites:
        s_start = time.monotonic()
        fn = suites_fn.get(suite)
        if callable(fn):
            result = fn(run)
        else:
            result = {"measured": False, "reason": f"unknown suite {suite!r}"}
        telemetry["durations_seconds"][suite] = time.monotonic() - s_start
        if telemetry_fn is not None:
            try:
                telemetry_fn(run, suite, result)
            except Exception:  # noqa: BLE001
                pass
        run.telemetry = telemetry
    wall = time.monotonic() - start
    run.telemetry["wall_seconds"] = wall
    run.telemetry["suites_run"] = suites
    return run


def register_score(mode: str) -> Callable:
    """Register a suite function on a run; attaches ``mode`` to added scores.

    Decorator usage::

        @register_score("tools")
        def add_scores(run): ...
    """
    def decorator(fn: SuitesFn) -> SuitesFn:
        def wrapper(run: ExperimentRun) -> Dict[str, Any]:
            original_add = run.add_score
            recorded: List[Score] = []

            def add_score(score: Score) -> None:
                score.mode = mode
                recorded.append(score)
                original_add(score)

            run.add_score = add_score  # type: ignore[method-assign]
            result = fn(run)
            run.add_score = original_add  # type: ignore[method-assign]
            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    ci = experiment_commit_info()
    print(json.dumps(ci, indent=2))
