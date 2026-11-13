"""Provenance-checked data mixtures and pilot experiment accounting (v3 §8.4–8.5).

The three candidate mixtures are *pilot hypotheses*, not magic numbers. This module
makes that explicit: it never declares a winner, it never silently pads a domain
that ran out of data, and every sampled batch carries a manifest with corpus hashes
so a mixture experiment is reproducible and comparable at the same token budget.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .dedup import DOMAINS, DataRecord

__all__ = [
    "MIXTURES",
    "MIXTURE_DISCLAIMER",
    "MixtureSpec",
    "MixtureReport",
    "DataMixer",
]

MIXTURE_DISCLAIMER = (
    "Pilot hypotheses only. Choose a mixture from a predeclared multi-domain "
    "held-out score at matched tokens/compute, never from training loss alone."
)

MIXTURES: Dict[str, Dict[str, float]] = {
    "balanced": {
        "general": 0.40,
        "code": 0.20,
        "math": 0.12,
        "science": 0.12,
        "research": 0.0,
        "engineering": 0.16,
    },
    "code_forward": {
        "general": 0.30,
        "code": 0.32,
        "math": 0.12,
        "science": 0.10,
        "research": 0.0,
        "engineering": 0.16,
    },
    "reasoning_forward": {
        "general": 0.30,
        "code": 0.20,
        "math": 0.18,
        "science": 0.14,
        "research": 0.0,
        "engineering": 0.18,
    },
}


@dataclass(frozen=True)
class MixtureSpec:
    """One named mixture, with declared weights that must sum to 1."""

    name: str
    weights: Dict[str, float]

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(DOMAINS)
        if unknown:
            raise ValueError(f"mixture {self.name!r} has unknown domains: {sorted(unknown)}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"mixture {self.name!r} weights sum to {total}, expected 1.0")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "weights": dict(self.weights), "disclaimer": MIXTURE_DISCLAIMER}


def get_mixture(name: str) -> MixtureSpec:
    if name not in MIXTURES:
        raise KeyError(f"unknown mixture {name!r}; valid: {sorted(MIXTURES)}")
    return MixtureSpec(name=name, weights=dict(MIXTURES[name]))


@dataclass
class MixtureReport:
    """What a mixture actually produced, including shortfalls."""

    name: str
    requested: int
    weights: Dict[str, float]
    counts: Dict[str, int] = field(default_factory=dict)
    target_counts: Dict[str, int] = field(default_factory=dict)
    shortfall: Dict[str, int] = field(default_factory=dict)
    realized: Dict[str, float] = field(default_factory=dict)
    tokens: int = 0
    records: List[DataRecord] = field(default_factory=list)
    source_counts: Dict[str, int] = field(default_factory=dict)
    seed: int = 0

    @property
    def filled(self) -> int:
        return len(self.records)

    def manifest(self) -> Dict[str, Any]:
        digest = hashlib.sha256()
        for record in sorted(self.records, key=lambda item: item.record_id):
            digest.update(f"{record.record_id}\x00{record.content_sha256}\n".encode("utf-8"))
        return {
            "mixture": self.name,
            "seed": self.seed,
            "requested": self.requested,
            "filled": self.filled,
            "weights": dict(self.weights),
            "target_counts": dict(self.target_counts),
            "counts": dict(self.counts),
            "shortfall": dict(self.shortfall),
            "realized": dict(self.realized),
            "tokens": self.tokens,
            "dataset_sha256": digest.hexdigest(),
            "disclaimer": MIXTURE_DISCLAIMER,
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self.manifest()
        payload["records"] = [record.to_dict() for record in self.records]
        return payload


class DataMixer:
    """Deterministic, cap-aware, provenance-validated domain sampling."""

    def __init__(
        self,
        records: Sequence[DataRecord],
        *,
        seed: int = 0,
        max_per_source: int = 4,
        require_provenance: bool = True,
    ):
        if max_per_source <= 0:
            raise ValueError("max_per_source must be positive")
        self.seed = int(seed)
        self.max_per_source = int(max_per_source)
        self.records = list(records)
        if require_provenance:
            for record in self.records:
                record.provenance.validate()

    def _by_domain(self) -> Dict[str, List[DataRecord]]:
        buckets: Dict[str, List[DataRecord]] = {domain: [] for domain in DOMAINS}
        for record in self.records:
            buckets.setdefault(record.domain, []).append(record)
        return buckets

    def mix(self, name: str, n_samples: int, *, tokens_per_sample: Optional[int] = None) -> MixtureReport:
        """Sample ``n_samples`` records according to the named mixture."""
        if int(n_samples) <= 0:
            raise ValueError("n_samples must be positive")
        spec = get_mixture(name)
        rng = random.Random(self.seed)
        buckets = self._by_domain()

        targets: Dict[str, int] = {}
        for domain, weight in spec.weights.items():
            targets[domain] = int(round(weight * int(n_samples)))

        chosen: List[DataRecord] = []
        counts: Dict[str, int] = {}
        shortfall: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}

        for domain in DOMAINS:
            target = targets.get(domain, 0)
            if target <= 0:
                counts[domain] = 0
                continue
            candidates = sorted(
                buckets.get(domain, []),
                key=lambda record: (-float(record.provenance.quality_score), record.record_id),
            )
            picked: List[DataRecord] = []
            for record in candidates:
                source = record.provenance.source_id
                if source_counts.get(source, 0) >= self.max_per_source:
                    continue
                picked.append(record)
                source_counts[source] = source_counts.get(source, 0) + 1
                if len(picked) >= target:
                    break
            rng.shuffle(picked)  # deterministic given the seed; breaks quality ordering
            counts[domain] = len(picked)
            shortfall[domain] = max(0, target - len(picked))
            chosen.extend(picked)

        chosen.sort(key=lambda record: record.record_id)
        tokens = sum(
            int(tokens_per_sample)
            if tokens_per_sample is not None
            else max(1, len(record.text.split()))
            for record in chosen
        )
        realized = {
            domain: (counts.get(domain, 0) / len(chosen)) if chosen else 0.0
            for domain in DOMAINS
        }
        return MixtureReport(
            name=name,
            requested=int(n_samples),
            weights=dict(spec.weights),
            counts=counts,
            target_counts=targets,
            shortfall={domain: value for domain, value in shortfall.items() if value},
            realized=realized,
            tokens=tokens,
            records=chosen,
            source_counts=source_counts,
            seed=self.seed,
        )

    def compare(
        self, n_samples: int, names: Optional[Iterable[str]] = None
    ) -> Dict[str, Any]:
        """Pilot comparison skeleton: same seed, same budget, no winner declared."""
        names = list(names or MIXTURES)
        reports = [self.mix(name, n_samples) for name in names]
        return {
            "budget_records": int(n_samples),
            "seed": self.seed,
            "mixtures": {report.name: report.manifest() for report in reports},
            "winner": None,
            "decision_rule": (
                "Choose from a predeclared multi-domain held-out score at matched "
                "tokens; this harness records the runs and never picks a winner."
            ),
        }
