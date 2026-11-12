"""Five evaluation identities, kept structurally separate.

The audit's failure mode was a single "score" that silently mixed a training loss
with a benchmark pass rate. Here a :class:`Measurement` must declare exactly one
identity, and an un-run measurement cannot carry a number at all:

* ``TRAIN_LOSS``           — optimisation progress on the training split
* ``HELD_OUT_VALIDATION``  — perplexity / CE on the held-out split
* ``BENCHMARK``            — external benchmark pass rates
* ``AGENT_TASK``           — repo-level task success with tools
* ``ABLATION``             — the A–P ablation matrix

Mixing them raises :class:`MeasurementError`; reporting a ``NOT_RUN`` result as a
score is impossible by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

__all__ = [
    "MeasurementIdentity",
    "MeasurementStatus",
    "Measurement",
    "EvalReport",
    "MeasurementError",
]


class MeasurementError(ValueError):
    """Raised when measurements are mixed or misreported."""


class MeasurementIdentity(str, Enum):
    TRAIN_LOSS = "train_loss"
    HELD_OUT_VALIDATION = "held_out_validation"
    BENCHMARK = "benchmark"
    AGENT_TASK = "agent_task"
    ABLATION = "ablation"


class MeasurementStatus(str, Enum):
    MEASURED = "MEASURED"
    NOT_RUN = "NOT_RUN"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


@dataclass
class Measurement:
    """One number with exactly one identity, or an explicit non-result."""

    identity: MeasurementIdentity
    name: str
    value: Optional[float] = None
    status: MeasurementStatus = MeasurementStatus.MEASURED
    n: int = 0
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is MeasurementStatus.MEASURED:
            if self.value is None:
                raise MeasurementError(
                    f"measured quantity {self.name!r} needs a value; use NOT_RUN instead"
                )
            self.value = float(self.value)
            if self.n < 0:
                raise MeasurementError("n must be non-negative")
        else:
            if self.value is not None:
                raise MeasurementError(
                    f"{self.name!r} is {self.status.value} but carries value {self.value}; "
                    "an un-run experiment has no score"
                )

    @property
    def measured(self) -> bool:
        return self.status is MeasurementStatus.MEASURED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.value,
            "name": self.name,
            "value": self.value,
            "status": self.status.value,
            "n": int(self.n),
            "unit": self.unit,
            "metadata": dict(self.metadata),
        }


@dataclass
class EvalReport:
    """A collection of same-identity measurements plus their provenance."""

    title: str = "evaluation"
    measurements: List[Measurement] = field(default_factory=list)

    def add(self, measurement: Measurement) -> Measurement:
        if self.measurements and self.measurements[0].identity is not measurement.identity:
            raise MeasurementError(
                f"report {self.title!r} holds {self.measurements[0].identity.value} "
                f"measurements; refusing to mix in {measurement.identity.value}"
            )
        self.measurements.append(measurement)
        return measurement

    @property
    def identity(self) -> Optional[MeasurementIdentity]:
        return self.measurements[0].identity if self.measurements else None

    @property
    def status(self) -> MeasurementStatus:
        if not self.measurements:
            return MeasurementStatus.NOT_RUN
        if all(m.status is MeasurementStatus.NOT_RUN for m in self.measurements):
            return MeasurementStatus.NOT_RUN
        if any(m.status is MeasurementStatus.BLOCKED for m in self.measurements):
            return MeasurementStatus.BLOCKED
        if all(m.status is MeasurementStatus.FAILED for m in self.measurements):
            return MeasurementStatus.FAILED
        return MeasurementStatus.MEASURED

    def get(self, name: str) -> Optional[Measurement]:
        for measurement in self.measurements:
            if measurement.name == name:
                return measurement
        return None

    def require_measured(self, name: str) -> float:
        measurement = self.get(name)
        if measurement is None:
            raise MeasurementError(f"no measurement named {name!r}")
        if not measurement.measured:
            raise MeasurementError(
                f"{name!r} is {measurement.status.value}: {measurement.metadata.get('reason', '')}"
            )
        return float(measurement.value)

    def compare_with(self, other: "EvalReport") -> None:
        """Guard against comparing a perplexity report with a pass-rate report."""
        if self.identity is None or other.identity is None:
            raise MeasurementError("cannot compare empty reports")
        if self.identity is not other.identity:
            raise MeasurementError(
                f"refusing to compare {self.identity.value} with {other.identity.value}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "identity": self.identity.value if self.identity else None,
            "status": self.status.value,
            "measurements": [m.to_dict() for m in self.measurements],
        }
