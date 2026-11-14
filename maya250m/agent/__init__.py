"""Bounded CodeAct agent: deterministic tools, state machine, firewall, rewards."""

from __future__ import annotations

from .firewall import (
    TOOL_DATA_END,
    TOOL_DATA_START,
    FirewallViolation,
    MessageChannel,
    MessageKind,
    ToolOutputFirewall,
    TypedMessage,
    neutralize_markers,
    separate_messages,
)
from .runtime import (
    LOOP_GUARD_TEMPLATE,
    AgentRuntime,
    CompletionEvidence,
    ConstraintEvidence,
    ToolObservation,
)
from .sandbox import (
    PathGuard,
    Sandbox,
    SandboxLimits,
    SandboxPolicy,
    SandboxResult,
    SandboxViolation,
)
from .tools import TOOL_SCHEMAS, RepoTools, ToolCallError
from .trajectories import (
    AgentTrajectory,
    ProcessReward,
    RESOLVED_THRESHOLD,
    TrajectoryStep,
    build_sft_example,
    observation_mask_spans,
    score_trajectory,
)

__all__ = [
    "TOOL_DATA_END",
    "TOOL_DATA_START",
    "ToolOutputFirewall",
    "neutralize_markers",
    "FirewallViolation",
    "MessageChannel",
    "MessageKind",
    "TypedMessage",
    "separate_messages",
    "PathGuard",
    "Sandbox",
    "SandboxLimits",
    "SandboxPolicy",
    "SandboxResult",
    "SandboxViolation",
    "LOOP_GUARD_TEMPLATE",
    "AgentRuntime",
    "CompletionEvidence",
    "ConstraintEvidence",
    "ToolObservation",
    "TOOL_SCHEMAS",
    "RepoTools",
    "ToolCallError",
    "AgentTrajectory",
    "ProcessReward",
    "RESOLVED_THRESHOLD",
    "TrajectoryStep",
    "build_sft_example",
    "observation_mask_spans",
    "score_trajectory",
]
