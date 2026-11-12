"""Bounded CodeAct agent: deterministic tools, state machine, firewall, rewards."""

from __future__ import annotations

from .firewall import (
    TOOL_DATA_END,
    TOOL_DATA_START,
    ToolOutputFirewall,
    neutralize_markers,
)
from .runtime import LOOP_GUARD_TEMPLATE, AgentRuntime, ToolObservation
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
    "LOOP_GUARD_TEMPLATE",
    "AgentRuntime",
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
