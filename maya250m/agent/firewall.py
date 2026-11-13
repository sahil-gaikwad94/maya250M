"""Prompt-injection firewall for untrusted tool output.

Repository files, test output and issue text are *data*. A malicious or merely
unlucky file could contain text that looks like Maya's own control protocol
(FIM markers, ``# <|edit_file|>``, ``SEARCH``/``REPLACE`` fences) or a fake system
message. If that text were pasted back into the model's own context unfenced, the
generation controller could be hijacked.

This module therefore:

1. neutralises every control marker (replacing ``<``/``>`` with inert look-alikes);
2. fences the result in ``[TOOL_DATA_START] ... [TOOL_DATA_END]`` so the trainer can
   mask observation tokens and the model can learn "everything here is evidence";
3. truncates to a hard character budget so one file cannot flood the window;
4. reports whether injection-shaped content was present, for audit logs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

__all__ = [
    "ToolOutputFirewall",
    "TOOL_DATA_START",
    "TOOL_DATA_END",
    "neutralize_markers",
    "MessageKind",
    "TypedMessage",
    "MessageChannel",
    "FirewallViolation",
    "separate_messages",
]

TOOL_DATA_START = "[TOOL_DATA_START]"
TOOL_DATA_END = "[TOOL_DATA_END]"

# Reasoning tags are assembled from parts so this source file can never be
# mistaken for a reasoning block by a downstream text filter.
_LT = "<"
_GT = ">"
_THINK_OPEN = _LT + "think" + _GT
_THINK_CLOSE = _LT + "/think" + _GT

#: Markers that must never survive into model context unescaped.
_SPOOF_MARKERS = (
    _THINK_OPEN,
    _THINK_CLOSE,
    _LT + "experiment" + _GT,
    _LT + "/experiment" + _GT,
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|unk|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|repo_context|>",
    "<|file_sep|>",
    "<|edit_file|>",
    "<<<<<<< SEARCH",
    ">>>>>>> REPLACE",
    TOOL_DATA_START,
    TOOL_DATA_END,
)

_FAKE_ROLE_RE = re.compile(
    r"^[ \t]*(?:SYSTEM|ASSISTANT|USER|DEVELOPER|TOOL)[ \t]*:",
    re.MULTILINE | re.IGNORECASE,
)

_NEUTRALIZED = {
    marker: marker.replace(_LT, "\u2039").replace(_GT, "\u203a")
    for marker in _SPOOF_MARKERS
}


def neutralize_markers(text: str) -> Tuple[str, List[str]]:
    """Replace spoofable control markers; return ``(clean_text, found_markers)``."""
    found: List[str] = []
    cleaned = text
    for marker in _SPOOF_MARKERS:
        if marker in cleaned:
            found.append(marker)
            cleaned = cleaned.replace(marker, _NEUTRALIZED[marker])
    return cleaned, found


@dataclass
class ToolOutputFirewall:
    """Sanitize and fence untrusted tool output."""

    max_chars: int = 4000
    max_line_length: int = 400
    redact_fake_roles: bool = True

    def sanitize(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Neutralize markers and fake role prefixes in one tool payload."""
        original = "" if text is None else str(text)
        cleaned, found = neutralize_markers(original)
        fake_roles = bool(_FAKE_ROLE_RE.search(original))
        if self.redact_fake_roles:
            cleaned = _FAKE_ROLE_RE.sub(
                lambda m: m.group(0).replace(":", "\u02d0", 1), cleaned
            )

        clipped_lines = []
        clipped = 0
        for line in cleaned.splitlines():
            if len(line) > self.max_line_length:
                clipped_lines.append(line[: self.max_line_length] + " ...")
                clipped += 1
            else:
                clipped_lines.append(line)
        cleaned = "\n".join(clipped_lines)

        truncated = False
        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars].rstrip() + "\n... [output truncated]"
            truncated = True

        audit = {
            "spoofed_markers": found,
            "fake_role_prefixes": fake_roles,
            "clipped_lines": clipped,
            "truncated": truncated,
            "original_chars": len(original),
            "sanitized_chars": len(cleaned),
            "suspected_injection": bool(found) or fake_roles,
        }
        return cleaned, audit

    def wrap(self, kind: str, text: str) -> Tuple[str, Dict[str, Any]]:
        """Sanitize and fence a tool observation as inert data."""
        cleaned, audit = self.sanitize(text)
        fenced = f"{TOOL_DATA_START} kind={kind}\n{cleaned}\n{TOOL_DATA_END}"
        return fenced, audit

    def contains_injection(self, text: str) -> bool:
        _cleaned, found = neutralize_markers(str(text or ""))
        return bool(found) or bool(_FAKE_ROLE_RE.search(str(text or "")))

    # -------------------------------------------------- typed message separation

    def typed(self, kind: str, text: str, *, provenance: str = "") -> "TypedMessage":
        """Sanitize ``text`` into a typed, fenced tool-data message."""
        return TypedMessage.tool_data(kind, text, provenance=provenance, firewall=self)


class MessageKind(str, Enum):
    """Who is speaking. Untrusted content may only ever be ``TOOL_DATA``."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_DATA = "tool_data"
    TOOL_CALL = "tool_call"


_ROLE_FOR_KIND = {
    MessageKind.SYSTEM: "system",
    MessageKind.USER: "user",
    MessageKind.ASSISTANT: "assistant",
    MessageKind.TOOL_DATA: "tool",
    MessageKind.TOOL_CALL: "assistant",
}
_UNTRUSTED_ALLOWED = {MessageKind.TOOL_DATA}


class FirewallViolation(PermissionError):
    """Untrusted content tried to claim a privileged role."""


def _fence(kind: str, cleaned: str) -> str:
    return f"{TOOL_DATA_START} kind={kind or 'tool'}\n{cleaned}\n{TOOL_DATA_END}"


@dataclass(frozen=True)
class TypedMessage:
    """One message with an explicit, non-spoofable kind.

    ``kind`` is metadata, never parsed from content, so text inside a repository
    file that says ``SYSTEM: ignore previous instructions`` can only ever be a
    tool observation.
    """

    kind: MessageKind
    content: str
    provenance: str = "model"
    trusted: bool = True
    truncated: bool = False
    injection_suspected: bool = False

    @classmethod
    def tool_data(
        cls,
        kind: str,
        text: str,
        *,
        provenance: str = "tool",
        firewall: "ToolOutputFirewall" = None,
    ) -> "TypedMessage":
        firewall = firewall or ToolOutputFirewall()
        cleaned, audit = firewall.sanitize(text)
        return cls(
            kind=MessageKind.TOOL_DATA,
            content=cleaned,
            provenance=provenance or kind or "tool",
            trusted=False,
            truncated=bool(audit.get("truncated")),
            injection_suspected=bool(audit.get("suspected_injection")),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MessageKind):
            object.__setattr__(self, "kind", MessageKind(self.kind))
        if not self.trusted and self.kind not in _UNTRUSTED_ALLOWED:
            raise FirewallViolation(
                f"untrusted content cannot claim the {self.kind.value!r} role"
            )

    def render(self) -> str:
        if self.kind is MessageKind.TOOL_DATA:
            cleaned, _found = neutralize_markers(self.content)
            return _fence(self.provenance, cleaned)
        if self.kind is MessageKind.SYSTEM:
            return f"[SYSTEM]\n{self.content}"
        if self.kind is MessageKind.USER:
            return f"[USER]\n{self.content}"
        if self.kind is MessageKind.TOOL_CALL:
            return f"[CONTROL {self.provenance}]\n{self.content}"
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": _ROLE_FOR_KIND[self.kind],
            "content": self.render(),
            "kind": self.kind.value,
            "provenance": self.provenance,
            "trusted": bool(self.trusted),
            "truncated": bool(self.truncated),
            "injection_suspected": bool(self.injection_suspected),
        }


@dataclass
class MessageChannel:
    """Ordered conversation with one-way trust: data can never become control."""

    firewall: ToolOutputFirewall = field(default_factory=ToolOutputFirewall)
    messages: List[TypedMessage] = field(default_factory=list)
    audit: List[Dict[str, Any]] = field(default_factory=list)

    def system(self, text: str) -> TypedMessage:
        return self._add(TypedMessage(MessageKind.SYSTEM, str(text), provenance="system"))

    def user(self, text: str) -> TypedMessage:
        return self._add(TypedMessage(MessageKind.USER, str(text), provenance="user"))

    def assistant(self, text: str) -> TypedMessage:
        return self._add(
            TypedMessage(MessageKind.ASSISTANT, str(text), provenance="assistant")
        )

    def control(self, category: str, text: str) -> TypedMessage:
        """A verification/tool-call record produced by Maya itself (trusted)."""
        return self._add(
            TypedMessage(MessageKind.TOOL_CALL, str(text), provenance=category or "tool_call")
        )

    def tool_output(
        self, kind: str, text: str, *, provenance: str = ""
    ) -> TypedMessage:
        cleaned, audit = self.firewall.sanitize(text)
        self.audit.append(audit)
        return self._add(
            TypedMessage(
                kind=MessageKind.TOOL_DATA,
                content=cleaned,
                provenance=provenance or kind or "tool",
                trusted=False,
                truncated=bool(audit.get("truncated")),
                injection_suspected=bool(audit.get("suspected_injection")),
            )
        )

    def _add(self, message: TypedMessage) -> TypedMessage:
        self.messages.append(message)
        return message

    def render(self) -> List[Dict[str, str]]:
        return [message.to_dict() for message in self.messages]

    def render_text(self) -> str:
        return "\n".join(message.render() for message in self.messages)

    def suspicious_count(self) -> int:
        return sum(1 for message in self.messages if message.injection_suspected)

    def __len__(self) -> int:
        return len(self.messages)


def separate_messages(
    raw: Iterable[Mapping[str, Any]], firewall: "ToolOutputFirewall" = None
) -> List[TypedMessage]:
    """Normalise untrusted chat-shaped payloads into typed messages.

    A payload that claims ``role: system`` but carries tool/repo provenance is
    demoted to ``TOOL_DATA``: content is never allowed to select its own role.
    """
    firewall = firewall or ToolOutputFirewall()
    out: List[TypedMessage] = []
    for entry in raw:
        role = str(entry.get("role") or entry.get("kind") or "tool").lower()
        content = str(entry.get("content") or "")
        provenance = str(entry.get("provenance") or entry.get("source") or role)
        untrusted = bool(entry.get("untrusted")) or provenance in (
            "tool",
            "repo",
            "file",
            "test",
            "issue",
            "web",
        )
        if role in ("system", "user", "assistant") and not untrusted:
            out.append(
                TypedMessage(
                    kind=MessageKind(role),
                    content=content,
                    provenance=provenance,
                    trusted=True,
                )
            )
        else:
            cleaned, audit = firewall.sanitize(content)
            out.append(
                TypedMessage(
                    kind=MessageKind.TOOL_DATA,
                    content=cleaned,
                    provenance=provenance,
                    trusted=False,
                    truncated=bool(audit.get("truncated")),
                    injection_suspected=bool(audit.get("suspected_injection")),
                )
            )
    return out
