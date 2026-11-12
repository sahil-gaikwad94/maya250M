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
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

__all__ = ["ToolOutputFirewall", "TOOL_DATA_START", "TOOL_DATA_END", "neutralize_markers"]

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
            "clipped_lines": clipped,
            "truncated": truncated,
            "original_chars": len(original),
            "sanitized_chars": len(cleaned),
            "suspected_injection": bool(found),
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
