"""Autoregressive & MTP-Speculative generation with Vexx 3.5 in the decode loop.

Supports:
1. Deterministic greedy argmax (``temperature is None`` or ``<= 0``) and top-p sampling.
2. Vexx 3.5 Tier 0/1 in-flight logit masking across ``plain``, ``python``, ``json``, and ``think``.
3. Inline Tier-2 CAS splicing (``<<expr=val>>``) inside ``<think>...</think>``.
4. Inline Tier-5 ``<experiment>...</experiment>`` sandbox execution splicing
   (``=> [OBS: ...]``) for ground-up scientific hypothesis verification.
5. Optional MTP-3 self-speculative drafting verified by Vexx 3.5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from .model import MayaModel
from .tokenizer import BOS, EOS, EXP_END, EXP_START, BPETokenizer

PLAIN = "plain"
PYTHON = "python"
JSON = "json"
THINK = "think"
VALID_MODES = {PLAIN, PYTHON, JSON, THINK}


@dataclass
class GenerationResult:
    prompt: str
    continuation: str
    full_text: str
    finish_reason: str
    tokens_generated: int
    truncated: bool
    provenance: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.full_text


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    squeeze = logits.dim() == 1
    work = logits.unsqueeze(0) if squeeze else logits
    sorted_logits, sorted_indices = torch.sort(work, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = work.clone()
    filtered.scatter_(1, sorted_indices, torch.where(remove, float("-inf"), sorted_logits))
    return filtered.squeeze(0) if squeeze else filtered


def sample_next(
    logits: torch.Tensor,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled = logits / temperature
    if top_p is not None and top_p < 1.0:
        scaled = top_p_filter(scaled, top_p)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def generate_maya(
    model: MayaModel,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    vexx=None,
    add_bos: bool = True,
    seed: Optional[int] = None,
    mode: str = PLAIN,
    provenance: Optional[List[str]] = None,
    context_mode: str = "rolling_window",
) -> GenerationResult:
    """Generate a continuation with Vexx 3.5 neuro-symbolic verification."""
    if mode not in VALID_MODES:
        raise ValueError(f"unknown decoding mode {mode!r}; valid: {sorted(VALID_MODES)}")
    if context_mode not in {"rolling_window", "strict"}:
        raise ValueError(f"unknown context_mode {context_mode!r}")

    model.eval()
    origins: List[str] = ["neural_generation"]

    ids = tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
    if len(ids) == 0:
        ids = [tokenizer.token_to_id[BOS]]
    truncated = False
    if len(ids) > model.max_ctx:
        ids = ids[-model.max_ctx :]
        truncated = True
    input_ids = torch.tensor([ids], dtype=torch.long)

    logits, cache, truncation_info = model.prefill(input_ids)
    seq_start = int(input_ids.shape[1]) - truncation_info["tokens_dropped"]

    if context_mode == "strict" and (truncated or truncation_info["truncated"]):
        note = "context_limit_reached"
        if provenance is not None:
            provenance.extend(origins + [note])
        return GenerationResult(
            prompt=prompt,
            continuation="",
            full_text=prompt + "\n[CONTEXT_LIMIT_REACHED] " + note,
            finish_reason="context_limit_reached",
            tokens_generated=0,
            truncated=True,
            provenance=origins if provenance is None else provenance,
        )

    if truncated or truncation_info["truncated"]:
        origins.append("prompt_truncated")

    generated: List[int] = []
    eos_id = tokenizer.token_to_id.get(EOS)
    exp_end_id = tokenizer.token_to_id.get(EXP_END)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    cur_text = prompt
    finish_reason = "length"
    for step_idx in range(max_new_tokens):
        step_logits = logits[0] if logits.dim() == 2 else logits[0, -1]
        if vexx is not None:
            masked = vexx.apply_in_flight_mask(
                step_logits.detach().cpu().numpy(),
                cur_text,
                tokenizer.id_to_token,
                mode=mode,
                remaining_token_budget=max_new_tokens - step_idx,
            )
            step_logits = torch.tensor(
                masked, dtype=step_logits.dtype, device=step_logits.device
            )
            if not bool(torch.isfinite(step_logits).any()):
                raise RuntimeError("Vexx 3.5 masked every token; refusing arbitrary argmax")

        next_id = int(sample_next(step_logits.unsqueeze(0), temperature, top_p, generator).item())
        if eos_id is not None and next_id == eos_id:
            finish_reason = "eos"
            break
        generated.append(next_id)
        cur_text = prompt + tokenizer.decode(generated, preserve_protocol=True)

        # 1. Tier-2 inline CAS splice inside <think>: <<expr=
        if (
            vexx is not None
            and cur_text.endswith("=")
            and "<<" in cur_text
            and vexx.crane_phase(cur_text) == "UNCONSTRAINED_THINK"
        ):
            m_cas = re.search(r"<<([^<>\n]+)=$", cur_text)
            if m_cas:
                value = vexx.crane_cas_evaluate(m_cas.group(1).strip()) + ">>"
                if not value.startswith("ERROR"):
                    origins.append("deterministic_calculation")
                    logits, cache = model.decode_step(
                        torch.tensor([[next_id]], dtype=torch.long), cache, seq_start
                    )
                    seq_start += 1
                    for cid in tokenizer.encode(value, add_bos=False, add_eos=False):
                        logits, cache = model.decode_step(
                            torch.tensor([[cid]], dtype=torch.long), cache, seq_start
                        )
                        seq_start += 1
                        generated.append(cid)
                    cur_text = prompt + tokenizer.decode(generated, preserve_protocol=True)
                    continue

        # 2. Tier-5 live <experiment>...</experiment> sandbox splice
        if vexx is not None and exp_end_id is not None and next_id == exp_end_id:
            last_open = cur_text.rfind(EXP_START)
            last_close = cur_text.rfind(EXP_END)
            if last_open != -1 and last_close > last_open:
                script_body = cur_text[last_open + len(EXP_START) : last_close].strip()
                obs = f" => [OBS: {vexx.execute_experiment_block(script_body)}] "
                origins.append("experiment_execution")
                logits, cache = model.decode_step(
                    torch.tensor([[next_id]], dtype=torch.long), cache, seq_start
                )
                seq_start += 1
                for cid in tokenizer.encode(obs, add_bos=False, add_eos=False):
                    logits, cache = model.decode_step(
                        torch.tensor([[cid]], dtype=torch.long), cache, seq_start
                    )
                    seq_start += 1
                    generated.append(cid)
                cur_text = prompt + tokenizer.decode(generated, preserve_protocol=True)
                continue

        logits, cache = model.decode_step(
            torch.tensor([[next_id]], dtype=torch.long), cache, seq_start
        )
        seq_start += 1

    continuation = tokenizer.decode(generated, preserve_protocol=True)
    full_text = prompt + continuation
    if provenance is not None:
        provenance.extend(origins)
    return GenerationResult(
        prompt=prompt,
        continuation=continuation,
        full_text=full_text,
        finish_reason=finish_reason,
        tokens_generated=len(generated),
        truncated=truncated,
        provenance=origins if provenance is None else provenance,
    )
