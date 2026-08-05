"""ITFormer + frozen Qwen3 time-language model for weld-quality classification.

Adaptation of the ICML 2025 ITFormer pipeline (arXiv:2506.20093) to binary
classification via likelihood scoring:

  prompt:  <context text> <25 placeholder tokens> Question: Is this weld good
           or bad? Answer:
  - encoder + ITFormer fuse the (V, I) cycle into 25 temporal tokens
  - fused tokens (projected to LLM hidden size) replace the placeholders in
    the prompt embeddings (and ONLY those positions)
  - training: cross-entropy at the answer position only
  - eval: single teacher-forced-style forward; score = logit[good] - logit[bad]
    at the position that predicts the answer. No free generation (base Qwen3
    would generate gibberish; metrics are likelihood-based only).

The LLM is fully frozen (matches the paper: trainable < 1%).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from wndt.models.encoder import WeldTSEncoder
from wndt.models.itformer import ITFormer
from wndt.utils.logging import get_logger

log = get_logger(__name__)

CONTEXT_TEXT = ("Welding cycle signals (current and voltage, 200 samples each) "
                "are given. ")
QUESTION_TEXT = "Question: Is this weld good or bad? Answer:"
PLACEHOLDER_CANDIDATES = ["<|extra_0|>", "<|image_pad|>", "<|pad|>"]
ANSWER_GOOD = "good"
ANSWER_BAD = "bad"


@dataclass
class PromptLayout:
    context_ids: list[int]
    question_ids: list[int]
    placeholder_id: int
    prefix_num: int
    good_id: int
    bad_id: int
    full_prefix_ids: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.full_prefix_ids = (self.context_ids
                                + [self.placeholder_id] * self.prefix_num
                                + self.question_ids)

    @property
    def ph_start(self) -> int:
        return len(self.context_ids)

    @property
    def text_ids(self) -> list[int]:
        return self.context_ids + self.question_ids


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) == 1, f"{text!r} tokenizes to {len(ids)} tokens: {ids}"
    return ids[0]


def resolve_placeholder_id(tokenizer) -> int:
    """Pick an existing single-token placeholder, else add a new special token."""
    for cand in PLACEHOLDER_CANDIDATES:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            log.info("placeholder token: %r -> id %d", cand, ids[0])
            return ids[0]
    tok = "<|wndt_ts_pad|>"
    tokenizer.add_tokens([tok], special_tokens=True)
    ids = tokenizer.encode(tok, add_special_tokens=False)
    assert len(ids) == 1
    log.warning("no existing placeholder worked; added new token %r -> id %d "
                "(LLM embeddings must be resized; the new row is always "
                "overwritten before the LLM forward, so it never trains)", tok, ids[0])
    return ids[0]


def build_prompt_layout(tokenizer, prefix_num: int) -> PromptLayout:
    return PromptLayout(
        context_ids=tokenizer.encode(CONTEXT_TEXT, add_special_tokens=False),
        question_ids=tokenizer.encode(QUESTION_TEXT, add_special_tokens=False),
        placeholder_id=resolve_placeholder_id(tokenizer),
        prefix_num=prefix_num,
        good_id=_single_token_id(tokenizer, ANSWER_GOOD),
        bad_id=_single_token_id(tokenizer, ANSWER_BAD),
    )


class ITFormerTLM(nn.Module):
    """Encoder + ITFormer bridge + frozen Qwen3."""

    def __init__(self, llm_path: str, *, prefix_num: int = 25, d_model: int = 512,
                 n_heads: int = 8, it_layers: int = 2, enc_layers: int = 4,
                 patch_len: int = 20, seq_len: int = 200, n_vars: int = 2,
                 dropout: float = 0.1, attn_impl: str = "sdpa"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        self.layout = build_prompt_layout(self.tokenizer, prefix_num)

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, dtype=torch.bfloat16, attn_implementation=attn_impl,
            trust_remote_code=True)
        if len(self.tokenizer) > self.llm.get_input_embeddings().weight.shape[0]:
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.config.use_cache = False
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.gradient_checkpointing_enable()
        llm_hidden = self.llm.config.hidden_size

        self.encoder = WeldTSEncoder(seq_len=seq_len, n_vars=n_vars,
                                     patch_len=patch_len, stride=patch_len,
                                     d_model=d_model, n_heads=n_heads,
                                     e_layers=enc_layers, dropout=dropout)
        self.itformer = ITFormer(d_model=d_model, n_heads=n_heads,
                                 n_layers=it_layers, prefix_num=prefix_num,
                                 dropout=dropout,
                                 max_text_len=len(self.layout.text_ids) + 8)
        self.q_proj = nn.Linear(llm_hidden, d_model, bias=False)
        self.ts_proj = nn.Linear(d_model, llm_hidden, bias=False)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        log.info("trainable params: %.2fM / %.2fM (%.3f%%)",
                 n_train / 1e6, n_total / 1e6, 100.0 * n_train / n_total)

    # ------------------------------------------------------------------ helpers
    @property
    def good_id(self) -> int:
        return self.layout.good_id

    @property
    def bad_id(self) -> int:
        return self.layout.bad_id

    def _llm_embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(ids)

    def _fuse(self, waves: torch.Tensor) -> torch.Tensor:
        """waves (B, 2, 200) -> fused tokens projected to LLM space (B, P, H)."""
        memory = self.encoder.memory(waves.transpose(1, 2))      # (B, P, V, d)
        device = waves.device
        text_ids = torch.tensor(self.layout.text_ids, dtype=torch.long,
                                device=device)
        # LLM embeddings are bf16; cast to the bridge dtype (fp32 outside
        # autocast, handled by autocast inside the trainer)
        text_embeds = self._llm_embed(text_ids).to(self.q_proj.weight.dtype)
        q = self.q_proj(text_embeds).unsqueeze(0).expand(waves.shape[0], -1, -1)
        fused = self.itformer(q, memory)                          # (B, 25, d)
        return self.ts_proj(fused)

    def _prefix_embeds(self, waves: torch.Tensor) -> torch.Tensor:
        """Prompt embeddings with fused tokens injected at placeholder positions."""
        b = waves.shape[0]
        device = waves.device
        layout = self.layout
        prefix_ids = torch.tensor(layout.full_prefix_ids, dtype=torch.long,
                                  device=device)
        embeds = self._llm_embed(prefix_ids)                      # (T, H)
        embeds = embeds.unsqueeze(0).expand(b, -1, -1).clone()
        ts_embeds = self._fuse(waves).to(embeds.dtype)
        ph = layout.ph_start
        embeds[:, ph:ph + layout.prefix_num] = ts_embeds
        return embeds

    # ------------------------------------------------------------------ forward
    def forward(self, waves: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Training forward: CE loss at the answer position only.

        waves: (B, 2, 200); labels: (B,) in {0, 1}. Returns scalar loss.
        """
        prefix_embeds = self._prefix_embeds(waves)                # (B, T, H)
        answer_ids = torch.where(labels == 1, self.good_id, self.bad_id)
        answer_ids_t = answer_ids.to(prefix_embeds.device)
        answer_embeds = self._llm_embed(answer_ids_t).unsqueeze(1)  # (B, 1, H)
        inputs_embeds = torch.cat([prefix_embeds, answer_embeds], dim=1)
        out = self.llm(inputs_embeds=inputs_embeds)
        logits = out.logits[:, -2, :]                             # predicts answer
        return F.cross_entropy(logits, answer_ids_t)

    @torch.no_grad()
    def score(self, waves: torch.Tensor) -> torch.Tensor:
        """Likelihood score logit(good) - logit(bad); >0 predicts 'good'.

        waves: (B, 2, 200). Returns (B,) float.
        """
        prefix_embeds = self._prefix_embeds(waves)
        out = self.llm(inputs_embeds=prefix_embeds)
        logits = out.logits[:, -1, :].float()
        return logits[:, self.good_id] - logits[:, self.bad_id]
