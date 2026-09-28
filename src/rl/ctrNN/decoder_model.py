import dataclasses
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import mode, nn

from jdh import JDHConfig

"""
Decoder do create tokens from JEPA embedding prediction
--> B,T,D
"""


class Decoder(nn.Module) :
    def __init__(self, config: JDHConfig):
        super().__init__()
        D = config.n_embd

        self.net = nn.Sequential(nn.Linear(D,D), nn.ReLU())
        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

    def forward(self, x, targets=None):
        logits = self.net(x) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
            

    @torch.no_grad()
    def generate(
        self,
        x,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            x = self.net(x)
            logits = x @ self.lm_head
            if top_k is not None:
                v, _ = torch.topk(logits, top_k, dim=-1)
                # v: B, T, k -> k-th largest per position, keep dims for broadcast
                logits = logits.masked_fill(logits < v[:, :, -1:], -float("Inf"))
            logits = logits[:, -1, :] / temperature # scale by temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            # NOTE: this decoder maps embeddings -> tokens, not tokens ->
            # embeddings, so we cannot feed idx_next back in. Each step re-reads
            # the same embedding sequence; for true autoregression you would
            # need a token embedding here or re-encode the grown idx with the
            # JDH encoder + predictor.
        return idx