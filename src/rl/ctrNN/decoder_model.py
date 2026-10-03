import dataclasses
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import mode, nn

from jdh import JDHConfig
from config import TARGET_SIZE

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
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Decode a single next token from a predicted embedding.

        x: B, T, D predicted embedding (only the last position is used).
        Returns: B, 1 token indices.
        """

        x = self.net(x)
        logits = x @ self.lm_head  # B, T, vocab
        if top_k is not None:
            v, _ = torch.topk(logits, top_k, dim=-1)
            # v: B, T, k -> k-th largest per position, keep dims for broadcast
            logits = logits.masked_fill(logits < v[:, :, -1:], -float("Inf"))
        logits = logits[:, -1, :] / temperature  # B, vocab
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return idx_next