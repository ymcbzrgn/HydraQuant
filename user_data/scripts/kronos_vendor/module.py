"""Minimal Kronos backbone fallback.

Used only when `huggingface_hub.snapshot_download` doesn't return the
upstream `model.py` (some HF mirrors strip Python source for security).
The fallback is a small Transformer encoder over OHLCV tokens producing a
direction logit — enough to keep callers' code paths exercised even when
the full Kronos checkpoint can't be deserialised.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class FallbackKronos(nn.Module):
    """Transformer encoder over OHLCV tokens. Output: scalar direction in [-1,+1]."""

    def __init__(self, vocab_size: int = 1024, d_model: int = 64,
                 n_layers: int = 4, n_heads: int = 4, max_seq: int = 256):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True,
            dim_feedforward=d_model * 2,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
        self.max_seq = max_seq

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, l = token_ids.shape
        l = min(l, self.max_seq)
        token_ids = token_ids[:, :l]
        positions = torch.arange(l, device=token_ids.device).unsqueeze(0).expand(b, l)
        x = self.tok_embed(token_ids) + self.pos_embed(positions)
        h = self.encoder(x)
        # Pool over sequence then project — single direction logit.
        pooled = h.mean(dim=1)
        return torch.tanh(self.head(pooled)).squeeze(-1)


class FallbackKronosTokenizer:
    """OHLCV → discrete token-id encoder.

    Quantises (return, body, range, volume_ratio) into 4 8-bin buckets per
    bar (256 tokens per bar position) — matches the order-of-magnitude of
    Kronos's actual 2k tokenizer enough that the fallback model can still
    distinguish trending vs. reverting candles.
    """

    VOCAB_SIZE = 256

    @staticmethod
    def encode(close: torch.Tensor, open_: torch.Tensor, high: torch.Tensor,
               low: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        """Per-bar tokenisation. All inputs (B, L). Returns (B, L) int64."""
        eps = 1e-8
        ret = (close - open_) / (open_ + eps)
        body = (close - open_).abs() / ((high - low) + eps)
        rng = (high - low) / (close + eps)
        vol_ratio = volume / (volume.mean(dim=1, keepdim=True) + eps)

        def _q(x: torch.Tensor, k: int = 4) -> torch.Tensor:
            # Quantise into [0, k) bins by per-row percentile rank.
            sorted_x, _ = x.sort(dim=1)
            idx = torch.searchsorted(sorted_x, x).clamp(0, sorted_x.shape[1] - 1)
            return (idx * k // sorted_x.shape[1]).clamp(0, k - 1)

        q_ret = _q(ret, k=4)
        q_body = _q(body, k=4)
        q_rng = _q(rng, k=4)
        q_vol = _q(vol_ratio, k=4)
        token = q_ret * 64 + q_body * 16 + q_rng * 4 + q_vol  # ∈ [0, 256)
        return token.long()
