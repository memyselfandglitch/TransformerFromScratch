"""Tiny local transformer KV-cache playground.

Run this to inspect how KV-cache shapes and memory change when you adjust
layers, heads, KV heads, batch size, dtype, prompt length, and generated tokens.

Example:
    python3 kv_cache_playground.py --n-layers 4 --n-heads 8 --n-kv-heads 2 \
        --d-model 256 --prompt-len 16 --new-tokens 8 --dtype float16
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


KVCache = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class TinyConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 4
    max_seq_len: int = 1024

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("--d-model must be divisible by --n-heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("--n-heads must be divisible by --n-kv-heads")


class TinyAttention(nn.Module):
    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        batch_size, seq_len, _ = x.shape
        head_dim = self.cfg.head_dim

        q = self.q_proj(x).view(batch_size, seq_len, self.cfg.n_heads, head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.cfg.n_kv_heads, head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.cfg.n_kv_heads, head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v)

        # Repeat KV heads for grouped-query or multi-query attention.
        if self.cfg.n_kv_heads != self.cfg.n_heads:
            repeat_factor = self.cfg.n_heads // self.cfg.n_kv_heads
            k_for_attention = k.repeat_interleave(repeat_factor, dim=1)
            v_for_attention = v.repeat_interleave(repeat_factor, dim=1)
        else:
            k_for_attention = k
            v_for_attention = v

        total_len = k_for_attention.shape[2]
        scores = q @ k_for_attention.transpose(-2, -1)
        scores = scores / (head_dim**0.5)

        # When seq_len > 1 this is prompt/prefill mode, so apply a causal mask.
        if seq_len > 1:
            start = total_len - seq_len
            query_positions = torch.arange(start, total_len, device=x.device)[:, None]
            key_positions = torch.arange(total_len, device=x.device)[None, :]
            mask = key_positions > query_positions
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=-1)
        out = weights @ v_for_attention
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.cfg.d_model)
        return self.out_proj(out), new_cache


class TinyBlock(nn.Module):
    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = TinyAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        attn_out, new_cache = self.attn(self.ln_1(x), cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class TinyTransformer(nn.Module):
    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TinyBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        caches: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, List[KVCache]]:
        x = self.token_embedding(input_ids)
        new_caches: List[KVCache] = []

        for layer_index, block in enumerate(self.blocks):
            layer_cache = None if caches is None else caches[layer_index]
            x, next_cache = block(x, layer_cache)
            new_caches.append(next_cache)

        logits = self.lm_head(self.ln_f(x))
        return logits, new_caches


def dtype_from_name(name: str) -> torch.dtype:
    lookup = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return lookup[name]


def cache_bytes(caches: List[KVCache]) -> int:
    total = 0
    for key, value in caches:
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return total


def theoretical_cache_bytes(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    return batch_size * seq_len * n_layers * 2 * n_kv_heads * head_dim * element_size


def mib(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def print_cache_report(step_label: str, caches: List[KVCache], cfg: TinyConfig) -> None:
    key, value = caches[0]
    print(f"\n{step_label}")
    print(f"  layer 0 key shape:   {tuple(key.shape)}")
    print(f"  layer 0 value shape: {tuple(value.shape)}")
    print("  shape meaning:       (batch, n_kv_heads, cached_tokens, head_dim)")
    print(f"  layers cached:       {len(caches)}")
    print(f"  actual cache memory: {mib(cache_bytes(caches)):.4f} MiB")
    print(
        "  formula memory:      "
        "batch * cached_tokens * layers * 2(K,V) * n_kv_heads * head_dim * dtype_bytes"
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--new-tokens", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    cfg = TinyConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.prompt_len + args.new_tokens,
    )
    cfg.validate()

    device = "cpu"
    model = TinyTransformer(cfg).to(device=device, dtype=dtype).eval()
    input_ids = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(args.batch_size, args.prompt_len),
        device=device,
    )

    print("Tiny transformer KV-cache playground")
    print(f"  device:       {device}")
    print(f"  dtype:        {args.dtype}")
    print(f"  d_model:      {cfg.d_model}")
    print(f"  layers:       {cfg.n_layers}")
    print(f"  query heads:  {cfg.n_heads}")
    print(f"  KV heads:     {cfg.n_kv_heads}")
    print(f"  head_dim:     {cfg.head_dim}")
    print(f"  batch size:   {args.batch_size}")

    logits, caches = model(input_ids)
    print_cache_report(f"After prompt prefill ({args.prompt_len} tokens)", caches, cfg)

    for token_index in range(1, args.new_tokens + 1):
        next_token = logits[:, -1:].argmax(dim=-1)
        logits, caches = model(next_token, caches)
        cached_tokens = args.prompt_len + token_index
        expected = theoretical_cache_bytes(
            args.batch_size,
            cached_tokens,
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            dtype,
        )
        print_cache_report(f"After decode step {token_index} ({cached_tokens} cached tokens)", caches, cfg)
        print(f"  expected memory:     {mib(expected):.4f} MiB")


if __name__ == "__main__":
    main()
