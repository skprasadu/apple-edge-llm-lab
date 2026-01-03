# patches/qwen2_gqa_broadcast_patch.py
from __future__ import annotations

import sys
from typing import Any, Optional

import torch

from .gqa_broadcast_sdpa import gqa_broadcast_sdpa

ATTN_IMPL = "sdpa_gqa_broadcast"


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _broadcast_mask_for_gqa(
    attention_mask: torch.Tensor,
    *,
    B: int,
    H: int,
    KV: int,
    rep: int,
) -> torch.Tensor:
    """
    Convert attention_mask into something compatible with [B*KV, rep, Q, K] SDPA.

    Supported input shapes:
      - [B, 1, Q, K]   -> [B*KV, 1, Q, K]
      - [B, H, Q, K]   -> [B*KV, rep, Q, K]
      - [B, KV, Q, K]  -> [B*KV, rep, Q, K]
      - [B, Q, K]      -> [B*KV, Q, K]
    """
    if attention_mask.dim() == 4:
        b, hm, q, k = attention_mask.shape
        if b != B:
            raise ValueError(f"attention_mask batch {b} != B {B}")

        if hm == 1:
            return attention_mask[:, :, None, :, :].expand(B, 1, KV, q, k).reshape(B * KV, 1, q, k)

        if hm == H:
            return attention_mask.view(B, KV, rep, q, k).reshape(B * KV, rep, q, k)

        if hm == KV:
            return attention_mask[:, :, None, :, :].expand(B, KV, rep, q, k).reshape(B * KV, rep, q, k)

        raise ValueError(f"Unsupported attention_mask head dim: {hm} (expected 1, H={H}, or KV={KV})")

    if attention_mask.dim() == 3:
        b, q, k = attention_mask.shape
        if b != B:
            raise ValueError(f"attention_mask batch {b} != B {B}")
        return attention_mask[:, None, :, :].expand(B, KV, q, k).reshape(B * KV, q, k)

    raise ValueError(f"Unsupported attention_mask dims: {attention_mask.dim()} (expected 3 or 4)")


def apply_qwen2_gqa_broadcast_patch(verbose: bool = True) -> str:
    """
    Register a new attention implementation key with Transformers:
      ATTN_IMPL = "sdpa_gqa_broadcast"

    This avoids breaking AttentionInterface (no dict replacement).
    """
    import transformers.modeling_utils as mu

    iface = getattr(mu, "ALL_ATTENTION_FUNCTIONS", None)
    if iface is None or not hasattr(iface, "register") or not hasattr(iface, "valid_keys"):
        raise RuntimeError(
            "Transformers attention dispatch is not compatible with this patch.\n"
            "Expected modeling_utils.ALL_ATTENTION_FUNCTIONS to be an AttentionInterface."
        )

    # If already registered, do nothing.
    if ATTN_IMPL in iface.valid_keys():
        if verbose:
            _stderr(f"[apple-edge-llm-lab] {ATTN_IMPL} already registered (skipping).")
        return ATTN_IMPL

    orig = iface.get("sdpa", None) if hasattr(iface, "get") else iface["sdpa"]
    if orig is None or not callable(orig):
        raise RuntimeError("Could not access ALL_ATTENTION_FUNCTIONS['sdpa'].")

    fastpath_logged = {"done": False}

    def sdpa_gqa_broadcast_forward(
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        **kwargs: Any,
    ):
        # Delegate if inputs aren't the expected dense tensors.
        if not (isinstance(query, torch.Tensor) and isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor)):
            return orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)

        # Only target Apple MPS
        if query.device.type != "mps":
            return orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)

        # Nested tensors: leave to upstream impl
        if getattr(key, "is_nested", False) or getattr(query, "is_nested", False):
            return orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)

        # Expected [B,H,Q,D] and [B,KV,K,D]
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            return orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)

        B, H, Q, D = query.shape
        B2, KV, K, D2 = key.shape
        if B2 != B or D2 != D:
            return orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
        if value.shape[0] != B or value.shape[1] != KV or value.shape[-1] != D:
            return orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)

        # Not GQA -> delegate
        if KV == H or KV == 0 or (H % KV) != 0:
            return orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)

        rep = H // KV

        # Match upstream masking semantics: slice mask to key length.
        causal_mask = attention_mask
        if isinstance(causal_mask, torch.Tensor) and causal_mask.dim() == 4:
            causal_mask = causal_mask[:, :, :, :K]

        # Upstream sets is_causal like: (q_len > 1) and (mask is None)
        if is_causal is None:
            is_causal = (Q > 1) and (causal_mask is None)

        # Convert mask to broadcast form (best-effort)
        mask2 = None
        if isinstance(causal_mask, torch.Tensor):
            try:
                mask2 = _broadcast_mask_for_gqa(causal_mask, B=B, H=H, KV=KV, rep=rep)
            except Exception:
                return orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)

        if verbose and not fastpath_logged["done"]:
            fastpath_logged["done"] = True
            _stderr(
                f"[apple-edge-llm-lab] FASTPATH HIT: {ATTN_IMPL} "
                f"(B={B} H={H} KV={KV} rep={rep} Q={Q} K={K} D={D})"
            )

        out = gqa_broadcast_sdpa(
            query,
            key,
            value,
            is_causal=bool(is_causal),
            attn_mask=mask2,
            dropout_p=float(dropout),
            scale=scaling,
        )

        # SDPA attention interface returns [B, Q, H, D]
        out = out.transpose(1, 2).contiguous()
        return out, None

    # Register new key without breaking AttentionInterface
    iface.register(ATTN_IMPL, sdpa_gqa_broadcast_forward)

    if verbose:
        _stderr(f"[apple-edge-llm-lab] Registered attention implementation '{ATTN_IMPL}'.")

    return ATTN_IMPL