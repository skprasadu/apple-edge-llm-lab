# patches/gqa_broadcast_sdpa.py
from __future__ import annotations

import inspect
import math
from typing import Optional

import torch
import torch.nn.functional as F

# Some PyTorch builds expose SDPA as a C++ builtin with no inspectable signature.
# If we can't introspect, treat it as "no scale kwarg" and emulate scaling by scaling q.
try:
    _SDPA_HAS_SCALE = "scale" in inspect.signature(F.scaled_dot_product_attention).parameters
except (ValueError, TypeError):
    _SDPA_HAS_SCALE = False


def gqa_broadcast_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    attn_mask=None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    q: [B, H, Q, D]
    k: [B, KV, K, D]
    v: [B, KV, K, D]
    returns: [B, H, Q, D]
    """
    B, H, Q, D = q.shape
    _, KV, K, _ = k.shape
    assert H % KV == 0, f"H ({H}) must be multiple of KV ({KV})"
    rep = H // KV

    # [B, H, Q, D] -> [B*KV, rep, Q, D]
    q2 = q.view(B, KV, rep, Q, D).reshape(B * KV, rep, Q, D)

    # [B, KV, K, D] -> [B*KV, rep, K, D] via expand (no materialization)
    k2 = k[:, :, None, :, :].expand(B, KV, rep, K, D).reshape(B * KV, rep, K, D)
    v2 = v[:, :, None, :, :].expand(B, KV, rep, K, D).reshape(B * KV, rep, K, D)

    # If SDPA doesn't support `scale=`, emulate:
    # SDPA uses default 1/sqrt(D). Scaling q by (scale*sqrt(D)) gives the same effect.
    if scale is not None and not _SDPA_HAS_SCALE:
        q2 = q2 * (float(scale) * math.sqrt(D))
        scale = None

    kwargs = {}
    if _SDPA_HAS_SCALE:
        kwargs["scale"] = scale

    y2 = F.scaled_dot_product_attention(
        q2,
        k2,
        v2,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        **kwargs,
    )

    return y2.reshape(B, KV, rep, Q, D).reshape(B, H, Q, D)