import argparse, time
import torch
import torch.nn.functional as F

def ms(x): return x * 1000.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=1)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--causal", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    # Shapes: [B, H, Q, D] and [B, H, K, D]
    B = 1
    q = torch.randn(B, args.h, args.q, args.d, device=device, dtype=dtype)
    k = torch.randn(B, args.h, args.k, args.d, device=device, dtype=dtype)
    v = torch.randn(B, args.h, args.k, args.d, device=device, dtype=dtype)

    # Warmup
    for _ in range(args.warmup):
        _ = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=args.causal)
        if device.type == "mps":
            torch.mps.synchronize()

    # Timing
    t0 = time.time()
    for _ in range(args.repeats):
        _ = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=args.causal)
    if device.type == "mps":
        torch.mps.synchronize()
    t1 = time.time()

    per_call_ms = ms((t1 - t0) / args.repeats)
    print(f"SDPA {args.device} dtype={args.dtype} B=1 H={args.h} Q={args.q} K={args.k} D={args.d} causal={args.causal}")
    print(f"{per_call_ms:.3f} ms / call")

if __name__ == "__main__":
    main()