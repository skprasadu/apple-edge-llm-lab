import argparse, time
import torch
import torch.nn.functional as F

def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()

def bench(fn, warmup, repeats, dev):
    for _ in range(warmup):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync(dev)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / repeats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16", choices=["float16","float32"])
    ap.add_argument("--q", type=int, default=1)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--rep", type=int, default=7)  # attn_heads = kv_heads * rep
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--causal", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    B = 1
    kv = args.kv_heads
    rep = args.rep
    H = kv * rep
    Q = args.q
    K = args.k
    D = args.d

    q = torch.randn(B, H, Q, D, device=dev, dtype=dtype)
    k = torch.randn(B, kv, K, D, device=dev, dtype=dtype)
    v = torch.randn(B, kv, K, D, device=dev, dtype=dtype)

    def baseline():
        k_full = k.repeat_interleave(rep, dim=1)
        v_full = v.repeat_interleave(rep, dim=1)
        return F.scaled_dot_product_attention(q, k_full, v_full, dropout_p=0.0, is_causal=args.causal)

    def broadcast():
        # reshape q into groups: [B, kv, rep, Q, D] -> [B*kv, rep, Q, D]
        q2 = q.view(B, kv, rep, Q, D).reshape(B * kv, rep, Q, D)

        # expand k/v across rep with stride-0 (NO materialization here)
        k2 = k[:, :, None, :, :].expand(B, kv, rep, K, D).reshape(B * kv, rep, K, D)
        v2 = v[:, :, None, :, :].expand(B, kv, rep, K, D).reshape(B * kv, rep, K, D)

        y2 = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0, is_causal=args.causal)
        # back to [B, H, Q, D]
        return y2.reshape(B, kv, rep, Q, D).reshape(B, H, Q, D)

    # correctness (single check)
    with torch.no_grad():
        y_ref = baseline()
        y_alt = broadcast()
        max_err = (y_ref - y_alt).abs().max().item()

    # storage sharing check (k2 should share, since only expand + flatten B*kv)
    k2 = k[:, :, None, :, :].expand(B, kv, rep, K, D).reshape(B * kv, rep, K, D)
    shares = (k2.untyped_storage().data_ptr() == k.untyped_storage().data_ptr())

    ms_base = bench(baseline, args.warmup, args.repeats, dev)
    ms_brdc = bench(broadcast, args.warmup, args.repeats, dev)

    print(f"SDPA GQA test dev={args.device} dtype={args.dtype} B={B} kv={kv} rep={rep} H={H} Q={Q} K={K} D={D} causal={args.causal}")
    print(f"k2_shares_storage={shares}  k2_contig={k2.is_contiguous()}")
    print(f"max_abs_err={max_err:.6g}")
    print(f"baseline (repeat_kv + sdpa):  {ms_base:.3f} ms/call")
    print(f"broadcast (no repeat_kv op):  {ms_brdc:.3f} ms/call")
    print(f"speedup: {ms_base/ms_brdc:.2f}x")

if __name__ == "__main__":
    main()