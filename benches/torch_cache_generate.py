import argparse, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import benches._bootstrap  # noqa: F401
from patches.transformers_no_flash_attn import no_flash_attn_imports

def sync(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--prompt-len", type=int, default=4096)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--cache-impl", default="dynamic", choices=["dynamic", "static", "offloaded", "quantized"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--gqa-broadcast", action="store_true",
                help="Patch Qwen2 SDPA attention to avoid KV repeat (MPS only).")
    args = ap.parse_args()

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    attn_impl = args.attn

    if args.gqa_broadcast:
        from patches.qwen2_gqa_broadcast_patch import apply_qwen2_gqa_broadcast_patch, ATTN_IMPL
        apply_qwen2_gqa_broadcast_patch(verbose=True)
        if attn_impl == "sdpa":
            attn_impl = ATTN_IMPL

    with no_flash_attn_imports():
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch_dtype,
        attn_implementation=attn_impl,
    ).to(args.device)
    model.eval()
    print(f"attn={attn_impl}")
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, args.prompt_len), device=args.device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, device=args.device)

    # Warmup
    for _ in range(args.warmup):
        sync(args.device)
        _ = model.generate(
            input_ids,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
            cache_implementation=args.cache_impl,
            attention_mask=attention_mask,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        sync(args.device)

    # Timed
    sync(args.device)
    t0 = time.perf_counter()
    out = model.generate(
        input_ids,
        max_new_tokens=args.new_tokens,
        do_sample=False,
        use_cache=True,
        cache_implementation=args.cache_impl,
        attention_mask=attention_mask,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    sync(args.device)
    t1 = time.perf_counter()

    total = t1 - t0
    print(f"cache_impl={args.cache_impl} prompt_len={args.prompt_len} new_tokens={args.new_tokens}")
    print(f"total: {total:.3f}s  =>  {args.new_tokens/total:.2f} tok/s (end-to-end)")

if __name__ == "__main__":
    main()