#Qwen/Qwen2.5-0.5B-Instruct

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import benches._bootstrap  # noqa: F401
from patches.transformers_no_flash_attn import no_flash_attn_imports


def sync(device: str):
    # GPU backends are often async; synchronize for accurate timing.
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def prefill(model, input_ids):
    sync(model.device.type)
    t0 = time.perf_counter()
    out = model(input_ids, use_cache=True)
    sync(model.device.type)
    dt = time.perf_counter() - t0
    return dt, out.past_key_values, out.logits[:, -1, :]


@torch.inference_mode()
def decode_loop(model, last_token, past_key_values, steps, progress_every=8):
    sync(model.device.type)
    t0 = time.perf_counter()

    per_step = []
    for i in range(steps):
        sync(model.device.type)
        s0 = time.perf_counter()

        out = model(last_token, use_cache=True, past_key_values=past_key_values)

        sync(model.device.type)
        s1 = time.perf_counter()
        per_step.append(s1 - s0)

        logits = out.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        last_token = next_token
        past_key_values = out.past_key_values

        if progress_every > 0 and (i + 1) % progress_every == 0:
            done = i + 1
            avg_ms = (sum(per_step[-progress_every:]) / progress_every) * 1000.0
            print(f"[progress] decode {done}/{steps} ({(done/steps)*100:.1f}%)  avg_step={avg_ms:.2f}ms")
    
    mx = max(per_step)
    idx = per_step.index(mx) + 1
    print(f"[debug] worst_step={mx*1000:.2f}ms at token={idx}")
    dt = time.perf_counter() - t0
    return dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])  # baseline choices
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--progress-every", type=int, default=8)
    ap.add_argument("--gqa-broadcast", action="store_true",
                help="Patch Qwen2 SDPA attention to avoid KV repeat (MPS only).")
    args = ap.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS not available. Check PyTorch build and macOS version.")

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print(f"Loading: {args.model}")
    print(f"device={args.device} dtype={args.dtype} attn={args.attn}")
    print(f"prompt_len={args.prompt_len} new_tokens={args.new_tokens}")

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
    # Create a fixed-length prompt (no tokenization variance).
    vocab = getattr(model.config, "vocab_size", None)
    if vocab is None:
        raise SystemExit("model.config.vocab_size is missing; can't make synthetic input_ids.")

    input_ids = torch.randint(
        low=0,
        high=vocab,
        size=(1, args.prompt_len),
        device=args.device,
        dtype=torch.long
    )

    # Warmup
    for w in range(args.warmup):
        dt_prefill, pkv, logits = prefill(model, input_ids)
        last = torch.argmax(logits, dim=-1, keepdim=True)
        dt_decode = decode_loop(model, last, pkv, steps=args.new_tokens, progress_every=0)
        print(f"[warmup {w+1}/{args.warmup}] prefill={dt_prefill*1000:.2f}ms decode(8)={dt_decode*1000:.2f}ms")

    # Measure prefill
    dt_prefill, pkv, logits = prefill(model, input_ids)
    last = torch.argmax(logits, dim=-1, keepdim=True)

    # Measure decode
    dt_decode = decode_loop(
        model, last, pkv,
        steps=args.new_tokens,
        progress_every=args.progress_every
    )

    tok_s = args.new_tokens / dt_decode if dt_decode > 0 else float("inf")

    print("\nRESULTS")
    print(f"prefill: {dt_prefill*1000:.2f} ms  (prompt_len={args.prompt_len})")
    print(f"decode : {dt_decode:.3f} s  for {args.new_tokens} tokens => {tok_s:.2f} tok/s")


if __name__ == "__main__":
    main()