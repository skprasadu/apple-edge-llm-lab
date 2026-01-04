# apple-edge-llm-lab

Benchmarks + patches for LLM inference on Apple Silicon (MPS/MLX/Metal).

## Benchmarks

<!-- BENCHMARKS:START -->
| run | model | device | headline(pl/nt/cache) | gen tok/s (base) | gen tok/s (gqa) | gen speedup | gen mem ΔMB (base) | gen mem ΔMB (gqa) | gen mem saving | prefill ms (base) | prefill ms (gqa) | prefill speedup | prefill mem ΔMB (base) | prefill mem ΔMB (gqa) | prefill mem saving | git |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [20260103_182837](results/summary/20260103_182837.md) | Qwen/Qwen2.5-0.5B-Instruct | mps/float16 | 4096/64/dynamic | 5.95 | 6.47 | 1.09x | 2330.3 | 4122.3 | 0.57x | 5369 | 5353 | 1.00x | 6490.3 | 6490.3 | 1.00x | d4d9373* |
<!-- BENCHMARKS:END -->