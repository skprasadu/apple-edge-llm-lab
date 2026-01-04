# apple-edge-llm-lab

Benchmarks + patches for LLM inference on Apple Silicon (MPS/MLX/Metal).

## Benchmarks

<!-- BENCHMARKS:START -->
| run | model | device | headline(pl/nt/cache) | gen tok/s (base) | gen tok/s (gqa) | gen speedup | prefill ms (base) | prefill ms (gqa) | prefill speedup | git |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [20260103_175200](results/summary/20260103_175200.md) | Qwen/Qwen2.5-0.5B-Instruct | mps/float16 | 4096/64/dynamic | 5.81 | 7.43 | 1.28x | 4191 | 4420 | 0.95x | 08eb868* |
| [20260103_171550](results/summary/20260103_171550.md) | Qwen/Qwen2.5-0.5B-Instruct | mps/float16 | 4096/64/dynamic | 6.85 | 8.26 | 1.21x | 4641 | 4105 | 1.13x | 08eb868* |
<!-- BENCHMARKS:END -->