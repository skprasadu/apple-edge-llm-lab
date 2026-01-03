# apple-edge-llm-lab

Benchmarks + patches for LLM inference on Apple Silicon (MPS/MLX/Metal).

## Benchmarks

<!-- BENCHMARKS:START -->
| run | model | device | headline(pl/nt/cache) | gen tok/s (base) | gen tok/s (gqa) | gen speedup | prefill ms (base) | prefill ms (gqa) | prefill speedup | git |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [20260103_105235](results/summary/20260103_105235.md) | Qwen/Qwen2.5-0.5B-Instruct | mps/float16 | 2048/64/dynamic | 17.69 | 20.31 | 1.15x | 18974 | 2619 | 7.25x | 166d2c0* |
| [20260103_083446](results/summary/20260103_083446.md) | Qwen/Qwen2.5-0.5B-Instruct | mps/float16 | 2048/32/dynamic | 10.23 | 11.72 | 1.15x | 1390 | 1373 | 1.01x | 2a37b5b* |
<!-- BENCHMARKS:END -->