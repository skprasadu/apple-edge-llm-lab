#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"
PROMPT_FILE="${2:-prompt_approx_4096.txt}"
MAX_TOKENS="${3:-64}"
MAX_KV="${4:-4096}"

python -m mlx_lm generate \
  --model "$MODEL" \
  --prompt - \
  --max-tokens "$MAX_TOKENS" \
  --max-kv-size "$MAX_KV" < "$PROMPT_FILE"