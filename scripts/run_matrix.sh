#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DEVICE="${DEVICE:-mps}"
DTYPE="${DTYPE:-float16}"
ATTN="${ATTN:-sdpa}"

# Keep these small initially; expand once stable.
PROMPT_LENS=(${PROMPT_LENS:-1024 2048 4096})
NEW_TOKENS_LIST=(${NEW_TOKENS_LIST:-32 64})
CACHE_IMPLS=(${CACHE_IMPLS:-dynamic})

RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-2}"

# Optional: add a one-line note per run:
#   NOTES="gqa-broadcast on; testing cache=dynamic" bash scripts/run_matrix.sh
NOTES="${NOTES:-}"

OUT_DIR="${OUT_DIR:-results/raw/${TS}}"
mkdir -p "${OUT_DIR}"

echo "Writing logs to: ${OUT_DIR}"
echo "MODEL=${MODEL} DEVICE=${DEVICE} DTYPE=${DTYPE} ATTN=${ATTN} RUNS=${RUNS} WARMUP=${WARMUP}"
echo "RUN_ID=${TS}"
echo "OUT_DIR=${OUT_DIR}"

# Repro snapshot for this run (parse_results.py will pick this up)
META="${OUT_DIR}/meta.txt"
{
  echo "timestamp=${TS}"
  echo "model=${MODEL}"
  echo "device=${DEVICE}"
  echo "dtype=${DTYPE}"
  echo "attn=${ATTN}"
  echo "runs=${RUNS}"
  echo "warmup=${WARMUP}"
  echo "prompt_lens=${PROMPT_LENS[*]}"
  echo "new_tokens_list=${NEW_TOKENS_LIST[*]}"
  echo "cache_impls=${CACHE_IMPLS[*]}"
  echo "notes=${NOTES}"

  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "git_sha=$(git rev-parse --short HEAD)"
    if ! git diff --quiet >/dev/null 2>&1 || ! git diff --cached --quiet >/dev/null 2>&1; then
      echo "git_status=dirty"
    else
      echo "git_status=clean"
    fi
  fi
} > "${META}"

run_prefill_decode () {
  local variant="$1"  # baseline|gqa
  local pl="$2"
  local nt="$3"
  local run="$4"

  local -a extra=()
  if [[ "${variant}" == "gqa" ]]; then
    extra+=(--gqa-broadcast)
  fi

  local fn="${OUT_DIR}/torch_prefill_decode__${variant}__pl${pl}__nt${nt}__run${run}.txt"
  PYTHONPATH=. python -u benches/torch_prefill_decode.py \
    --model "${MODEL}" \
    --device "${DEVICE}" --dtype "${DTYPE}" --attn "${ATTN}" \
    --prompt-len "${pl}" --new-tokens "${nt}" \
    --warmup "${WARMUP}" \
    "${extra[@]+"${extra[@]}"}" | tee "${fn}" >/dev/null
}

run_cache_generate () {
  local variant="$1"  # baseline|gqa
  local cache="$2"
  local pl="$3"
  local nt="$4"
  local run="$5"

  local -a extra=()
  if [[ "${variant}" == "gqa" ]]; then
    extra+=(--gqa-broadcast)
  fi

  local fn="${OUT_DIR}/torch_cache_generate__${variant}__cache${cache}__pl${pl}__nt${nt}__run${run}.txt"
  PYTHONPATH=. python -u benches/torch_cache_generate.py \
    --model "${MODEL}" \
    --device "${DEVICE}" --dtype "${DTYPE}" --attn "${ATTN}" \
    --prompt-len "${pl}" --new-tokens "${nt}" \
    --cache-impl "${cache}" \
    --warmup 1 \
    "${extra[@]+"${extra[@]}"}" | tee "${fn}" >/dev/null
}

for pl in "${PROMPT_LENS[@]}"; do
  for nt in "${NEW_TOKENS_LIST[@]}"; do
    for run in $(seq 1 "${RUNS}"); do
      echo "[prefill_decode] baseline pl=${pl} nt=${nt} run=${run}"
      run_prefill_decode baseline "${pl}" "${nt}" "${run}"

      echo "[prefill_decode] gqa      pl=${pl} nt=${nt} run=${run}"
      run_prefill_decode gqa "${pl}" "${nt}" "${run}"
    done

    for cache in "${CACHE_IMPLS[@]}"; do
      for run in $(seq 1 "${RUNS}"); do
        echo "[cache_generate] baseline cache=${cache} pl=${pl} nt=${nt} run=${run}"
        run_cache_generate baseline "${cache}" "${pl}" "${nt}" "${run}"

        echo "[cache_generate] gqa      cache=${cache} pl=${pl} nt=${nt} run=${run}"
        run_cache_generate gqa "${cache}" "${pl}" "${nt}" "${run}"
      done
    done
  done
done

echo "Done. Logs in: ${OUT_DIR}"
