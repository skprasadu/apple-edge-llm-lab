#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from statistics import median


START = "<!-- BENCHMARKS:START -->"
END = "<!-- BENCHMARKS:END -->"


def _git_meta() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        return sha, dirty
    except Exception as e:
        raise SystemExit(
            "ERROR: git metadata unavailable.\n"
            "Run this inside a git checkout and ensure `git` is installed.\n"
            f"Details: {e}"
        )


def _to_int(x: str) -> int:
    return int(x.strip())


def _to_float(x: str) -> float:
    return float(x.strip())


def _med(vals: list[float]) -> float:
    if not vals:
        raise ValueError("median() called with empty list")
    return float(median(vals))

def _available_cache_generate(rows: list[dict[str, str]]) -> list[tuple[str, int, int]]:
    """
    Return (cache_impl, prompt_len, new_tokens) where BOTH baseline and gqa exist
    and end_to_end_tok_s is present.
    """
    seen: dict[tuple[str, int, int], set[str]] = {}
    for r in rows:
        if r.get("bench") != "cache_generate":
            continue
        cache = (r.get("cache_impl") or "").strip()
        pl_s = (r.get("prompt_len") or "").strip()
        nt_s = (r.get("new_tokens") or "").strip()
        tok_s = (r.get("end_to_end_tok_s") or "").strip()
        if not (cache and pl_s and nt_s and tok_s):
            continue
        key = (cache, _to_int(pl_s), _to_int(nt_s))
        seen.setdefault(key, set()).add((r.get("variant") or "").strip())
    good = [k for k, vs in seen.items() if ("baseline" in vs and "gqa" in vs)]
    return sorted(good, key=lambda k: (k[0], k[1], k[2]))


def _available_prefill(rows: list[dict[str, str]]) -> list[tuple[int, int]]:
    """
    Return (prompt_len, new_tokens) where BOTH baseline and gqa exist and prefill_ms is present.
    """
    seen: dict[tuple[int, int], set[str]] = {}
    for r in rows:
        if r.get("bench") != "prefill_decode":
            continue
        pl_s = (r.get("prompt_len") or "").strip()
        nt_s = (r.get("new_tokens") or "").strip()
        pm_s = (r.get("prefill_ms") or "").strip()
        if not (pl_s and nt_s and pm_s):
            continue
        key = (_to_int(pl_s), _to_int(nt_s))
        seen.setdefault(key, set()).add((r.get("variant") or "").strip())
    good = [k for k, vs in seen.items() if ("baseline" in vs and "gqa" in vs)]
    return sorted(good, key=lambda k: (k[0], k[1]))

def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--md", required=True)  # not strictly required, but useful for link validation

    ap.add_argument("--headline-prompt-len", type=int, default=0, help="0=auto (pick best available)")
    ap.add_argument("--headline-new-tokens", type=int, default=0, help="0=auto (pick best available)")
    ap.add_argument("--headline-cache-impl", default="dynamic")
    args = ap.parse_args()

    readme_path = Path(args.readme)
    csv_path = Path(args.csv)
    md_path = Path(args.md)

    if not readme_path.exists():
        raise SystemExit(f"ERROR: README not found: {readme_path}")
    if not csv_path.exists():
        raise SystemExit(f"ERROR: CSV not found: {csv_path}")
    if not md_path.exists():
        raise SystemExit(f"ERROR: MD summary not found: {md_path}")

    text = readme_path.read_text()

    if START not in text or END not in text:
        raise SystemExit(
            "ERROR: README is missing benchmark markers.\n\n"
            "Add this exact block somewhere in README.md:\n\n"
            f"{START}\n{END}\n"
        )

    pre, rest = text.split(START, 1)
    section, post = rest.split(END, 1)

    rows = _load_rows(csv_path)

    req_pl = int(args.headline_prompt_len)
    req_nt = int(args.headline_new_tokens)
    req_cache = args.headline_cache_impl

    avail_gen = _available_cache_generate(rows)
    if not avail_gen:
        raise SystemExit(
            "ERROR: No usable cache_generate rows found (need baseline+gqa with end_to_end_tok_s).\n"
            "Did the run produce cache_generate logs and were they parsed correctly?"
        )

    avail_caches = sorted({c for (c, _, _) in avail_gen})
    cache = req_cache
    if cache not in avail_caches:
        # Be explicit but helpful: fall back to something that exists.
        cache = avail_caches[0]
        print(f"NOTE: Requested cache_impl={req_cache!r} not present in this run; using {cache!r} instead.")

    avail_gen_cache = [k for k in avail_gen if k[0] == cache]

    if req_pl > 0 and req_nt > 0:
        desired = (cache, req_pl, req_nt)
        if desired not in avail_gen_cache:
            combos = ", ".join([f"{pl}/{nt}" for (_, pl, nt) in avail_gen_cache]) or "<none>"
            raise SystemExit(
                "ERROR: Could not compute headline generate() median tok/s.\n"
                f"Requested cache_impl={cache} prompt_len={req_pl} new_tokens={req_nt}, but it does not exist in CSV.\n"
                f"Available for cache_impl={cache}: {combos}\n"
                "Fix: pass --headline-prompt-len/--headline-new-tokens to match your matrix, or use 0/0 for auto."
            )
        _, pl, nt = desired
    else:
        # Auto-pick: max prompt_len, then max new_tokens.
        _, pl, nt = max(avail_gen_cache, key=lambda k: (k[1], k[2]))

    avail_pre = _available_prefill(rows)
    if (pl, nt) not in avail_pre:
        combos = ", ".join([f"{p}/{n}" for (p, n) in avail_pre]) or "<none>"
        raise SystemExit(
            "ERROR: Headline (prompt_len/new_tokens) exists for cache_generate, but not for prefill_decode.\n"
            f"Headline picked: {pl}/{nt}\n"
            f"Available prefill_decode combos: {combos}\n"
            "This usually means prefill_decode logs were missing or failed to parse."
        )

    # headline: end-to-end generate() tok/s
    base_gen = []
    gqa_gen = []

    # headline: prefill ms (prefill_decode bench)
    base_prefill = []
    gqa_prefill = []

    model = None
    device = None
    dtype = None

    for r in rows:
        # Grab model/device/dtype from prefill_decode rows (cache_generate rows often omit these)
        if r.get("bench") == "prefill_decode" and r.get("variant") == "baseline":
            if r.get("model"):
                model = r["model"]
            if r.get("device"):
                device = r["device"]
            if r.get("dtype"):
                dtype = r["dtype"]

        # Parse headline end-to-end tok/s from cache_generate
        if r.get("bench") == "cache_generate":
            if (r.get("cache_impl") or "").strip() != cache:
                continue
            if not r.get("prompt_len") or not r.get("new_tokens"):
                continue
            if _to_int(r["prompt_len"]) != pl or _to_int(r["new_tokens"]) != nt:
                continue

            v = r.get("end_to_end_tok_s", "").strip()
            if not v:
                continue

            if r.get("variant") == "baseline":
                base_gen.append(_to_float(v))
            elif r.get("variant") == "gqa":
                gqa_gen.append(_to_float(v))

        # Parse headline prefill_ms from prefill_decode
        if r.get("bench") == "prefill_decode":
            if not r.get("prompt_len") or not r.get("new_tokens"):
                continue
            if _to_int(r["prompt_len"]) != pl or _to_int(r["new_tokens"]) != nt:
                continue

            v = r.get("prefill_ms", "").strip()
            if not v:
                continue

            if r.get("variant") == "baseline":
                base_prefill.append(_to_float(v))
            elif r.get("variant") == "gqa":
                gqa_prefill.append(_to_float(v))

    if not model or not device or not dtype:
        raise SystemExit(
            "ERROR: Could not infer model/device/dtype from CSV.\n"
            "Expected at least one prefill_decode baseline row with model/device/dtype filled."
        )

    if not base_gen or not gqa_gen:
        raise SystemExit(
            "ERROR: Could not compute headline generate() median tok/s.\n"
            f"Expected cache_generate rows for baseline+gqa with cache_impl={cache} prompt_len={pl} new_tokens={nt}.\n"
            f"Found baseline={len(base_gen)} values, gqa={len(gqa_gen)} values."
        )

    if not base_prefill or not gqa_prefill:
        raise SystemExit(
            "ERROR: Could not compute headline prefill median ms.\n"
            f"Expected prefill_decode rows for baseline+gqa with prompt_len={pl} new_tokens={nt}.\n"
            f"Found baseline={len(base_prefill)} values, gqa={len(gqa_prefill)} values."
        )

    base_gen_med = _med(base_gen)
    gqa_gen_med = _med(gqa_gen)
    gen_speedup = gqa_gen_med / base_gen_med if base_gen_med > 0 else float("inf")

    base_prefill_med = _med(base_prefill)
    gqa_prefill_med = _med(gqa_prefill)
    prefill_speedup = base_prefill_med / gqa_prefill_med if gqa_prefill_med > 0 else float("inf")

    sha, dirty = _git_meta()
    git_str = sha + ("*" if dirty else "")

    # Link to the per-run markdown summary
    run_link = f"[{args.run_id}](results/summary/{args.run_id}.md)"

    headline = f"{pl}/{nt}/{cache}"

    row_line = (
        f"| {run_link} | {model} | {device}/{dtype} | {headline} | "
        f"{base_gen_med:.2f} | {gqa_gen_med:.2f} | {gen_speedup:.2f}x | "
        f"{base_prefill_med:.0f} | {gqa_prefill_med:.0f} | {prefill_speedup:.2f}x | "
        f"{git_str} |"
    )

    header = (
        "| run | model | device | headline(pl/nt/cache) | gen tok/s (base) | gen tok/s (gqa) | gen speedup | "
        "prefill ms (base) | prefill ms (gqa) | prefill speedup | git |\n"
    )
    align = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n"

    sec_lines = [ln for ln in section.strip("\n").splitlines() if ln.strip()]

    # If empty OR legacy content (like a pasted summary), overwrite with a clean table.
    if (not sec_lines) or (not sec_lines[0].startswith("| run | model | device | headline")):
        if sec_lines:
            print("NOTE: Benchmarks section was not a table; overwriting it with a fresh benchmarks table.")
        new_lines = [header.rstrip("\n"), align.rstrip("\n"), row_line]
    else:
        # Replace existing row if run already present, else insert after header+align.
        body = sec_lines[:]
        replaced = False
        for i, ln in enumerate(body):
            if f"[{args.run_id}]" in ln:
                body[i] = row_line
                replaced = True
                break

        if not replaced:
            insert_at = 2 if len(body) >= 2 else len(body)
            body.insert(insert_at, row_line)

        new_lines = body

    new_section = "\n".join(new_lines) + "\n"

    new_text = pre + START + "\n" + new_section + END + post
    readme_path.write_text(new_text)
    print(f"Updated: {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())