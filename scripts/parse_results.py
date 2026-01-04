#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import csv
import re
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

START_MARKER = "<!-- BENCHMARKS:START -->"
END_MARKER = "<!-- BENCHMARKS:END -->"

RE_PREFILL = re.compile(r"^prefill:\s+([0-9.]+)\s+ms")
RE_DECODE = re.compile(r"^decode\s+:\s+([0-9.]+)\s+s\s+for\s+(\d+)\s+tokens\s+=>\s+([0-9.]+)\s+tok/s")
RE_TOTAL = re.compile(r"^total:\s+([0-9.]+)s\s+=>\s+([0-9.]+)\s+tok/s")
RE_CACHE = re.compile(r"^cache_impl=(\w+)\s+prompt_len=(\d+)\s+new_tokens=(\d+)")
RE_ATTN = re.compile(r"^attn=(\S+)")
RE_MODEL = re.compile(r"^Loading:\s+(.+)$")
RE_DEV = re.compile(r"^device=(\S+)\s+dtype=(\S+)\s+attn=(\S+)")
RE_PROGRESS = re.compile(r"avg_step=([0-9.]+)ms")
RE_WORST = re.compile(r"^\[debug\]\s+worst_step=([0-9.]+)ms\s+at token=(\d+)")
RE_PROMPT = re.compile(r"^prompt_len=(\d+)\s+new_tokens=(\d+)$")

RE_FILE_PREFILL = re.compile(r"^torch_prefill_decode__([a-zA-Z0-9]+)__pl(\d+)__nt(\d+)__run(\d+)\.txt$")
RE_FILE_CACHE = re.compile(r"^torch_cache_generate__([a-zA-Z0-9]+)__cache([a-zA-Z0-9]+)__pl(\d+)__nt(\d+)__run(\d+)\.txt$")
RE_TIMESTAMP_DIR = re.compile(r"^\d{8}_\d{6}$")


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)

def _display_path(p: Path) -> str:
    """
    Render paths nicely for markdown:
    - if under repo root, show repo-relative (no /Users/...)
    - otherwise, fall back to absolute
    """
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)

def _median(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return median(xs) if xs else None


def _fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return ""
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if abs(x) < 0.0000001:
            x = 0.0
        return f"{x:.{nd}f}"
    return str(x)


def _speedup_time(baseline_ms: float | None, alt_ms: float | None) -> float | None:
    # smaller is better
    if baseline_ms is None or alt_ms is None or alt_ms == 0:
        return None
    return baseline_ms / alt_ms


def _speedup_rate(baseline: float | None, alt: float | None) -> float | None:
    # larger is better
    if baseline is None or alt is None or baseline == 0:
        return None
    return alt / baseline


def _read_kv_file(path: Path) -> dict[str, str]:
    """
    Read simple "key=value" OR "key: value" lines. Ignores empty lines and markdown headings.
    """
    if not path.exists():
        return {}
    meta: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue
        meta[k.strip()] = v.strip()
    return meta


def _select_latest_run_dir(raw_dir: Path) -> Path:
    if not raw_dir.exists():
        raise SystemExit(f"Input directory does not exist: {raw_dir}")

    candidates = [p for p in raw_dir.iterdir() if p.is_dir() and RE_TIMESTAMP_DIR.match(p.name)]
    if not candidates:
        raise SystemExit(f"No timestamped run directories found under: {raw_dir}")

    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def parse_file(path: Path) -> dict[str, object] | None:
    row: dict[str, object] = {
        "file": str(path),
        "bench": None,
        "variant": None,
        "run": None,
        "model": None,
        "device": None,
        "dtype": None,
        "attn_impl": None,
        "prompt_len": None,
        "new_tokens": None,
        "cache_impl": None,
        "prefill_ms": None,
        "decode_s": None,
        "decode_tok_s": None,
        "avg_step_ms_last": None,
        "worst_step_ms": None,
        "worst_step_token": None,
        "total_s": None,
        "end_to_end_tok_s": None,
    }

    txt = path.read_text(errors="replace").splitlines()
    if not txt:
        return None

    name = path.name

    # Infer from filename if it matches run_matrix.sh convention.
    m = RE_FILE_PREFILL.match(name)
    if m:
        row["bench"] = "prefill_decode"
        row["variant"] = m.group(1)
        row["prompt_len"] = int(m.group(2))
        row["new_tokens"] = int(m.group(3))
        row["run"] = int(m.group(4))
    else:
        m = RE_FILE_CACHE.match(name)
        if m:
            row["bench"] = "cache_generate"
            row["variant"] = m.group(1)
            row["cache_impl"] = m.group(2)
            row["prompt_len"] = int(m.group(3))
            row["new_tokens"] = int(m.group(4))
            row["run"] = int(m.group(5))
        else:
            # Fallback to prefix-based detection.
            if name.startswith("torch_prefill_decode__"):
                row["bench"] = "prefill_decode"
            elif name.startswith("torch_cache_generate__"):
                row["bench"] = "cache_generate"
            else:
                return None

            if "__gqa__" in name:
                row["variant"] = "gqa"
            elif "__baseline__" in name:
                row["variant"] = "baseline"

    last_avg: float | None = None

    for line in txt:
        m = RE_MODEL.match(line)
        if m:
            row["model"] = m.group(1).strip()
            continue

        m = RE_DEV.match(line)
        if m:
            row["device"] = m.group(1)
            row["dtype"] = m.group(2)
            continue

        m = RE_ATTN.match(line)
        if m:
            row["attn_impl"] = m.group(1)
            continue

        m = RE_PROMPT.match(line)
        if m:
            row["prompt_len"] = int(m.group(1))
            row["new_tokens"] = int(m.group(2))
            continue

        m = RE_CACHE.match(line)
        if m:
            row["cache_impl"] = m.group(1)
            row["prompt_len"] = int(m.group(2))
            row["new_tokens"] = int(m.group(3))
            continue

        m = RE_PREFILL.match(line)
        if m:
            row["prefill_ms"] = float(m.group(1))
            continue

        m = RE_DECODE.match(line)
        if m:
            row["decode_s"] = float(m.group(1))
            row["new_tokens"] = int(m.group(2))
            row["decode_tok_s"] = float(m.group(3))
            continue

        m = RE_TOTAL.match(line)
        if m:
            row["total_s"] = float(m.group(1))
            row["end_to_end_tok_s"] = float(m.group(2))
            continue

        m = RE_PROGRESS.search(line)
        if m:
            last_avg = float(m.group(1))

        m = RE_WORST.match(line)
        if m:
            row["worst_step_ms"] = float(m.group(1))
            row["worst_step_token"] = int(m.group(2))

    row["avg_step_ms_last"] = last_avg
    return row


def _update_markdown_between_markers(md_path: Path, replacement: str) -> None:
    if not md_path.exists():
        raise SystemExit(f"Markdown file not found: {md_path}")

    text = md_path.read_text(errors="replace")
    if START_MARKER not in text or END_MARKER not in text:
        raise SystemExit(
            f"Missing benchmark markers in: {md_path}\n"
            f"Add these lines where you want the table inserted:\n"
            f"{START_MARKER}\n\n{END_MARKER}\n"
        )

    if text.index(START_MARKER) > text.index(END_MARKER):
        raise SystemExit(f"Marker order is wrong in {md_path}: START occurs after END")

    pattern = re.compile(
        re.escape(START_MARKER) + r"(.*?)" + re.escape(END_MARKER),
        flags=re.DOTALL,
    )

    def repl(match: re.Match[str]) -> str:
        return START_MARKER + "\n\n" + replacement.rstrip() + "\n\n" + END_MARKER

    new_text, n = pattern.subn(repl, text, count=1)
    if n != 1:
        raise SystemExit(f"Failed to update markers in {md_path} (expected 1 match, got {n})")

    md_path.write_text(new_text)


def _make_summary_markdown(
    *,
    run_id: str,
    input_dir: Path,
    rows: list[dict[str, object]],
    meta: dict[str, str],
) -> str:
    def first(field: str) -> str:
        for r in rows:
            v = r.get(field)
            if v:
                return str(v)
        return ""

    model = meta.get("model") or first("model")
    device = meta.get("device") or first("device")
    dtype = meta.get("dtype") or first("dtype")
    attn = meta.get("attn") or ""

    pre_groups: dict[tuple, dict[str, list[dict[str, object]]]] = {}
    cache_groups: dict[tuple, dict[str, list[dict[str, object]]]] = {}

    for r in rows:
        bench = r.get("bench")
        variant = str(r.get("variant") or "")
        if bench == "prefill_decode":
            key = (r.get("prompt_len"), r.get("new_tokens"))
            pre_groups.setdefault(key, {}).setdefault(variant, []).append(r)
        elif bench == "cache_generate":
            key = (r.get("cache_impl"), r.get("prompt_len"), r.get("new_tokens"))
            cache_groups.setdefault(key, {}).setdefault(variant, []).append(r)

    lines: list[str] = []
    lines.append(f"**Run:** `{run_id}`  ")
    if model:
        lines.append(f"**Model:** `{model}`  ")
    if device or dtype:
        lines.append(f"**Device:** `{device}`  **dtype:** `{dtype}`  ")
    if attn:
        lines.append(f"**Requested attn:** `{attn}`  ")

    env_bits = []
    for k in ("platform", "python", "torch", "transformers"):
        v = meta.get(k)
        if v:
            env_bits.append(f"{k}={v}")
    if env_bits:
        lines.append(f"**Env:** {' | '.join(env_bits)}  ")

    if meta.get("git_sha"):
        gs = meta.get("git_sha", "")
        st = meta.get("git_status", "")
        lines.append(f"**Git:** `{gs}`{f' ({st})' if st else ''}  ")

    if meta.get("runs") or meta.get("warmup"):
        lines.append(f"**Runs:** {meta.get('runs','')}  **Warmup:** {meta.get('warmup','')}  ")
    if meta.get("notes"):
        lines.append(f"**Notes:** {meta.get('notes')}  ")
    if meta.get("benches"):
        lines.append(f"**Benches:** `{meta.get('benches')}`  ")

    lines.append(f"**Raw logs:** `{_display_path(input_dir)}`\n")

    lines.append("### Prefill + decode (median over runs)\n")
    lines.append(
        "| prompt_len | new_tokens | n_baseline | prefill_ms_base | decode_tok/s_base | worst_step_ms_base | "
        "n_gqa | prefill_ms_gqa | decode_tok/s_gqa | worst_step_ms_gqa | "
        "prefill_speedup | decode_speedup |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for key in sorted(pre_groups.keys(), key=lambda k: (k[0] is None, k[0], k[1] is None, k[1])):
        pl, nt = key
        b = pre_groups[key].get("baseline", [])
        g = pre_groups[key].get("gqa", [])

        b_prefill = _median([x.get("prefill_ms") for x in b])  # type: ignore[arg-type]
        b_decode = _median([x.get("decode_tok_s") for x in b])  # type: ignore[arg-type]
        b_worst = _median([x.get("worst_step_ms") for x in b])  # type: ignore[arg-type]

        g_prefill = _median([x.get("prefill_ms") for x in g])  # type: ignore[arg-type]
        g_decode = _median([x.get("decode_tok_s") for x in g])  # type: ignore[arg-type]
        g_worst = _median([x.get("worst_step_ms") for x in g])  # type: ignore[arg-type]

        pre_su = _speedup_time(b_prefill, g_prefill)
        dec_su = _speedup_rate(b_decode, g_decode)

        lines.append(
            f"| {_fmt(pl,0)} | {_fmt(nt,0)} | {len(b)} | {_fmt(b_prefill)} | {_fmt(b_decode)} | {_fmt(b_worst)} | "
            f"{len(g)} | {_fmt(g_prefill)} | {_fmt(g_decode)} | {_fmt(g_worst)} | "
            f"{_fmt(pre_su)}x | {_fmt(dec_su)}x |"
        )

    lines.append("\n### End-to-end generate() (median over runs)\n")
    lines.append(
        "| cache_impl | prompt_len | new_tokens | n_baseline | tok/s_base | total_s_base | "
        "n_gqa | tok/s_gqa | total_s_gqa | speedup |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for key in sorted(cache_groups.keys(), key=str):
        cache, pl, nt = key
        b = cache_groups[key].get("baseline", [])
        g = cache_groups[key].get("gqa", [])

        b_tok = _median([x.get("end_to_end_tok_s") for x in b])  # type: ignore[arg-type]
        b_tot = _median([x.get("total_s") for x in b])  # type: ignore[arg-type]
        g_tok = _median([x.get("end_to_end_tok_s") for x in g])  # type: ignore[arg-type]
        g_tot = _median([x.get("total_s") for x in g])  # type: ignore[arg-type]

        su = _speedup_rate(b_tok, g_tok)

        lines.append(
            f"| {cache or ''} | {_fmt(pl,0)} | {_fmt(nt,0)} | {len(b)} | {_fmt(b_tok)} | {_fmt(b_tot)} | "
            f"{len(g)} | {_fmt(g_tok)} | {_fmt(g_tot)} | {_fmt(su)}x |"
        )

    lines.append("\n_Generated by `scripts/parse_results.py`._")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/raw", help="results/raw or results/raw/<timestamp>")
    ap.add_argument("--latest", action="store_true", help="If --input is results/raw, pick the latest timestamp dir.")
    ap.add_argument("--csv-out", default="results/summary/summary.csv")
    ap.add_argument("--md-out", default="results/summary/summary.md")
    ap.add_argument("--update-md", nargs="*", default=[], help="Markdown files to update between BENCHMARKS markers.")
    args = ap.parse_args()

    input_dir = _resolve(args.input)

    if args.latest:
        input_dir = _select_latest_run_dir(input_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    run_id = input_dir.name if RE_TIMESTAMP_DIR.match(input_dir.name) else "summary"

    out_csv = _resolve(args.csv_out)
    out_md = _resolve(args.md_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for p in sorted(input_dir.rglob("*.txt")):
        r = parse_file(p)
        if r:
            rows.append(r)

    if not rows:
        raise SystemExit(f"No parsable bench logs found under: {input_dir}")

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    meta = _read_kv_file(input_dir / "meta.txt")

    md_text = _make_summary_markdown(run_id=run_id, input_dir=input_dir, rows=rows, meta=meta)
    out_md.write_text(md_text)

    # Per-run outputs for history
    if run_id != "summary":
        run_csv = out_csv.parent / f"{run_id}.csv"
        run_md = out_md.parent / f"{run_id}.md"
        run_csv.write_text(out_csv.read_text(errors="replace"))
        run_md.write_text(md_text)

    for md in args.update_md:
        _update_markdown_between_markers(_resolve(md), md_text)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    if run_id != "summary":
        print(f"Wrote: {run_csv}")
        print(f"Wrote: {run_md}")
    for md in args.update_md:
        print(f"Updated: {_resolve(md)}")


if __name__ == "__main__":
    main()