#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

def _csv_has_bench(csv_path: Path, bench_name: str) -> bool:
    if not csv_path.exists():
        return False
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("bench") or "").strip() == bench_name:
                return True
    return False

def _ws_list(s: str | None) -> list[str]:
    return [x for x in (s or "").strip().split() if x]

def _env_int(env: dict[str, str], key: str) -> int | None:
    v = (env.get(key) or "").strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        raise SystemExit(f"ERROR: {key} must be an int in bench.env, got: {v!r}")

def _env_int_list(env: dict[str, str], key: str, *, default: list[int]) -> list[int]:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    out: list[int] = []
    for tok in raw.split():
        try:
            out.append(int(tok))
        except ValueError:
            raise SystemExit(f"ERROR: {key} must be space-separated ints in bench.env, got token={tok!r}")
    return out

def _read_env_file(path: Path) -> dict[str, str]:
    """
    Minimal, deterministic parser for KEY=VALUE lines.
    - ignores blank lines and comments (# ...)
    - allows optional 'export ' prefix
    - strips optional surrounding single/double quotes
    This is intentionally NOT a full shell parser (less magic, less tech debt).
    """
    if not path.exists():
        raise SystemExit(
            f"ERROR: Missing env file: {path}\n"
            f"Fix: cp config/bench.env.template {path} && edit it"
        )

    out: dict[str, str] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise SystemExit(f"ERROR: Invalid line in {path} (expected KEY=VALUE): {raw}")
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
            v = v[1:-1]
        out[k] = v
    return out

def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

def _preflight_publish_config(cfg: dict[str, str], *, no_readme: bool) -> None:
    """
    Deterministic contract:
      - If we're publishing (no_readme=False), BENCHES MUST include both benches
        because README row requires both generate + prefill metrics.
      - Fail BEFORE running the matrix.
    """
    if no_readme:
        return

    benches = _ws_list(cfg.get("BENCHES")) or ["prefill_decode", "cache_generate"]
    missing = [b for b in ("prefill_decode", "cache_generate") if b not in benches]
    if missing:
        raise SystemExit(
            "ERROR: This is a publish run (README update enabled), but BENCHES is missing required benches.\n"
            f"Required: prefill_decode + cache_generate\n"
            f"Found   : {benches}\n"
            "Fix: set BENCHES=\"prefill_decode cache_generate\" in config/bench.env\n"
            "     (or run a debug run with --no-readme)\n"
        )

    # Optional but useful: validate headline values if explicitly set.
    prompt_lens = _env_int_list(cfg, "PROMPT_LENS", default=[1024, 2048, 4096])
    new_tokens  = _env_int_list(cfg, "NEW_TOKENS_LIST", default=[32, 64])
    caches      = _ws_list(cfg.get("CACHE_IMPLS")) or ["dynamic"]

    h_pl = _env_int(cfg, "HEADLINE_PROMPT_LEN")
    h_nt = _env_int(cfg, "HEADLINE_NEW_TOKENS")
    h_ca = (cfg.get("HEADLINE_CACHE_IMPL") or "dynamic").strip()

    if h_pl and h_pl not in prompt_lens:
        raise SystemExit(f"ERROR: HEADLINE_PROMPT_LEN={h_pl} not in PROMPT_LENS={prompt_lens}")
    if h_nt and h_nt not in new_tokens:
        raise SystemExit(f"ERROR: HEADLINE_NEW_TOKENS={h_nt} not in NEW_TOKENS_LIST={new_tokens}")
    if h_ca and h_ca not in caches:
        raise SystemExit(f"ERROR: HEADLINE_CACHE_IMPL={h_ca!r} not in CACHE_IMPLS={caches}")

def main() -> int:
    ap = argparse.ArgumentParser(description="Run benchmark matrix + parse + update README.")
    ap.add_argument("--run-id", default=None, help="Timestamp like 20260102_215515 (default: now).")
    ap.add_argument(
        "--env-file",
        default="config/bench.env",
        help="Path to bench.env (default: config/bench.env)",
    )
    # Headline row selection (what goes into README as the single-row summary)
    ap.add_argument("--headline-prompt-len", type=int, default=0, help="0=auto (pick best available from this run)")
    ap.add_argument("--headline-new-tokens", type=int, default=0, help="0=auto (pick best available from this run)")

    ap.add_argument("--headline-cache-impl", default="", help="If empty, use bench.env or default=dynamic")

    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--no-readme", action="store_true", help="Skip updating README.md.")
    args = ap.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path("results/raw") / run_id
    summary_dir = Path("results/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    csv_out = summary_dir / f"{run_id}.csv"
    md_out = summary_dir / f"{run_id}.md"

    # Load bench.env first, then allow existing process env to override if desired.
    # (If you prefer the opposite precedence, swap the update order.)
    env = os.environ.copy()
    env_path = (ROOT / args.env_file) if not Path(args.env_file).is_absolute() else Path(args.env_file)
    cfg = _read_env_file(env_path)
    env.update(cfg)

    # Fail fast if this run is expected to publish.
    _preflight_publish_config(cfg, no_readme=args.no_readme)

    env["TS"] = run_id
    env["OUT_DIR"] = str(out_dir)
    # Headline defaults from bench.env, unless CLI explicitly set.
    if args.headline_prompt_len == 0:
        pl = _env_int(cfg, "HEADLINE_PROMPT_LEN")
        if pl is not None:
            args.headline_prompt_len = pl
    if args.headline_new_tokens == 0:
        nt = _env_int(cfg, "HEADLINE_NEW_TOKENS")
        if nt is not None:
            args.headline_new_tokens = nt
    if not args.headline_cache_impl.strip():
        args.headline_cache_impl = (cfg.get("HEADLINE_CACHE_IMPL") or "dynamic").strip()
    if not args.headline_cache_impl:
        args.headline_cache_impl = "dynamic"
    
    env["PYTHON"] = sys.executable
    print(f"[stage 1/3] running benchmark matrix (run_id={run_id})", flush=True)
    # 1) Run the full matrix
    _run(["bash", "scripts/run_matrix.sh"], env=env)

    print("[stage 2/3] parsing logs", flush=True)
    # 2) Parse into timestamped artifacts
    _run(
        [
            sys.executable,
            "scripts/parse_results.py",
            "--input",
            str(out_dir),
        ]
    )

    # parse_results.py should have created these per-run artifacts:
    if not csv_out.exists() or not md_out.exists():
        raise SystemExit(
            f"ERROR: Expected {csv_out} and {md_out} to exist after parsing.\n"
            "Check scripts/parse_results.py output paths."
        )

    if args.no_readme:
        print("[stage 3/3] skipping README update (--no-readme)", flush=True)
    else:
        # Sanity: publish run MUST have both benches in CSV.
        if not _csv_has_bench(csv_out, "cache_generate") or not _csv_has_bench(csv_out, "prefill_decode"):
            raise SystemExit(
                "ERROR: Publish run expected both benches in CSV, but at least one is missing.\n"
                f"CSV: {csv_out}\n"
                "Check raw logs for failures.\n"
            )

        print("[stage 3/3] updating README", flush=True)
        _run([
            sys.executable,
            "scripts/update_readme_benchmarks.py",
            "--readme", args.readme,
            "--run-id", run_id,
            "--csv", str(csv_out),
            "--md", str(md_out),
            "--headline-prompt-len", str(args.headline_prompt_len),
            "--headline-new-tokens", str(args.headline_new_tokens),
            "--headline-cache-impl", args.headline_cache_impl,
        ])

    print("")
    print("Done.")
    print(f"Raw logs : {out_dir}")
    print(f"Summary  : {md_out}")
    print(f"CSV      : {csv_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())