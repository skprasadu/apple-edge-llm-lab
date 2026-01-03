#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run benchmark matrix + parse + update README.")
    ap.add_argument("--run-id", default=None, help="Timestamp like 20260102_215515 (default: now).")

    # Headline row selection (what goes into README as the single-row summary)
    ap.add_argument("--headline-prompt-len", type=int, default=0, help="0=auto (pick best available from this run)")
    ap.add_argument("--headline-new-tokens", type=int, default=0, help="0=auto (pick best available from this run)")

    ap.add_argument("--headline-cache-impl", default="dynamic")

    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--no-readme", action="store_true", help="Skip updating README.md.")
    args = ap.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path("results/raw") / run_id
    summary_dir = Path("results/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    csv_out = summary_dir / f"{run_id}.csv"
    md_out = summary_dir / f"{run_id}.md"

    env = os.environ.copy()
    env["TS"] = run_id
    env["OUT_DIR"] = str(out_dir)

    # 1) Run the full matrix
    _run(["bash", "scripts/run_matrix.sh"], env=env)

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

    # 4) Update root README with a single row for this run
    if not args.no_readme:
        _run(
            [
                sys.executable,
                "scripts/update_readme_benchmarks.py",
                "--readme",
                args.readme,
                "--run-id",
                run_id,
                "--csv",
                str(csv_out),
                "--md",
                str(md_out),
                "--headline-prompt-len",
                str(args.headline_prompt_len),
                "--headline-new-tokens",
                str(args.headline_new_tokens),
                "--headline-cache-impl",
                args.headline_cache_impl,
            ]
        )

    print("")
    print("Done.")
    print(f"Raw logs : {out_dir}")
    print(f"Summary  : {md_out}")
    print(f"CSV      : {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())