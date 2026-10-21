#!/usr/bin/env python3
"""
average_alpha.py

Average the alpha values produced by alphaTest.py (or measure_alpha.py) and
report one average per model pair.

Files it understands:
  alpha_<pair>_<set>_<timestamp>.csv   per-sample output of alphaTest.py
                                       (the alpha_summary_*.csv files are skipped,
                                       since the per-sample file has everything)
  alpha_results/<run>/summary.csv      summary output of measure_alpha.py

Two averages are reported for each model pair and temperature:
  alpha_w  token-weighted: total beta / total tokens. This is the per-token
           alpha that belongs in the speedup formula.
  alpha_m  simple mean of the per-task alphas (every task counts equally).

Usage:
    python average_alpha.py                        # search the current folder
    python average_alpha.py alpha_llama_builtin_20261020-101500.csv
    python average_alpha.py some/folder            # search another folder
    python average_alpha.py --temperature 0        # greedy only
    python average_alpha.py --out alpha_averages.csv
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def find_csvs(paths):
    found = []
    for p in paths:
        if os.path.isdir(p):
            found += glob.glob(os.path.join(p, "**", "alpha_*.csv"), recursive=True)
            found += glob.glob(os.path.join(p, "**", "summary.csv"), recursive=True)
        elif os.path.isfile(p):
            found.append(p)
        else:
            print(f"Skipping {p}: not found")
    # The alphaTest.py summary files repeat the per-sample data, so skip them.
    found = [f for f in found if not os.path.basename(f).startswith("alpha_summary_")]
    return sorted(set(found))


def short(name):
    return name.split("/")[-1]


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_rows(path):
    """Return (model_label, list of (task, temperature, beta_sum, tokens))."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        rows = list(reader)

    out = []
    if {"draft_model", "target_model", "beta_sum", "generated_tokens"} <= cols:
        # alphaTest.py per-sample file: one row per sample.
        labels = set()
        for r in rows:
            tokens = to_float(r["generated_tokens"])
            beta = to_float(r["beta_sum"])
            if not tokens or beta is None:
                continue
            labels.add(f"{short(r['draft_model'])} -> {short(r['target_model'])}")
            out.append((r["task"], to_float(r["temperature"]), beta, tokens))
        label = " / ".join(sorted(labels)) or os.path.basename(path)
        return label, out

    if {"category", "alpha_analytic", "tokens"} <= cols:
        # measure_alpha.py summary file: one row per category, plus ALL rows.
        run_dir = os.path.dirname(os.path.abspath(path))
        label = os.path.basename(run_dir)
        cfg = os.path.join(run_dir, "run_config.json")
        if os.path.exists(cfg):
            with open(cfg) as f:
                c = json.load(f)
            label = f"{short(c['draft'])} -> {short(c['target'])}"
        for r in rows:
            if r["category"] == "ALL":
                continue
            tokens = to_float(r["tokens"])
            alpha = to_float(r["alpha_analytic"])
            if tokens and alpha is not None:
                out.append((r["category"], to_float(r["temperature"]), alpha * tokens, tokens))
        return label, out

    return None, []


def main():
    ap = argparse.ArgumentParser(description="Average alpha per model pair.")
    ap.add_argument("paths", nargs="*", default=["."],
                    help="CSV files or folders to search (default: current folder)")
    ap.add_argument("--temperature", type=float, help="Only use rows at this temperature.")
    ap.add_argument("--out", help="Also write the results to this CSV file.")
    args = ap.parse_args()

    csvs = find_csvs(args.paths)
    if not csvs:
        print("No alpha CSV files found. Looked for alpha_*.csv (from alphaTest.py) and")
        print("summary.csv (from measure_alpha.py) in:", ", ".join(os.path.abspath(p) for p in args.paths))
        print("Run this from the folder alphaTest.py ran in, or pass the CSV path.")
        return

    # data[model][temperature][task] = [beta_sum, tokens, samples]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0])))
    for path in csvs:
        label, rows = read_rows(path)
        if not rows:
            print(f"Skipping {path}: no alpha columns or no finished rows")
            continue
        print(f"Read {len(rows):>5} rows from {path}")
        for task, temp, beta, tokens in rows:
            if args.temperature is not None and temp != args.temperature:
                continue
            cell = data[label][temp][task]
            cell[0] += beta
            cell[1] += tokens
            cell[2] += 1

    results = []
    for model in sorted(data):
        temps = data[model]
        groups = [(f"{t:g}", {t: temps[t]}) for t in sorted(temps)]
        if len(groups) > 1:
            groups.append(("all", temps))
        for temp_label, sub in groups:
            beta = sum(c[0] for tasks in sub.values() for c in tasks.values())
            tokens = sum(c[1] for tasks in sub.values() for c in tasks.values())
            samples = sum(c[2] for tasks in sub.values() for c in tasks.values())
            task_alphas = [c[0] / c[1] for tasks in sub.values() for c in tasks.values()]
            results.append({
                "model": model,
                "temperature": temp_label,
                "samples": samples,
                "tokens": int(tokens),
                "alpha_weighted": beta / tokens,
                "alpha_task_mean": sum(task_alphas) / len(task_alphas),
            })

    if not results:
        print("No rows matched.")
        return

    width = max(len(r["model"]) for r in results) + 2
    hdr = f"{'model':<{width}}{'temp':>6}{'samples':>9}{'tokens':>9}{'alpha_w':>10}{'alpha_m':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))
    last = None
    for r in results:
        name = r["model"] if r["model"] != last else ""
        last = r["model"]
        print(f"{name:<{width}}{r['temperature']:>6}{r['samples']:>9}{r['tokens']:>9}"
              f"{r['alpha_weighted']:>10.4f}{r['alpha_task_mean']:>10.4f}")
    print("\nalpha_w = token-weighted alpha (use this in the speedup formula)")
    print("alpha_m = simple mean over tasks (each task counts equally)")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            for r in results:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
