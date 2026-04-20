#!/usr/bin/env python3
"""
Fast single-run analyzer. Works on run.log, PBS .OU files, or a run directory.

Usage:
    python3 parse_run.py results/bench_2node/260420_105317/run.log
    python3 parse_run.py results/bench_2node/260420_105317/
    python3 parse_run.py /path/to/bench_2node.OU12345
    python3 parse_run.py results/bench_2node/*/run.log   # compare multiple
"""

import re
import sys
from pathlib import Path
from statistics import mean, stdev

BASELINE_STEP   = 1.82   # 1-node step time (s)
BASELINE_FWD    = 0.54
BASELINE_BWD    = 0.80
WARMUP_STEPS    = 2      # steps to skip at start

_STEP_RE = re.compile(
    r"\[Train\].*?Step\s+(\d+)/\d+.*?"
    r"forward_time:\s*([\d.]+)s.*?"
    r"backward_time:\s*([\d.]+)s.*?"
    r"optim_time:\s*([\d.]+)s.*?"
    r"time:\s*([\d.]+)s"
)
_RANK_RE  = re.compile(r"\[RANK\s+(\d+)\]")
_NODES_RE = re.compile(r"Nodes:\s*(\d+)\s+GPUs/node:\s*(\d+)")
_ALLOC_RE = re.compile(r"Allocated nodes:\s*(.+)")
_CFG_RE   = re.compile(
    r"Namespace\(.*?shard_strategy='(\w+)'.*?fsdp_group_size=(\d+).*?hybrid_shard=(True|False)"
)


def find_log(path: Path) -> Path:
    if path.is_file():
        return path
    for candidate in [path / "run.log", *path.glob("*/run.log")]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No run.log found under {path}")


def parse(log_path: Path, warmup: int = WARMUP_STEPS) -> dict:
    text = log_path.read_text(errors="replace")

    # Config
    nodes, gpus_per_node = 0, 0
    m = _NODES_RE.search(text)
    if m:
        nodes, gpus_per_node = int(m.group(1)), int(m.group(2))

    allocated = ""
    m = _ALLOC_RE.search(text)
    if m:
        allocated = m.group(1).strip()

    shard, group_size, hybrid = "?", "?", "?"
    m = _CFG_RE.search(text)
    if m:
        shard      = m.group(1)
        group_size = m.group(2)
        hybrid     = m.group(3)

    # Per-step data: {step: {rank: {fwd, bwd, optim, total}}}
    steps: dict[int, dict[int, dict]] = {}
    for line in text.splitlines():
        sm = _STEP_RE.search(line)
        if not sm:
            continue
        rm = _RANK_RE.search(line)
        rank = int(rm.group(1)) if rm else 0
        step = int(sm.group(1))
        steps.setdefault(step, {})[rank] = {
            "fwd":   float(sm.group(2)),
            "bwd":   float(sm.group(3)),
            "optim": float(sm.group(4)),
            "total": float(sm.group(5)),
        }

    if not steps:
        raise ValueError("No step lines found")

    # Wall-clock per step = max across ranks
    rows = []
    for step in sorted(steps):
        ranks = steps[step]
        rows.append({
            "step":  step,
            "fwd":   max(r["fwd"]   for r in ranks.values()),
            "bwd":   max(r["bwd"]   for r in ranks.values()),
            "optim": max(r["optim"] for r in ranks.values()),
            "total": max(r["total"] for r in ranks.values()),
            "nranks": len(ranks),
        })

    steady = [r for r in rows if r["step"] > warmup]
    if not steady:
        steady = rows

    def stat(key):
        vals = [r[key] for r in steady]
        m = mean(vals)
        s = stdev(vals) if len(vals) > 1 else 0.0
        return m, s, 100 * s / m if m else 0

    fwd_m,   fwd_s,   fwd_cv   = stat("fwd")
    bwd_m,   bwd_s,   bwd_cv   = stat("bwd")
    total_m, total_s, total_cv = stat("total")

    eff = 100 * BASELINE_STEP / total_m if total_m else 0

    return {
        "log":         log_path,
        "nodes":       nodes,
        "gpus_per_node": gpus_per_node,
        "allocated":   allocated,
        "shard":       shard,
        "group_size":  group_size,
        "hybrid":      hybrid,
        "rows":        rows,
        "steady":      steady,
        "warmup":      warmup,
        "fwd":   (fwd_m,   fwd_s,   fwd_cv),
        "bwd":   (bwd_m,   bwd_s,   bwd_cv),
        "total": (total_m, total_s, total_cv),
        "efficiency":  eff,
        "bwd_overhead": bwd_m - BASELINE_BWD,
    }


def print_report(r: dict, show_steps: bool = True):
    print("=" * 70)
    print(f"Log   : {r['log']}")
    print(f"Nodes : {r['nodes']} × {r['gpus_per_node']} GPUs  ({r['allocated']})")
    print(f"Config: shard={r['shard']}  fsdp_group_size={r['group_size']}  hybrid={r['hybrid']}")
    print(f"Steps : {len(r['rows'])} total, {len(r['steady'])} steady-state (skipped first {r['warmup']})")

    if show_steps:
        print()
        print(f"  {'Step':>4}  {'fwd':>7}  {'bwd':>7}  {'total':>7}  {'ranks':>5}")
        print(f"  {'----':>4}  {'-------':>7}  {'-------':>7}  {'-------':>7}  {'-----':>5}")
        for row in r["rows"]:
            marker = "" if row["step"] > r["warmup"] else " *"
            print(f"  {row['step']:>4}  {row['fwd']:>7.2f}  {row['bwd']:>7.2f}  {row['total']:>7.2f}  {row['nranks']:>5}{marker}")
        print("  (* = warmup, excluded from stats)")

    fwd_m,   fwd_s,   fwd_cv   = r["fwd"]
    bwd_m,   bwd_s,   bwd_cv   = r["bwd"]
    total_m, total_s, total_cv = r["total"]

    print()
    print(f"  {'Phase':<8}  {'mean':>7}  {'std':>7}  {'CV%':>6}")
    print(f"  {'--------':<8}  {'-------':>7}  {'-------':>7}  {'------':>6}")
    print(f"  {'forward':<8}  {fwd_m:>7.2f}s  {fwd_s:>7.2f}s  {fwd_cv:>5.1f}%")
    print(f"  {'backward':<8}  {bwd_m:>7.2f}s  {bwd_s:>7.2f}s  {bwd_cv:>5.1f}%")
    print(f"  {'total':<8}  {total_m:>7.2f}s  {total_s:>7.2f}s  {total_cv:>5.1f}%")

    print()
    print(f"  Baseline (1-node): fwd={BASELINE_FWD}s  bwd={BASELINE_BWD}s  total={BASELINE_STEP}s")
    print(f"  Scaling efficiency : {r['efficiency']:>5.1f}%  (ideal=100%,  >80% good,  <50% poor)")
    print(f"  IB backward overhead: {r['bwd_overhead']:>+.2f}s vs 1-node backward")
    print()


def print_comparison(results: list[dict]):
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Label':<40}  {'fwd':>6}  {'bwd':>6}  {'total':>6}  {'eff%':>6}  {'bwd_oh':>7}")
    print(f"  {'-'*40}  {'------':>6}  {'------':>6}  {'------':>6}  {'------':>6}  {'-------':>7}")
    print(f"  {'1-node baseline':<40}  {BASELINE_FWD:>6.2f}  {BASELINE_BWD:>6.2f}  {BASELINE_STEP:>6.2f}  {'100.0':>6}  {'  0.00':>7}")
    for r in results:
        label = f"{r['shard']} gs={r['group_size']} hybrid={r['hybrid']}"
        fwd_m  = r["fwd"][0]
        bwd_m  = r["bwd"][0]
        total_m = r["total"][0]
        print(
            f"  {label:<40}  {fwd_m:>6.2f}  {bwd_m:>6.2f}  {total_m:>6.2f}"
            f"  {r['efficiency']:>6.1f}  {r['bwd_overhead']:>+7.2f}"
        )
    print()


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        sys.exit(0)

    results = []
    for p in paths:
        path = Path(p)
        try:
            log = find_log(path)
            r = parse(log)
            results.append(r)
            print_report(r, show_steps=(len(paths) == 1))
        except Exception as e:
            print(f"ERROR: {p}: {e}", file=sys.stderr)

    if len(results) > 1:
        print_comparison(results)


if __name__ == "__main__":
    main()
