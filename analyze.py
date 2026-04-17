#!/usr/bin/env python3
"""
Analyze benchmark run.log files to measure throughput and identify bottlenecks.

Usage:
    # Compare multiple runs:
    python3 analyze.py results/bench_1node/260416_100000/run.log \
                       results/bench_2node/260416_110000/run.log \
                       results/bench_4node/260416_120000/run.log

    # With explicit labels:
    python3 analyze.py \
        --runs "1-node=results/bench_1node/260416_100000/run.log" \
               "2-node=results/bench_2node/260416_110000/run.log" \
               "4-node=results/bench_4node/260416_120000/run.log"

    # Auto-discover all run.log files under results/:
    python3 analyze.py --auto

Output:
    - Per-run statistics: mean step time, throughput (steps/sec and tokens/sec)
    - Time breakdown per step: data loading, forward, backward (incl. AllReduce), optimizer
    - Scaling efficiency table: actual speedup vs ideal linear speedup
    - Bottleneck flag: which phase limits scaling

Key metric for bottleneck detection:
    backward_time includes the NCCL AllReduce (gradient synchronization).
    If backward_time grows significantly from single-node to multi-node,
    inter-node communication bandwidth is the bottleneck.
    If data_time is high and constant, data loading (I/O) is the bottleneck.
"""

import argparse
import re
import sys
from pathlib import Path
from statistics import mean, stdev


# Regex to parse a step log line from unify_internvl2_train_r16.py:
#   [Train] (Epoch 0) Step 5/35  lr: ...  tgs: 1234  data_time: 0.45s  ...  time: 4.55s  eta: ...
_STEP_RE = re.compile(
    r"\[Train\].*?Step\s+(\d+)/(\d+).*?"
    r"tgs:\s*([\d.]+).*?"
    r"data_time:\s*([\d.]+)s.*?"
    r"prepare_time:\s*([\d.]+)s.*?"
    r"pack_time:\s*([\d.]+)s.*?"
    r"forward_time:\s*([\d.]+)s.*?"
    r"backward_time:\s*([\d.]+)s.*?"
    r"clip_time:\s*([\d.]+)s.*?"
    r"optim_time:\s*([\d.]+)s.*?"
    r"time:\s*([\d.]+)s"
)

# Regex to extract node/GPU info from the header written by _common.sh
_NODES_RE = re.compile(r"Nodes:\s*(\d+)\s+GPUs/node:\s*(\d+)\s+Total GPUs:\s*(\d+)")
_BATCH_RE = re.compile(r"global_batch=(\d+)")


def parse_log(path: Path, skip_steps: int = 5) -> dict:
    """
    Parse a run.log file and return per-step timing statistics.

    skip_steps: number of initial steps to discard (first steps are slow due to
                CUDA kernel compilation, data pipeline warmup, etc.)
    """
    steps = []
    nodes = gpus_per_node = total_gpus = global_batch = None

    with open(path) as f:
        for line in f:
            # Parse header metadata
            m = _NODES_RE.search(line)
            if m:
                nodes, gpus_per_node, total_gpus = int(m.group(1)), int(m.group(2)), int(m.group(3))
            m = _BATCH_RE.search(line)
            if m:
                global_batch = int(m.group(1))

            # Parse step timing line
            m = _STEP_RE.search(line)
            if m:
                step_idx = int(m.group(1))
                steps.append({
                    "step":         step_idx,
                    "tgs":          float(m.group(3)),
                    "data":         float(m.group(4)),
                    "prepare":      float(m.group(5)),
                    "pack":         float(m.group(6)),
                    "forward":      float(m.group(7)),
                    "backward":     float(m.group(8)),
                    "clip":         float(m.group(9)),
                    "optim":        float(m.group(10)),
                    "total":        float(m.group(11)),
                })

    if not steps:
        raise ValueError(f"No step timing lines found in {path}")

    # Skip warmup steps
    steps = steps[skip_steps:]
    if not steps:
        raise ValueError(f"No steps remaining after skipping {skip_steps} warmup steps in {path}")

    phases = ["data", "prepare", "pack", "forward", "backward", "clip", "optim"]
    totals = [s["total"] for s in steps]
    tgs_vals = [s["tgs"] for s in steps]

    result = {
        "path":           path,
        "nodes":          nodes,
        "gpus_per_node":  gpus_per_node,
        "total_gpus":     total_gpus,
        "global_batch":   global_batch,
        "n_steps":        len(steps),
        "step_time_mean": mean(totals),
        "step_time_std":  stdev(totals) if len(totals) > 1 else 0.0,
        "steps_per_sec":  1.0 / mean(totals),
        "tgs_mean":       mean(tgs_vals),
        "phases":         {p: mean(s[p] for s in steps) for p in phases},
    }

    # samples/sec = global_batch / step_time  (if we know global_batch)
    if global_batch:
        result["samples_per_sec"] = global_batch / result["step_time_mean"]

    return result


def format_table(headers: list[str], rows: list[list], col_width: int = 14) -> str:
    lines = []
    fmt = ("{{:<{}}}".format(col_width)) * len(headers)
    lines.append(fmt.format(*headers))
    lines.append("-" * (col_width * len(headers)))
    for row in rows:
        lines.append(fmt.format(*[str(v) for v in row]))
    return "\n".join(lines)


def analyze(runs: dict[str, dict], skip_steps: int = 5):
    """
    runs: {label: parsed_result_dict}
    """
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    # --- Per-run summary ---
    print("\n[Throughput]\n")
    headers = ["Label", "Nodes", "TotalGPUs", "Steps", "StepTime(s)", "Steps/s", "Tokens/s"]
    rows = []
    for label, r in runs.items():
        rows.append([
            label,
            r["nodes"] or "?",
            r["total_gpus"] or "?",
            r["n_steps"],
            f"{r['step_time_mean']:.3f}±{r['step_time_std']:.3f}",
            f"{r['steps_per_sec']:.3f}",
            f"{r['tgs_mean']:.0f}",
        ])
    print(format_table(headers, rows))

    # --- Time breakdown per phase ---
    print("\n[Step Time Breakdown — % of total step time]\n")
    phases = ["data", "prepare", "pack", "forward", "backward", "clip", "optim"]
    headers = ["Label"] + [p[:8] for p in phases] + ["accounted"]
    rows = []
    for label, r in runs.items():
        t = r["step_time_mean"]
        pct = [f"{100 * r['phases'][p] / t:.1f}%" for p in phases]
        accounted = sum(r["phases"][p] for p in phases)
        rows.append([label] + pct + [f"{100 * accounted / t:.1f}%"])
    print(format_table(headers, rows, col_width=12))

    print("\n  NOTE: backward_time includes NCCL AllReduce (gradient sync across GPUs/nodes).")
    print("        If backward% grows from 1-node to multi-node, inter-node bandwidth is the bottleneck.")
    print("        If data% is high, data loading (I/O from shared filesystem) is the bottleneck.")

    # --- Scaling efficiency (only if we have a 1-node baseline) ---
    baseline = next(
        (r for r in runs.values() if r["nodes"] == 1),
        next(iter(runs.values()))  # fallback: use first run as baseline
    )
    baseline_label = next(
        (l for l, r in runs.items() if r["nodes"] == 1),
        next(iter(runs.keys()))
    )

    if len(runs) > 1:
        print(f"\n[Scaling Efficiency]  (baseline: {baseline_label})\n")
        headers = ["Label", "Nodes", "TotalGPUs", "Speedup", "IdealSpeedup", "Efficiency", "Bottleneck"]
        rows = []
        for label, r in runs.items():
            if r is baseline:
                continue
            if r["nodes"] and baseline["nodes"]:
                node_ratio = r["nodes"] / baseline["nodes"]
            else:
                node_ratio = None

            speedup = baseline["step_time_mean"] / r["step_time_mean"]
            ideal = node_ratio or speedup  # ideal = linear with node count
            efficiency = speedup / ideal if ideal else 0.0

            # Identify bottleneck: phase with largest absolute time increase
            bottleneck = max(
                ["data", "forward", "backward", "optim"],
                key=lambda p: r["phases"][p] - baseline["phases"][p]
            )
            bottleneck_delta = r["phases"][bottleneck] - baseline["phases"][bottleneck]
            bottleneck_str = f"{bottleneck}(+{bottleneck_delta:.2f}s)"

            rows.append([
                label,
                r["nodes"] or "?",
                r["total_gpus"] or "?",
                f"{speedup:.2f}x",
                f"{ideal:.1f}x",
                f"{efficiency * 100:.1f}%",
                bottleneck_str,
            ])
        print(format_table(headers, rows, col_width=16))

        print("\n  Efficiency > 80%: good scaling")
        print("  Efficiency 50-80%: moderate, check the flagged bottleneck phase")
        print("  Efficiency < 50%: poor — likely I/O bound or communication overhead too high")

    print()


def auto_discover(results_dir: Path) -> dict[str, Path]:
    """Find all run.log files under results/ and label them by parent dir names."""
    logs = sorted(results_dir.glob("*/*/run.log"))
    result = {}
    for log in logs:
        job = log.parent.parent.name   # e.g. bench_2node
        ts = log.parent.name            # e.g. 260416_110000
        result[f"{job}/{ts}"] = log
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multinode benchmark run.log files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "logs", nargs="*",
        help="run.log file paths (optionally prefixed with LABEL=path)"
    )
    parser.add_argument(
        "--runs", nargs="+", metavar="LABEL=PATH",
        help="Labeled run logs: 1-node=path/run.log 2-node=path/run.log ..."
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-discover all run.log files under results/"
    )
    parser.add_argument(
        "--skip", type=int, default=5, metavar="N",
        help="Steps to skip at the start of each run (default: 5, accounts for CUDA warmup)"
    )
    args = parser.parse_args()

    # Build {label: path} mapping
    log_paths: dict[str, Path] = {}

    if args.auto:
        results_dir = Path(__file__).parent / "results"
        log_paths = auto_discover(results_dir)
        if not log_paths:
            print(f"No run.log files found under {results_dir}")
            sys.exit(1)
    elif args.runs:
        for entry in args.runs:
            if "=" in entry:
                label, path = entry.split("=", 1)
            else:
                path = entry
                label = Path(path).parent.parent.name
            log_paths[label] = Path(path)
    elif args.logs:
        for entry in args.logs:
            if "=" in entry:
                label, path = entry.split("=", 1)
            else:
                path = entry
                label = Path(path).parent.parent.name
            log_paths[label] = Path(path)
    else:
        parser.print_help()
        sys.exit(0)

    # Parse each log
    parsed = {}
    for label, path in log_paths.items():
        try:
            parsed[label] = parse_log(path, skip_steps=args.skip)
            print(f"Parsed {label}: {parsed[label]['n_steps']} steps (skipped first {args.skip})")
        except Exception as e:
            print(f"WARNING: could not parse {label} ({path}): {e}")

    if not parsed:
        print("No logs could be parsed.")
        sys.exit(1)

    analyze(parsed, skip_steps=args.skip)


if __name__ == "__main__":
    main()
