#!/usr/bin/env python3
"""
Analyze benchmark run directories to measure throughput and identify bottlenecks.

Usage:
    # Auto-discover all runs under results/:
    python3 analyze.py --auto

    # Explicit run directories or run.log paths (optionally labeled):
    python3 analyze.py results/bench_1node/260417_182942 \\
                       results/bench_2node/260417_183417 \\
                       results/bench_4node/260417_183425

    python3 analyze.py \\
        "1-node=results/bench_1node/260417_182942" \\
        "2-node=results/bench_2node/260417_183417" \\
        "4-node=results/bench_4node/260417_183425"

When per-rank log files are present (results/<job>/<ts>/<ts2>/rank*.log),
step time is measured as max(time across all ranks) — the true wall-clock cost.
Falls back to rank-0-only parsing of run.log if per-rank logs are absent.
"""

import argparse
import re
import sys
from pathlib import Path
from statistics import mean, stdev

PHASES = ["data", "prepare", "pack", "forward", "backward", "clip", "optim"]

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
_RANK0_RE = re.compile(r"\[RANK\s+0\]")
_NODES_RE = re.compile(r"Nodes:\s*(\d+)\s+GPUs/node:\s*(\d+)\s+Total GPUs:\s*(\d+)")
_BATCH_RE = re.compile(r"global_batch=(\d+)")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_step_line(line: str) -> dict:
    m = _STEP_RE.search(line)
    if not m:
        return None
    return {
        "step":     int(m.group(1)),
        "tgs":      float(m.group(3)),
        "data":     float(m.group(4)),
        "prepare":  float(m.group(5)),
        "pack":     float(m.group(6)),
        "forward":  float(m.group(7)),
        "backward": float(m.group(8)),
        "clip":     float(m.group(9)),
        "optim":    float(m.group(10)),
        "total":    float(m.group(11)),
    }


def _read_header(run_log: Path) -> dict:
    nodes = gpus_per_node = total_gpus = global_batch = None
    with open(run_log) as f:
        for line in f:
            m = _NODES_RE.search(line)
            if m:
                nodes = int(m.group(1))
                gpus_per_node = int(m.group(2))
                total_gpus = int(m.group(3))
            m = _BATCH_RE.search(line)
            if m:
                global_batch = int(m.group(1))
            if nodes and global_batch:
                break
    return {"nodes": nodes, "gpus_per_node": gpus_per_node,
            "total_gpus": total_gpus, "global_batch": global_batch}


def _find_rank_log_dir(run_dir: Path):
    for subdir in sorted(run_dir.iterdir()):
        if subdir.is_dir() and list(subdir.glob("rank*.log")):
            return subdir
    return None


def _parse_rank_file(path: Path) -> list[dict]:
    steps = []
    with open(path) as f:
        for line in f:
            s = _parse_step_line(line)
            if s:
                steps.append(s)
    return steps


# ---------------------------------------------------------------------------
# Main parse entry point
# ---------------------------------------------------------------------------

def parse_run(run_dir: Path, skip_steps: int = 5) -> dict:
    """
    Parse a benchmark run directory.
    Prefers per-rank logs (wall-clock = max across ranks).
    Falls back to run.log rank-0 parsing if per-rank logs are absent.
    """
    run_log = run_dir / "run.log"
    header = _read_header(run_log) if run_log.exists() else {}

    rank_log_dir = _find_rank_log_dir(run_dir)
    if rank_log_dir:
        rank_files = sorted(rank_log_dir.glob("rank*.log"),
                            key=lambda p: int(p.stem[4:]))
        rank_data = {int(p.stem[4:]): _parse_rank_file(p) for p in rank_files}
        return _aggregate_ranks(rank_data, skip_steps, **header)

    if run_log.exists():
        return _parse_run_log(run_log, skip_steps, **header)

    raise ValueError(f"No run.log or rank logs found in {run_dir}")


def _aggregate_ranks(rank_data: dict, skip_steps: int,
                     nodes=None, gpus_per_node=None,
                     total_gpus=None, global_batch=None) -> dict:
    """
    Aggregate per-rank step data into a single result dict.

    Wall-clock step time = max(time across ranks) per step — the true cost
    because all ranks must finish before the next step can begin.

    Phase times = max across ranks (represents the slowest rank per phase).
    Forward imbalance = stdev(forward_time across ranks) — how uneven the load is.
    """
    # Build step_map[step][rank] = timing dict
    step_map: dict[int, dict[int, dict]] = {}
    for rank, steps in rank_data.items():
        for s in steps:
            step_map.setdefault(s["step"], {})[rank] = s

    n_ranks = len(rank_data)
    common = sorted(k for k, v in step_map.items() if len(v) == n_ranks)
    common = common[skip_steps:]

    if not common:
        raise ValueError(f"No steps remaining after skipping {skip_steps} warmup steps")

    records = []
    for step in common:
        rv = step_map[step]
        fwd_times = [v["forward"] for v in rv.values()]
        all_totals = [v["total"] for v in rv.values()]
        records.append({
            "step":          step,
            "wall_clock":    max(all_totals),
            "tgs":           rv[0]["tgs"] if 0 in rv else mean(v["tgs"] for v in rv.values()),
            "fwd_imbalance": stdev(fwd_times) if len(fwd_times) > 1 else 0.0,
            "fwd_max":       max(fwd_times),
            **{p: max(v[p] for v in rv.values()) for p in PHASES},
        })

    totals = [r["wall_clock"] for r in records]

    result = {
        "nodes":              nodes,
        "gpus_per_node":      gpus_per_node,
        "total_gpus":         total_gpus,
        "global_batch":       global_batch,
        "n_steps":            len(records),
        "n_ranks":            n_ranks,
        "step_time_mean":     mean(totals),
        "step_time_std":      stdev(totals) if len(totals) > 1 else 0.0,
        "steps_per_sec":      1.0 / mean(totals),
        "tgs_mean":           mean(r["tgs"] for r in records),
        "phases":             {p: mean(r[p] for r in records) for p in PHASES},
        "fwd_imbalance_mean": mean(r["fwd_imbalance"] for r in records),
        "fwd_imbalance_pct":  mean(
            r["fwd_imbalance"] / r["fwd_max"] * 100
            for r in records if r["fwd_max"] > 0
        ),
        "source": "per-rank logs  (wall-clock = max across ranks)",
    }
    if global_batch:
        result["samples_per_sec"] = global_batch / result["step_time_mean"]
    return result


def _parse_run_log(path: Path, skip_steps: int,
                   nodes=None, gpus_per_node=None,
                   total_gpus=None, global_batch=None) -> dict:
    """Fallback: parse run.log filtering to RANK 0 lines only."""
    steps = []
    with open(path) as f:
        for line in f:
            if not _RANK0_RE.search(line):
                continue
            s = _parse_step_line(line)
            if s:
                steps.append(s)
    if not steps:
        raise ValueError(f"No step lines found in {path}")
    steps = steps[skip_steps:]
    if not steps:
        raise ValueError(f"No steps after skipping {skip_steps} warmup in {path}")

    totals = [s["total"] for s in steps]
    result = {
        "nodes":              nodes,
        "gpus_per_node":      gpus_per_node,
        "total_gpus":         total_gpus,
        "global_batch":       global_batch,
        "n_steps":            len(steps),
        "n_ranks":            total_gpus,
        "step_time_mean":     mean(totals),
        "step_time_std":      stdev(totals) if len(totals) > 1 else 0.0,
        "steps_per_sec":      1.0 / mean(totals),
        "tgs_mean":           mean(s["tgs"] for s in steps),
        "phases":             {p: mean(s[p] for s in steps) for p in PHASES},
        "fwd_imbalance_mean": 0.0,
        "fwd_imbalance_pct":  0.0,
        "source": "run.log (rank 0 only — per-rank logs not found)",
    }
    if global_batch:
        result["samples_per_sec"] = global_batch / result["step_time_mean"]
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt_table(headers: list, rows: list, col_widths=14) -> str:
    if isinstance(col_widths, int):
        col_widths = [col_widths] * len(headers)
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-" * sum(col_widths)
    lines = [fmt.format(*[str(h) for h in headers]), sep]
    for row in rows:
        lines.append(fmt.format(*[str(v) for v in row]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(runs: dict[str, dict]):
    SEP = "=" * 76

    # Identify 1-node baseline (lowest node count, or first run)
    baseline_label, baseline = min(
        runs.items(),
        key=lambda kv: (kv[1]["nodes"] or 999, kv[0])
    )

    print(f"\n{SEP}")
    print("BENCHMARK RESULTS")
    print(SEP)

    # ------------------------------------------------------------------
    # 1. Throughput
    # ------------------------------------------------------------------
    print("\n[1. Throughput]\n")
    headers = ["Label", "Nodes", "GPUs", "Steps", "StepTime(s)", "Variance(%)", "Steps/s", "Tokens/s"]
    widths  = [20,       7,       6,      7,        18,            13,             9,         10]
    rows = []
    for label, r in runs.items():
        cv = 100 * r["step_time_std"] / r["step_time_mean"]
        rows.append([
            label,
            r["nodes"] or "?",
            r["total_gpus"] or "?",
            r["n_steps"],
            f"{r['step_time_mean']:.3f}±{r['step_time_std']:.3f}",
            f"{cv:.1f}%",
            f"{r['steps_per_sec']:.3f}",
            f"{r['tgs_mean']:.0f}",
        ])
    print(_fmt_table(headers, rows, widths))
    print(f"\n  Source: {next(iter(runs.values()))['source']}")
    print("  Variance% = step time coefficient of variation (high = unstable network or load imbalance)")

    # ------------------------------------------------------------------
    # 2. Phase breakdown with unaccounted time
    # ------------------------------------------------------------------
    print("\n[2. Step Time Breakdown — mean per step, % of wall-clock]\n")
    headers = ["Label"] + [p[:8] for p in PHASES] + ["unaccnt"]
    widths  = [20] + [10] * len(PHASES) + [16]
    rows = []
    for label, r in runs.items():
        t = r["step_time_mean"]
        pcts = [f"{100 * r['phases'][p] / t:.1f}%" for p in PHASES]
        unaccounted = t - sum(r["phases"][p] for p in PHASES)
        rows.append([label] + pcts + [f"{100*unaccounted/t:.1f}% ({unaccounted:.2f}s)"])
    print(_fmt_table(headers, rows, widths))
    print("\n  unaccnt = time not captured by any logged phase (MPI barriers, ZeRO gather, etc.)")
    print("  backward includes NCCL AllReduce — see section 3 for the isolated overhead.")

    # ------------------------------------------------------------------
    # 3. AllReduce overhead
    # ------------------------------------------------------------------
    print(f"\n[3. NCCL AllReduce Overhead]  (baseline backward = no inter-node AllReduce)\n")
    print(f"  {baseline_label} backward (intra-node only): {baseline['phases']['backward']:.3f}s\n")
    headers = ["Label",  "backward(s)", "allreduce_cost(s)", "cost_%_of_step"]
    widths  = [20,        14,             20,                  16]
    rows = []
    for label, r in runs.items():
        overhead = r["phases"]["backward"] - baseline["phases"]["backward"]
        pct = 100 * overhead / r["step_time_mean"]
        rows.append([
            label,
            f"{r['phases']['backward']:.3f}",
            f"{overhead:+.3f}",
            f"{pct:.1f}%",
        ])
    print(_fmt_table(headers, rows, widths))
    print("\n  allreduce_cost = backward[run] − backward[1-node baseline]")
    print("  A large positive value means inter-node gradient sync dominates step time.")

    # ------------------------------------------------------------------
    # 4. Load imbalance
    # ------------------------------------------------------------------
    has_imbalance = any(r.get("fwd_imbalance_mean", 0) > 0 for r in runs.values())
    if has_imbalance:
        print("\n[4. Forward Pass Load Imbalance across Ranks]\n")
        headers = ["Label",  "fwd_mean(s)", "fwd_std(s)", "imbalance%"]
        widths  = [20,        13,             12,            12]
        rows = []
        for label, r in runs.items():
            rows.append([
                label,
                f"{r['phases']['forward']:.3f}",
                f"{r.get('fwd_imbalance_mean', 0):.3f}",
                f"{r.get('fwd_imbalance_pct', 0):.1f}%",
            ])
        print(_fmt_table(headers, rows, widths))
        print("\n  imbalance% = std(forward_time across ranks) / max(forward_time) per step.")
        print("  Caused by --group-by-length assigning different sequence lengths to different ranks.")
        print("  The slowest rank stalls all others at the AllReduce barrier.")

    # ------------------------------------------------------------------
    # 5. Scaling efficiency
    # ------------------------------------------------------------------
    if len(runs) > 1:
        print(f"\n[5. Scaling Efficiency]  (baseline: {baseline_label})\n")
        headers = ["Label",  "Nodes", "GPUs", "Speedup", "Ideal", "Efficiency", "Bottleneck"]
        widths  = [20,        7,       6,       9,          7,       12,            22]
        rows = []
        for label, r in runs.items():
            if r is baseline:
                continue
            node_ratio = (r["nodes"] / baseline["nodes"]) if (r["nodes"] and baseline["nodes"]) else None
            speedup = baseline["step_time_mean"] / r["step_time_mean"]
            ideal = node_ratio or speedup
            eff = speedup / ideal if ideal else 0.0
            bottleneck = max(
                ["data", "forward", "backward", "optim"],
                key=lambda p: r["phases"][p] - baseline["phases"][p]
            )
            delta = r["phases"][bottleneck] - baseline["phases"][bottleneck]
            rows.append([
                label,
                r["nodes"] or "?",
                r["total_gpus"] or "?",
                f"{speedup:.2f}x",
                f"{ideal:.1f}x",
                f"{eff * 100:.1f}%",
                f"{bottleneck}(+{delta:.2f}s)",
            ])
        print(_fmt_table(headers, rows, widths))
        print("\n  Efficiency = actual_speedup / ideal_speedup")
        print("  > 80%: good   50-80%: moderate   < 50%: poor")

    # ------------------------------------------------------------------
    # 6. Recommendations
    # ------------------------------------------------------------------
    print("\n[6. Recommendations]\n")
    recs = []

    # NCCL / AllReduce
    for label, r in runs.items():
        if r is baseline:
            continue
        allreduce = r["phases"]["backward"] - baseline["phases"]["backward"]
        pct = 100 * allreduce / r["step_time_mean"]
        if pct > 20:
            recs.append((
                "CRITICAL",
                "Inter-node NCCL communication (AllReduce)",
                f"AllReduce adds {allreduce:.1f}s/step ({pct:.0f}% of wall-clock) on {label}.\n"
                "  → Check if InfiniBand is available on compute nodes:\n"
                "       ibstat | grep 'Port State'\n"
                "       ip link show | grep -E '^[0-9]+: (ib|mlx)'\n"
                "  → If IB is present: change NCCL_SOCKET_IFNAME=bond0 → ib0 in _common.sh\n"
                "  → Or remove NCCL_SOCKET_IFNAME entirely to let NCCL auto-select (prefers IB)\n"
                "  → If only Ethernet: try NCCL_ALGO=Ring and NCCL_NET_GDR_LEVEL=0"
            ))
            break

    # Data I/O
    for label, r in runs.items():
        if r is baseline:
            continue
        data_abs = r["phases"]["data"]
        data_pct = 100 * data_abs / r["step_time_mean"]
        data_vs_baseline = data_abs / baseline["step_time_mean"] * 100
        if data_vs_baseline > 15:
            recs.append((
                "HIGH",
                "Data I/O contention on shared filesystem",
                f"data_time is {data_abs:.2f}s on {label} ({data_pct:.0f}% of step, "
                f"{data_vs_baseline:.0f}% of 1-node step time).\n"
                "  Currently hidden behind NCCL overhead — will surface after fixing NCCL.\n"
                "  → Pre-copy dataset to local node storage (/local1) at job start\n"
                "  → Switch to LMDB format for faster random-access reads\n"
                "  → Reduce NUM_WORKERS if many ranks saturate the NFS mount"
            ))
            break

    # Load imbalance
    if has_imbalance:
        worst_label, worst = max(
            ((l, r) for l, r in runs.items()),
            key=lambda kv: kv[1].get("fwd_imbalance_pct", 0)
        )
        imb = worst.get("fwd_imbalance_pct", 0)
        if imb > 15:
            recs.append((
                "MEDIUM",
                "Forward pass load imbalance across ranks",
                f"forward_time std is {imb:.0f}% across ranks on {worst_label}.\n"
                "  Ranks processing longer sequences block all others at the AllReduce barrier.\n"
                "  → Tighten --group-by-length bin sizes to reduce within-batch length variance\n"
                "  → Consider sequence packing (pack_time) to equalize compute per rank"
            ))

    # Unaccounted time
    for label, r in runs.items():
        if r is baseline:
            continue
        unaccounted = r["step_time_mean"] - sum(r["phases"][p] for p in PHASES)
        pct = 100 * unaccounted / r["step_time_mean"]
        if pct > 15:
            recs.append((
                "MEDIUM",
                "Unaccounted overhead",
                f"{unaccounted:.2f}s/step ({pct:.0f}%) is not captured in any logged phase on {label}.\n"
                "  Likely: MPI barrier synchronization, ZeRO-2 parameter gather/scatter.\n"
                "  → Add timing instrumentation around ZeRO gather/scatter in the training script\n"
                "  → Check mpirun --mca options for unnecessary barrier overhead"
            ))
            break

    # High variance
    for label, r in runs.items():
        if r is baseline:
            continue
        cv = 100 * r["step_time_std"] / r["step_time_mean"]
        if cv > 25:
            recs.append((
                "LOW",
                "High step time variance",
                f"Step time CV is {cv:.0f}% on {label} — network is inconsistent.\n"
                "  May resolve after fixing NCCL (item 1).\n"
                "  If it persists: check for shared network contention with other running jobs"
            ))
            break

    if not recs:
        print("  No significant bottlenecks detected.")
    else:
        for i, (severity, title, detail) in enumerate(recs, 1):
            print(f"  {i}. [{severity}] {title}")
            for line in detail.splitlines():
                print(f"     {line}")
            print()

    print()


# ---------------------------------------------------------------------------
# Auto-discovery and CLI
# ---------------------------------------------------------------------------

def auto_discover(results_dir: Path) -> dict[str, Path]:
    """Find run directories under results/. Returns {label: run_dir}."""
    run_dirs: dict[str, Path] = {}
    for run_log in sorted(results_dir.glob("*/*/run.log")):
        run_dir = run_log.parent
        job = run_dir.parent.name
        ts  = run_dir.name
        run_dirs[f"{job}/{ts}"] = run_dir

    # Simplify labels to just job name when each job has exactly one run
    job_counts: dict[str, int] = {}
    for key in run_dirs:
        job = key.split("/")[0]
        job_counts[job] = job_counts.get(job, 0) + 1
    if all(v == 1 for v in job_counts.values()):
        return {key.split("/")[0]: path for key, path in run_dirs.items()}
    return run_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multinode benchmark runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "logs", nargs="*",
        help="Run directories or run.log paths (optionally prefixed LABEL=path)"
    )
    parser.add_argument("--runs", nargs="+", metavar="LABEL=PATH")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover all runs under results/")
    parser.add_argument("--skip", type=int, default=5, metavar="N",
                        help="Warmup steps to skip per run (default: 5)")
    args = parser.parse_args()

    path_map: dict[str, Path] = {}

    if args.auto:
        results_dir = Path(__file__).parent / "results"
        path_map = auto_discover(results_dir)
        if not path_map:
            print(f"No runs found under {results_dir}")
            sys.exit(1)
    elif args.runs:
        for entry in args.runs:
            if "=" in entry:
                label, path = entry.split("=", 1)
            else:
                label = Path(entry).parent.parent.name
                path = entry
            path_map[label] = Path(path)
    elif args.logs:
        for entry in args.logs:
            if "=" in entry:
                label, path = entry.split("=", 1)
            else:
                label = Path(entry).parent.parent.name
                path = entry
            path_map[label] = Path(path)
    else:
        parser.print_help()
        sys.exit(0)

    parsed: dict[str, dict] = {}
    for label, path in path_map.items():
        path = Path(path)
        run_dir = path.parent if path.name == "run.log" else path
        try:
            result = parse_run(run_dir, skip_steps=args.skip)
            parsed[label] = result
            print(f"Parsed {label}: {result['n_steps']} steps "
                  f"(skipped first {args.skip})  [{result['source']}]")
        except Exception as e:
            print(f"WARNING: could not parse {label} ({run_dir}): {e}")

    if not parsed:
        print("No runs could be parsed.")
        sys.exit(1)

    analyze(parsed)


if __name__ == "__main__":
    main()
