# Multinode Benchmark — How to Run

## Prerequisites

1. `paths.json` must be configured in `internvideo25_hpc/data/`:
   ```bash
   cp internvideo25_hpc/data/paths.example.json internvideo25_hpc/data/paths.json
   # Edit paths.json to fill in HANDYVQA_ROOT, UNCLIPPED_ANN_DIR, etc.
   ```

2. Verify `handyvqa` resolves correctly:
   ```bash
   cd internvideo25_hpc
   python3 scripts/make_train_json.py handyvqa -o /tmp/test.json && echo OK
   ```

---

## Step 1 — Submit benchmark jobs

Submit from the `multinode_bench/` directory:

```bash
qsub jobs/bench_1node.sh
qsub jobs/bench_2node.sh
qsub jobs/bench_4node.sh
```

You can submit all three at once — they are independent.

---

## Step 2 — Monitor progress

```bash
qstat                        # check job status (Q=queued, R=running, E=ending)
qstat -f <job_id>            # full details for a specific job

# Tail the live log of a running job:
tail -f multinode_bench/results/bench_1node/*/run.log
```

Each job runs for exactly `MAX_STEPS` optimizer steps then exits (default 200).
Expected wall-clock time per job: 10–30 min depending on GPU count and configuration.

---

## Step 3 — Analyze results

Once at least one job has finished:

```bash
cd /groups/input/internvideo25/multinode_bench

# Auto-discover all completed runs and compare:
python3 analyze.py --auto

# Or point to specific run directories:
python3 analyze.py \
    "1-node=results/bench_1node/<timestamp>" \
    "2-node=results/bench_2node/<timestamp>" \
    "4-node=results/bench_4node/<timestamp>"
```

The analyzer uses **per-rank log files** when available
(`results/<job>/<timestamp>/<timestamp2>/rank*.log`), measuring wall-clock step
time as `max(time across all ranks)`. Falls back to `run.log` (rank-0 only) if
per-rank logs are absent.

---

## Understanding the output

### 1. Throughput table
Shows wall-clock steps/sec and tokens/sec per run. Higher is better.
`Variance%` is the coefficient of variation of step times — high values (>25%)
indicate network instability or load imbalance.

### 2. Step time breakdown
Shows what fraction of each step is spent in each phase:

| Phase | What it means |
|---|---|
| `data` | Reading samples from disk / LMDB |
| `prepare` | Moving tensors to GPU |
| `forward` | Forward pass (compute) |
| `backward` | Backward pass **+ NCCL AllReduce** (gradient sync across GPUs/nodes) |
| `optim` | Optimizer step (ZeRO parameter update) |
| `unaccnt` | Unaccounted time (MPI barriers, ZeRO gather/scatter overhead) |

### 3. AllReduce overhead
Isolates the inter-node communication cost:
```
allreduce_cost = backward_time[run] − backward_time[1-node baseline]
```
The 1-node backward has no inter-node AllReduce, so the delta is pure communication overhead.

### 4. Forward pass load imbalance
`imbalance% = std(forward_time across ranks) / max(forward_time)` per step.
High values mean `--group-by-length` is assigning very different sequence lengths
to different ranks — the slowest rank stalls everyone at the AllReduce barrier.

### 5. Scaling efficiency
```
Efficiency = actual_speedup / ideal_speedup
```
- **> 80%** — good scaling
- **50–80%** — moderate; check the flagged bottleneck phase
- **< 50%** — poor scaling

### 6. Recommendations
Prioritized list of bottlenecks with concrete fix commands.

---

## Network interface (NCCL)

`_common.sh` sets `NCCL_SOCKET_IFNAME=bond0` (bonded Ethernet, the ABCI default).
If the cluster has **InfiniBand**, switching will dramatically reduce AllReduce cost:

```bash
# Check if InfiniBand is available on a compute node:
ibstat | grep "Port State"
ip link show | grep -E "^[0-9]+: (ib|mlx)"

# In multinode_bench/_common.sh, change:
export NCCL_SOCKET_IFNAME=bond0   # Ethernet
# to:
export NCCL_SOCKET_IFNAME=ib0     # InfiniBand
# or remove the line entirely to let NCCL auto-select (prefers IB over Ethernet):
# (remove the NCCL_SOCKET_IFNAME line)
```

If only Ethernet is available, also try:
```bash
export NCCL_ALGO=Ring
export NCCL_NET_GDR_LEVEL=0
```

---

## Data I/O

`data_time` in the logs measures how long each rank spends reading a batch from disk.
With many ranks hitting a shared NFS filesystem simultaneously, this grows with node count.
Currently hidden behind NCCL overhead — it becomes the next bottleneck once NCCL is fixed.

```bash
# Pre-copy dataset to fast local storage before the job starts:
cp -r /path/to/dataset /local1/

# Or use LMDB format, which has much faster random-access reads.
```

---

## Adjusting the benchmark

All parameters are set at the top of each job script:

```bash
MAX_STEPS=200       # number of optimizer steps to run
MICRO_BATCH_SIZE=1  # per-GPU batch size
ACCUMULATIVE_COUNTS=4
DATASETS="handyvqa"
```

Note: each rank processes `MICRO_BATCH_SIZE × ACCUMULATIVE_COUNTS × MAX_STEPS` samples.
With more GPUs the **global batch size** scales up, so the same 200 steps consume more
data in total — but the per-rank workload stays constant.

To benchmark connector-only training (frozen ViT + LLM):
```bash
EXTRA_ARGS="--freeze-vit --freeze-llm"
```

---

## Results location

```
multinode_bench/
└── results/
    ├── bench_1node/
    │   └── <timestamp>/
    │       ├── run.log              ← full output with per-step timings (all ranks)
    │       ├── datasets.json        ← dataset config used
    │       ├── bench_1node.sh       ← copy of job script
    │       └── <timestamp2>/
    │           ├── rank0.log        ← per-rank log (rank 0)
    │           ├── rank1.log
    │           └── ...
    ├── bench_2node/
    │   └── <timestamp>/
    └── bench_4node/
        └── <timestamp>/
```

Each run is self-contained. Re-submitting creates a new timestamped directory.
