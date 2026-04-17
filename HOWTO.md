# Multinode Benchmark — How to Run

## Prerequisites

1. `paths.json` must be configured in `internvideo_2_5/data/`:
   ```bash
   cp /groups/input/internvideo25/internvideo_2_5/data/paths.example.json \
      /groups/input/internvideo25/internvideo_2_5/data/paths.json
   # Edit paths.json to fill in HANDYVQA_ROOT, UNCLIPPED_ANN_DIR, etc.
   ```

2. Verify `handyvqa` resolves correctly:
   ```bash
   cd /groups/input/internvideo25/internvideo_2_5
   python3 scripts/make_train_json.py handyvqa -o /tmp/test.json && echo OK
   ```

---

## Step 1 — Submit benchmark jobs

Submit from anywhere — `_common.sh` always `cd`s to the repo using an absolute path:

```bash
qsub /groups/input/internvideo25/multinode_bench/jobs/bench_1node.sh
qsub /groups/input/internvideo25/multinode_bench/jobs/bench_2node.sh
qsub /groups/input/internvideo25/multinode_bench/jobs/bench_4node.sh
```

You can submit all three at once — they are independent.

---

## Step 2 — Monitor progress

```bash
qstat                        # check job status (Q=queued, R=running, E=ending)
qstat -f <job_id>            # full details for a specific job

# Tail the live log of a running job:
tail -f /groups/input/internvideo25/multinode_bench/results/bench_1node/*/run.log
```

Each job runs for exactly 200 optimizer steps then exits (controlled by `MAX_STEPS`).
Expected wall-clock time per job: 10–30 min depending on GPU count and node configuration.

---

## Step 3 — Analyze results

Once at least one job has finished:

```bash
cd /groups/input/internvideo25/multinode_bench

# Auto-discover all completed runs and compare:
python3 analyze.py --auto

# Or point to specific log files:
python3 analyze.py \
    "1-node=results/bench_1node/<timestamp>/run.log" \
    "2-node=results/bench_2node/<timestamp>/run.log" \
    "4-node=results/bench_4node/<timestamp>/run.log"
```

---

## Understanding the output

### Throughput table
Shows steps/sec and tokens/sec per run. Higher is better.

### Time breakdown
Shows what fraction of each step is spent in each phase:

| Phase | What it means |
|---|---|
| `data` | Reading samples from disk / LMDB |
| `prepare` | Moving tensors to GPU |
| `forward` | Forward pass (compute) |
| `backward` | Backward pass **+ NCCL AllReduce** (gradient sync across GPUs/nodes) |
| `optim` | Optimizer step (ZeRO parameter update) |

### Scaling efficiency table
```
Efficiency = actual_speedup / ideal_speedup
```
- **> 80%** — good scaling, multi-node is working well
- **50–80%** — moderate; check the flagged bottleneck phase
- **< 50%** — poor scaling; bottleneck phase tells you what to fix

### Bottleneck interpretation

| Symptom | Likely cause | Fix |
|---|---|---|
| `backward` time grows with nodes | Inter-node NCCL bandwidth | Check `NCCL_SOCKET_IFNAME`, use IB if available |
| `data` time is high and constant | Shared filesystem I/O saturation | Use LMDB datasets, pre-copy to `/local1` |
| `forward` time dominates, scales well | Compute-bound | No problem — this is ideal |
| Low efficiency even on 2 nodes | MPI not distributing correctly | Verify `--map-by node` and PBS_NODEFILE |

---

## Adjusting the benchmark

All parameters are set at the top of each job script:

```bash
MAX_STEPS=200       # number of steps to run (increase for more stable averages)
MICRO_BATCH_SIZE=1  # per-GPU batch size
ACCUMULATIVE_COUNTS=4
DATASETS="handyvqa" # change to any registry name from data/registry.json
```

To benchmark connector-only training (frozen ViT + LLM, as in the original multinode job):
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
    │       ├── run.log        ← full output with per-step timings
    │       ├── datasets.json  ← dataset config used
    │       └── bench_1node.sh ← copy of job script
    ├── bench_2node/
    │   └── <timestamp>/
    └── bench_4node/
        └── <timestamp>/
```

Each run is self-contained. Re-submitting creates a new timestamped directory.
