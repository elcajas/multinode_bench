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

Each job runs for `MAX_STEPS=20` optimizer steps then exits (~4 minutes wall-clock).
Increase `MAX_STEPS` in the job script if you need more stable averages.

---

## Step 2 — Monitor progress

```bash
qstat                        # check job status (Q=queued, R=running, E=ending)
qstat -f <job_id>            # full details for a specific job

# Tail the live log of a running job:
tail -f multinode_bench/results/bench_2node/*/run.log
```

The log header prints the allocated nodes, e.g.:
```
Allocated nodes: qh131 qh135
```
Check this immediately — if a known-bad node appears (qh129, qh138), the job will abort with an error message and you can resubmit.

---

## Step 3 — Analyze results

Once at least one job has finished:

```bash
cd /groups/input/internvideo25/multinode_bench

# Fast single-run summary (max across ranks, mean/std/CV, scaling efficiency):
python3 parse_run.py results/bench_2node/<timestamp>/run.log

# Compare multiple runs side-by-side:
python3 parse_run.py results/bench_2node_gs*/*/run.log

# Full multi-run analysis with breakdown and recommendations:
python3 analyze.py --auto
python3 analyze.py \
    "1-node=results/bench_1node/<timestamp>" \
    "2-node=results/bench_2node/<timestamp>"
```

`parse_run.py` accepts a run.log, a PBS .OU file, or a run directory.
It measures wall-clock step time as `max(time across all ranks)`.

`analyze.py` uses per-rank log files when available
(`results/<job>/<timestamp>/<timestamp2>/rank*.log`), falling back to `run.log`.

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
| `forward` | Forward pass (compute + AllGather params) |
| `backward` | Backward pass (compute + gradient ReduceScatter/AllReduce) |
| `optim` | Optimizer step |
| `unaccnt` | Unaccounted time (MPI barriers, ZeRO overhead) |

### 3. AllReduce overhead
Isolates the inter-node communication cost:
```
allreduce_cost = backward_time[run] − backward_time[1-node baseline]
```

### 4. Scaling efficiency
```
Efficiency = actual_speedup / ideal_speedup
```
- **> 80%** — good scaling
- **50–80%** — moderate
- **< 50%** — poor scaling

---

## Current configuration (bench_2node)

`bench_2node.sh` accepts `GROUP_SIZE` via `qsub -v` to sweep `--fsdp-group-size`:

```bash
# Submit a single variant:
qsub -v GROUP_SIZE=4 jobs/bench_2node.sh   # → results/bench_2node_gs4/

# Submit the full sweep (gs=0,4,8,16) in parallel:
bash jobs/submit_sweep.sh
```

| Flag | What it does |
|---|---|
| `--fsdp-group-size N` | N layers per FSDP unit. 0=whole LLM, 1=per-layer, 4/8/16=intermediate |
| `--hybrid-shard` | HSDP 2D mesh: shard within node (NVLink), replicate across nodes (IB AllReduce) |
| `--shard-strategy full` | ZeRO-3: reshard after forward — AllGather + ReduceScatter stay on NVLink |

---

## Network interface (NCCL)

**Cluster network facts (compute nodes — qhXXX):**
- IB HCAs: `mlx5_0` **and** `mlx5_1` (MT4129 ConnectX, 400 Gbps HDR each, active MTU 4096)
- IPoIB interface: `ibp56s0` (10.0.13.x/16, MTU 2044)
- GPU-NIC PCIe topology: NODE level (same NUMA node, different PCIe host bridges)

> **Important:** the login node (`qes04`) has **different** interface names (`ibs1` / `ibp35s0`,
> only one HCA). Always run network diagnostics on an actual compute node, not the login node.

`_common.sh` sets:
- `NCCL_IB_HCA=mlx5_0,mlx5_1` — stripes across both 400 Gbps links
- `NCCL_SOCKET_IFNAME=ibp56s0` — IPoIB interface for bootstrap/fallback
- `NCCL_NET_GDR_LEVEL=4` — enables GPU Direct RDMA for NODE-level PCIe topology
- `NCCL_DEBUG=INFO` — prints transport selection at startup

**Measured peak bandwidth (nccl-tests, 260419):**
- AllReduce bus bandwidth: ~80 GB/s (80% of 100 GB/s theoretical)
- The IB fabric is healthy — the bottleneck is cross-node collective overhead, not bandwidth.

---

## Bottleneck analysis (as of 260420)

### Benchmark progression

| Config | forward | backward | total | notes |
|---|---|---|---|---|
| 1-node baseline | 0.54 s | 0.80 s | 1.82 s | no cross-node comms |
| 2-node ZeRO-2 flat mesh | ~5.5–7.5 s | ~8.5–9.2 s | ~18 s | per-layer IB AllGather (fwd) + IB ReduceScatter (bwd); high variance |
| 2-node ZeRO-2 + fsdp_group_size=0 | 7.4 s | 6.0 s | 17.8 s | 1 IB AllGather in fwd (lost pipelining); backward improved |
| 2-node ZeRO-2 + hybrid_shard | 3.8 s | 7.6 s | 17.4 s | fwd AllGather on NVLink ✓; bwd IB gradient AllReduce is new bottleneck |
| 2-node ZeRO-3 + hybrid_shard | ~3–4 s | ~7–8 s | ~16 s | fwd NVLink ✓; bwd IB AllReduce still bottleneck — forward improved ~40% vs flat mesh, backward barely moved across all configs |

### Why ZeRO-3 + hybrid sharding is the right fix

With a flat 1D FSDP mesh (all 8 GPUs), every AllGather and ReduceScatter crosses IB.
With a 2D hybrid mesh (2 nodes × 4 GPUs), AllGather/ReduceScatter happen within each node over NVLink.
The only remaining IB traffic is the gradient AllReduce across the 2 replicas — unavoidable,
but it happens once per FSDP unit per backward, not per layer.

```
ZeRO-2 + hybrid:  NVLink AllGather (fwd) + IB AllReduce (bwd, full params) — bwd bottleneck
ZeRO-3 + hybrid:  NVLink AllGather (fwd) + NVLink ReduceScatter (bwd) + IB AllReduce (bwd, grad shards)
```

ZeRO-3 pays a small extra cost (re-gather params each layer) but keeps everything on NVLink.

---

## ECC hardware errors

Two nodes have confirmed uncorrectable ECC errors — hardware faults in GPU DRAM:

| Node | Discovered | How |
|---|---|---|
| qh138 | 260419 nccl-tests | Failed at 512 MB AllReduce |
| qh129 | 260419 training | Crashed at step 12, GPU 1 (rank 5) |

`_common.sh` aborts the job at startup if either node is allocated. Add new bad nodes to the
`BAD_NODES` list in `_common.sh` as they are discovered.

Report to cluster admins:
```bash
# Run on the suspect node:
nvidia-smi -q | grep -A3 "ECC Errors"
```

To exclude from PBS jobs, add to `_common.sh`'s `BAD_NODES` list (shell guard approach,
since PBS Pro does not support chaining multiple `host!=` in a single select statement).

---

## Adjusting the benchmark

All parameters are set at the top of each job script:

```bash
MAX_STEPS=20        # number of optimizer steps (20 is enough for stable timing)
MICRO_BATCH_SIZE=1  # per-GPU batch size
ACCUMULATIVE_COUNTS=4
DATASETS="handyvqa"
```

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
