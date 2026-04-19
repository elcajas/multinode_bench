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

**Cluster network facts (compute nodes — qhXXX):**
- IB HCAs: `mlx5_0` **and** `mlx5_1` (MT4129 ConnectX, 400 Gbps HDR each, active MTU 4096)
- IPoIB interface: `ibp56s0` (10.0.13.x/16, MTU 2044)
- GPU-NIC PCIe topology: NODE level (same NUMA node, different PCIe host bridges)
- No `bond0` on compute nodes

> **Important:** the login node (`qes04`) has **different** interface names (`ibs1` / `ibp35s0`,
> only one HCA). Always run network diagnostics on an actual compute node, not the login node.
> Use `diag.sh` for this purpose.

`_common.sh` sets:
- `NCCL_IB_HCA=mlx5_0,mlx5_1` — stripes AllReduce across both 400 Gbps links
- `NCCL_SOCKET_IFNAME=ibp56s0` — IPoIB interface for bootstrap/fallback
- `NCCL_NET_GDR_LEVEL=4` — enables GPU Direct RDMA despite NODE-level PCIe topology
  (NCCL's default level 3/PHB silently disables GDR for this topology)
- `NCCL_DEBUG=INFO` — prints transport selection at startup

**Verify IB + GDR are active** by checking `run.log` after a job completes:
```bash
grep "Using network\|GDR" results/bench_2node/*/run.log
# Should show:
# NCCL INFO Using network IB
# NCCL INFO Connected all rings, use ring PXN 0 GDR 1
# NCCL INFO Channel 00/0 : ... via NET/IB/0/GDRDMA
```

**Measured peak bandwidth (nccl-tests, 260419):**
- AllReduce bus bandwidth: ~80 GB/s at large message sizes (80% of 100 GB/s theoretical)
- The IB fabric is healthy — the training bottleneck is NOT the network.

---

## Bottleneck analysis (as of 260419)

The multi-node scaling is poor (5% efficiency at 2-node) due to **FSDP2 per-layer
AllGather overhead**, not network bandwidth. Here is the full chain of investigation:

### What we measured

| Phase | 1-node | 2-node | Overhead |
|---|---|---|---|
| forward_time | 0.54 s | 6.5 s | +5.96 s |
| backward_time | 0.80 s | 8.33 s | +7.53 s |
| Total step | 1.82 s | 18.8 s | +17 s |

### Why forward_time is 12× slower on 2-node

The training script uses PyTorch FSDP2 (`fully_shard`) with **one FSDP unit per LLM
decoder layer**. This means every layer's forward pass triggers a separate NCCL AllGather
collective to reconstruct that layer's parameters from sharded form across all ranks.

For the InternVL2-8B model (32 LLM layers + 1 ViT + 1 top-level):
- **34 sequential NCCL AllGather collectives per forward pass**
- Each AllGather is ~437 MB and takes ~9 ms at fabric speed
- But FSDP2 Python dispatch + CUDA synchronization adds ~165 ms fixed overhead per collective
- **34 × 175 ms = 5.95 s** — matches exactly the observed forward overhead

The same applies to backward: 34 ReduceScatter collectives × ~220 ms = **7.5 s**.

### Why GDR and dual-HCA didn't help

- `NCCL_IB_HCA=mlx5_0,mlx5_1` doubled bandwidth → helped only for the 9 ms data
  transfer portion of each collective, not the 165 ms overhead
- `NCCL_NET_GDR_LEVEL=4` eliminated the CPU-bounce path → same reason: only saves on
  the data transfer, not the per-collective fixed overhead
- nccl-tests confirmed the fabric reaches 80 GB/s — the network is not the bottleneck

### Fix: FSDP layer grouping

Wrap the entire LLM as a single FSDP unit instead of per-layer wrapping:

```
Current:  34 AllGathers × 175 ms = 5.95 s forward overhead
Fixed:     3 AllGathers × ~500 ms = ~1.5 s forward overhead  (estimated 4× improvement)
```

Implementation: `--fsdp-group-size 0` flag in `unify_internvl2_train_r16.py`
(0 = whole LLM as one unit, 1 = current per-layer default).

---

## ECC hardware error on qh138

An **uncorrectable ECC error** was detected on node `qh138` during nccl-tests (260419):
```
qh138: Test CUDA failure 'uncorrectable ECC error encountered'
```

An uncorrectable ECC error means a GPU DRAM cell has permanently failed. This is a
**hardware fault**:
- Training jobs that ran on `qh138` may have produced silently corrupted results
- The GPU may pass lighter workloads but fail under memory pressure
- **Action:** Report to cluster admins and avoid `qh138` until confirmed repaired

Check ECC error counts:
```bash
nvidia-smi -q | grep -A3 "ECC Errors"
```

To exclude `qh138` from PBS jobs, add to the job script:
```bash
#PBS -l select=2:ncpus=...:host!=qh138
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
