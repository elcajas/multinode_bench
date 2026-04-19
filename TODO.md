# Multinode Benchmark — TODO

Ordered by priority based on benchmark results.

Current best numbers (260419, IB RDMA + dual HCA confirmed, GDR still disabled):
- bench_1node: 1.82s/step
- bench_2node: 18.6s/step  (4.9% scaling efficiency)
- bench_4node: 22.1s/step  (2.1% scaling efficiency)

---

## 1. Enable GPU Direct RDMA via NCCL_NET_GDR_LEVEL=4 [CRITICAL — ready to test]

**Root cause:** GDR hardware is present (`nvidia_peermem` loaded, `/dev/gdrdrv` present)
but NCCL is NOT using it. Confirmed by `GDR 0` in NCCL logs:
```
NCCL INFO Connected all rings, use ring PXN 0 GDR 0
```

**Why:** GPU-NIC PCIe topology on compute nodes is **NODE** level (two PCIe host bridges
within the same NUMA node — confirmed via `nvidia-smi topo -m`). NCCL's default
`NCCL_NET_GDR_LEVEL=3` (PHB = single host bridge) excludes NODE-level paths, so
gradients bounce through CPU RAM instead of going GPU→NIC directly.

**Fix applied in `_common.sh`:**
- Added `NCCL_NET_GDR_LEVEL=4` (NODE) — enables GDR for the actual GPU-NIC topology

**Expected impact:** GDR eliminates the CPU-bounce path for gradient transfers.
AllReduce is currently 7.5s/step (40% of 2-node wall-clock) — should drop significantly.

**Verification:** After re-running, check `run.log` for:
```
NCCL INFO Connected all rings, use ring PXN 0 GDR 1    ← GDR active
```
And compare step times against 260419 baseline.

---

## 2. Fix data I/O before it becomes the next bottleneck [HIGH]

**Finding:** `data_time` is already 0.75s/step on 2-node and 1.58s/step on 4-node
— equal to 41% and 87% of the 1-node step time respectively. Currently hidden
behind NCCL overhead; will become the dominant bottleneck once NCCL is fixed.

**Action:**
- Pre-copy dataset to local node storage (`/local1`) at job start
- Switch to LMDB format for faster random-access reads
- If contention persists, reduce `NUM_WORKERS` in `_common.sh`

---

## 3. Investigate and reduce unaccounted overhead [MEDIUM — re-check after item 1]

**Finding:** Step time significantly exceeds the sum of logged phases:
- bench_1node/260418: ~0.34s (18% of step)
- bench_2node/260418: ~7.44s (33% of step)
- bench_4node/260418: ~9.72s (36% of step)

Likely source: ZeRO-2 post-optim parameter allgather (not timed in the training script).
Expected to improve significantly after enabling both HCAs (item 1).

**Action:** Re-check after item 1. If it persists, add timing instrumentation around
the ZeRO-2 allgather in the training script.

---

## 4. Monitor step time variance after fixing NCCL [LOW]

**Finding:** Step time CV is 29% on 2-node and 35% on 4-node (260418).
Expected to drop once both HCAs are used.

**Action:** Re-run after item 1 and check if variance drops below 15%.

---

## 5. Forward pass load imbalance [LOW / MONITOR]

**Finding:** `forward_time` std across ranks is ~5-7% — not a significant problem.

**Action:** No immediate action needed. Re-check after fixing NCCL; if imbalance
grows above 20%, tighten group-by-length bin sizes or consider sequence packing.

---

## Completed

- **analyze.py** updated to use per-rank logs for true wall-clock step time
  (max across ranks), with AllReduce overhead, load imbalance, unaccounted time,
  and prioritized recommendations.
- **NCCL IB RDMA confirmed** active on all runs via `NCCL_DEBUG=INFO`.
- **diag.sh** added to collect network/NCCL diagnostics from any node.
- **Two-HCA discovery:** compute nodes have `mlx5_0` + `mlx5_1`; previous config
  used only `mlx5_0`. Fixed by setting `NCCL_IB_HCA=mlx5_0,mlx5_1`.
- **GDR disabled discovered (260419):** `nvidia_peermem` + `/dev/gdrdrv` present but
  NCCL logs showed `GDR 0`. GPU-NIC topology is NODE level; default GDR level (PHB)
  excludes it. Fixed by adding `NCCL_NET_GDR_LEVEL=4`.
