# Multinode Benchmark — TODO

Ordered by priority based on benchmark results.

Current best numbers (260418, IB RDMA confirmed, but only one HCA used):
- bench_1node: 1.82s/step
- bench_2node: 22.4s/step  (4.1% scaling efficiency)
- bench_4node: 27.1s/step  (1.7% scaling efficiency)

---

## 1. Re-run benchmarks with both IB HCAs enabled [CRITICAL — ready to test]

**Root cause:** Compute nodes have **two** IB HCAs (`mlx5_0` and `mlx5_1`, 400 Gbps HDR
each), but `_common.sh` was pinning to `mlx5_0` only — using half the available bandwidth.

**Discovery method:** Running `diag.sh` on an actual compute node vs the login node.
The login node (`qes04`) has different interface names (`ibs1`/`ibp35s0`, one HCA only),
which led to an incorrect diagnosis in the previous iteration.

**Fix applied in `_common.sh`:**
- Changed `NCCL_IB_HCA=mlx5_0` → `NCCL_IB_HCA=mlx5_0,mlx5_1`
- Added `NCCL_SOCKET_IFNAME=ibp56s0` (correct IPoIB interface name on compute nodes)

**Expected impact:** Doubling IB bandwidth should reduce AllReduce cost and the large
unaccounted overhead (~7-9s/step on 2-node/4-node) that appeared in the 260418 runs.

**Verification:** After re-running, check `run.log` for:
```
[0] NCCL INFO Using network IB    ← correct
```
And compare step times and unaccounted overhead against 260418 baseline.

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
