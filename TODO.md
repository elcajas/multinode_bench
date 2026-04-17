# Multinode Benchmark — TODO

Ordered by priority based on benchmark results
(bench_1node: 1.82s/step, bench_2node: 18.6s/step, bench_4node: 22.0s/step).

---

## 1. Re-run benchmarks with NCCL using IB RDMA [CRITICAL — ready to test]

**Root cause:** Previous experiments had `NCCL_SOCKET_IFNAME=ibp56s0`, but that
interface does not exist on this cluster. The correct altname is `ibp35s0` (`ibs1`).
NCCL silently fell back to `bond0` — bonded 1GbE Ethernet — causing ~7.5s of
AllReduce overhead per step and 2–4% scaling efficiency.

**Cluster network facts:**
- IB HCA: `mlx5_0` (Mellanox ConnectX, active MTU 4096)
- IPoIB interface: `ibs1` / `ibp35s0` (MTU 2044)
- `bond0`: two bonded 1GbE links (MTU 1500) — previous fallback, too slow

**Fix applied in `_common.sh`:**
- Removed `NCCL_SOCKET_IFNAME=bond0` (was causing the fallback)
- Added `NCCL_IB_HCA=mlx5_0` to pin the IB HCA explicitly
- Added `NCCL_DEBUG=INFO` to confirm IB is used at runtime

**Verification:** After re-running, check `run.log` for:
```
[0] NCCL INFO Using network IB    ← correct
[0] NCCL INFO Using network Socket ← still falling back, investigate further
```

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

## 3. Investigate and reduce unaccounted overhead [MEDIUM]

**Finding:** A growing fraction of step time is not attributed to any logged phase:
- bench_1node: 0.30s (16% of step)
- bench_2node: 1.98s (11%)
- bench_4node: 3.74s (17%)

Likely sources: MPI barrier synchronization, ZeRO-2 parameter gather/scatter.
May reduce significantly after fixing NCCL (item 1).

**Action:** Re-check after item 1. If it persists, add timing instrumentation
around ZeRO gather/scatter in the training script and check `mpirun --mca` options.

---

## 4. Monitor step time variance after fixing NCCL [LOW]

**Finding:** Step time CV is 37% on 2-node and 44% on 4-node. Expected to drop
once NCCL uses IB RDMA instead of 1GbE Ethernet.

**Action:** Re-run after item 1 and check if variance drops below 15%.

---

## 5. Forward pass load imbalance [LOW / MONITOR]

**Finding:** `forward_time` std across ranks is ~6-7% — not a significant problem.

**Action:** No immediate action needed. Re-check after fixing NCCL; if imbalance
grows above 20%, tighten group-by-length bin sizes or consider sequence packing.

---

## Completed

- **analyze.py** updated to use per-rank logs for true wall-clock step time
  (max across ranks), with AllReduce overhead, load imbalance, unaccounted time,
  and prioritized recommendations.
