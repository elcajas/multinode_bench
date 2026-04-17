# Multinode Benchmark — TODO

Ordered by priority based on benchmark results
(bench_1node: 1.82s/step, bench_2node: 18.6s/step, bench_4node: 22.0s/step).

---

## 1. Fix NCCL inter-node communication [CRITICAL]

**Finding:** AllReduce adds ~7.5s/step on 2-node (40% of wall-clock). Scaling efficiency
is 4.9% on 2-node and 2.1% on 4-node — multi-node is slower than single-node.

**Action:** Check if InfiniBand is available on compute nodes:
```bash
ibstat | grep "Port State"
ip link show | grep -E "^[0-9]+: (ib|mlx)"
```
If available, edit `_common.sh`:
```bash
# Change:
export NCCL_SOCKET_IFNAME=bond0
# To:
export NCCL_SOCKET_IFNAME=ib0
# Or remove the line entirely (NCCL auto-selects IB over Ethernet when present)
```
If only Ethernet is available, also try:
```bash
export NCCL_ALGO=Ring
export NCCL_NET_GDR_LEVEL=0
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

**Action:** Add timing instrumentation around ZeRO gather/scatter in the training
script. Check `mpirun --mca` options for unnecessary collective operations.

---

## 4. Fix analyze.py to use wall-clock (max rank) step time [DONE]

Per-rank log files (`rank*.log`) are now parsed and step time is computed as
`max(time across all ranks)` — the true wall-clock cost. Previously only rank 0
was used, which could underestimate by ~20% on step 1.

---

## 5. Monitor step time variance after fixing NCCL [LOW]

**Finding:** Step time coefficient of variation is 37% on 2-node and 44% on 4-node
— network is inconsistent between steps.

**Action:** Re-run benchmarks after fixing NCCL (item 1) and check if variance
drops. If it persists, investigate shared network contention with cluster admins.

---

## 6. Forward pass load imbalance [LOW / MONITOR]

**Finding:** `forward_time` std across ranks is ~6-7% — currently not a significant
problem. `--group-by-length` is working adequately.

**Action:** No immediate action needed. Re-check after fixing NCCL; if imbalance
grows above 20%, tighten group-by-length bin sizes or consider sequence packing.
