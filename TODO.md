# Multinode Benchmark — TODO

Ordered by priority based on benchmark results.

Current best numbers (steady-state, max across ranks, skip first 2 steps):
- bench_1node: 1.82s/step  (baseline, 4 GPUs)
- bench_2node ZeRO-2 flat mesh:           ~18.8s/step  (4.8% eff)
- bench_2node ZeRO-2 + fsdp_group_size=0: ~17.8s/step  (fwd 7.4s, bwd 6.0s)
- bench_2node ZeRO-2 + hybrid_shard:      ~17.4s/step  (fwd 3.8s, bwd 7.6s — ECC crash)
- bench_2node ZeRO-3 + hybrid gs=0:       17.05s/step  (fwd 3.94s, bwd 8.57s, CV 7.8%,  eff 10.7%)
- bench_2node ZeRO-3 + hybrid gs=1:       16.34s/step  (fwd 4.14s, bwd 10.57s, CV 4.5%, eff 11.1%)

Quick analysis: `python3 parse_run.py results/bench_2node/<ts>/run.log`
Compare multiple: `python3 parse_run.py results/bench_2node_gs*/*/run.log`

---

## 1. fsdp_group_size sweep [NEXT — submit all at once]

**Status:** Infrastructure ready. Submit with:
```bash
bash jobs/submit_sweep.sh   # queues gs=0, 4, 8, 16 simultaneously
```
Results land in `results/bench_2node_gs<N>/`.

**What we know so far:**
- `gs=0` (1 big AllReduce): bwd=8.57s, high variance (CV 7.8%)
- `gs=1` (32 small AllReduces): bwd=10.57s, low variance (CV 3.9%), better memory (23.9 vs 46.2 GB)

**Per-collective overhead is ~0.27s** (`8.57s / ~32 effective ops` for gs=0 vs
`10.57s / 32 ops × overlap factor` for gs=1). The 32× serial overhead in gs=1
outweighs any pipelining benefit because the IB AllReduce stalls each layer's
completion before the next can start ReduceScatter.

**Expected sweet spot: gs=4 or gs=8** — fewer AllReduces (4–8 vs 32) while still
allowing some compute/comm overlap. Target: backward < 8s.

---

## 2. ECC hardware errors — report to admins [ACTION REQUIRED]

Two nodes confirmed with uncorrectable ECC errors:
- **qh138** — hit during nccl-tests (260419) at 512 MB AllReduce
- **qh129** — hit during training (260419) at step 12, GPU 1 (rank 5) — single GPU fault

Both guarded in `_common.sh`. Guard does not catch mid-run degradation.

**Action:** Report both nodes to cluster admins with job logs. Request `nvidia-smi -q | grep -A3 "ECC Errors"` per-GPU. For qh129, fault is confirmed to GPU 1 only — other GPUs may be healthy.

---

## 3. Step time variance [MEDIUM — monitor]

- gs=0: CV 7.8% (some steps 15–20s)
- gs=1: CV 3.9% (very consistent ~16s)

More IB AllReduces → higher variance? Or gs=0's single large AllReduce hits IB
congestion spikes? Re-measure after sweep.

---

## 4. Data I/O [LOW]

`data_time` ~0.02s/step. Not a bottleneck. No action needed.

---

## Completed

- **ZeRO-3 + hybrid, gs=1 (per-layer)** — 260420_105317. bwd=10.57s, worse than gs=0. Per-layer IB AllReduce serial overhead (×32) outweighs pipelining.
- **ZeRO-3 + hybrid, gs=0** — 260420_102244. 17.05s/step, bwd=8.57s. Single blocking IB AllReduce; no compute overlap but only 1× overhead.
- **ZeRO-2 + hybrid_shard** — Forward on NVLink ✓ (3.8s); backward bottlenecked by IB gradient AllReduce under ZeRO-2.
- **FSDP layer grouping (`--fsdp-group-size 0`)** — Reduces per-layer AllGathers to 1. Net: ~1s improvement.
- **Hybrid sharding (`--hybrid-shard`)** — HSDP 2D mesh. Forward time halved (7.4s → 3.8s).
- **MAX_STEPS=20** — jobs finish in ~4 min.
- **parse_run.py** — fast single-file log analyzer; max-across-ranks wall-clock, mean/std/CV, efficiency vs baseline, comparison table.
- **submit_sweep.sh** — submits multiple GROUP_SIZE variants in parallel.
- **ECC guard in `_common.sh`** — aborts if qh138/qh129 allocated.
- **NCCL:** IB RDMA + GDR confirmed. 80 GB/s peak (fabric healthy). Bottleneck is collective overhead, not bandwidth.
