# Multinode Benchmark — TODO

Ordered by priority based on benchmark results.

Current best numbers (260419-GDR, IB RDMA + dual HCA + GDR confirmed):
- bench_1node: 1.82s/step
- bench_2node: 18.8s/step  (4.8% scaling efficiency)
- bench_4node: 22.1s/step  (2.1% scaling efficiency)

GDR confirmed active (`GDR 1`) but made no improvement — root cause is now identified
as FSDP2 per-layer communication overhead (see item 1 below).

---

## 1. Reduce FSDP2 AllGather overhead by grouping LLM layers [CRITICAL]

**Root cause (confirmed via nccl-tests 260419):** The IB fabric is healthy — peak bus
bandwidth is ~80 GB/s (out of 100 GB/s theoretical). The training bottleneck is NOT
the network but FSDP2's per-collective overhead.

The model uses `fully_shard` (PyTorch FSDP2) with one FSDP unit per LLM layer:
- 1 AllGather for `vision_model` (~600 MB)
- 32 AllGathers for LLM decoder layers (~437 MB each)
- 1 AllGather for top-level remaining params

= **34 sequential NCCL AllGathers per forward pass**, each blocked until complete before
the next layer can compute. Each 437 MB AllGather should take ~9 ms at fabric speed,
but FSDP2 Python dispatch + CUDA synchronization overhead adds ~165 ms per collective.

```
34 collectives × 175 ms = 5.95 s forward overhead  (matches observed 6.5 − 0.54 = 5.96 s)
34 collectives × 220 ms = 7.5 s backward overhead  (matches observed AllReduce cost)
```

**Fix:** Wrap the entire LLM as a single FSDP unit instead of per-layer wrapping.
This reduces 34 AllGathers → 3 AllGathers per step:

```
3 collectives × ~500 ms = ~1.5 s (estimated forward overhead, 4× improvement)
```

**Implementation:** Add `--fsdp-group-size 0` flag to `unify_internvl2_train_r16.py`
(0 = whole LLM as one unit; 1 = current per-layer behavior).
Change is in the `bench-improvements` branch of `internvideo25_hpc`.

**Memory impact:** None — with `reshard_after_forward=False`, all 32 layers' params are
already held gathered simultaneously under the current scheme (14 GB). The single-unit
approach holds the same 14 GB.

**Verification:** After re-running with `--fsdp-group-size 0`, check that
`forward_time` drops from ~6.5 s toward ~1.5 s on 2-node.

---

## 2. ECC hardware error on qh138 [ACTION REQUIRED — report to admins]

**Finding (260419 nccl-tests):** Node `qh138` hit an uncorrectable ECC error during
the AllReduce bandwidth test at the 512 MB message size:
```
qh138: Test CUDA failure common.cu:422 'uncorrectable ECC error encountered'
```
An uncorrectable ECC error means a GPU memory cell failed — this is a hardware fault,
not a software issue. All training jobs that landed on `qh138` may have had silent
data corruption or been running with a degraded GPU.

**Action:** Report to cluster administrators with the node name and job log. Request
that qh138 be taken offline for GPU memory testing (`nvidia-smi -q | grep ECC` shows
error counts). Avoid requesting `qh138` in PBS job submissions until confirmed healthy.

---

## 3. Fix data I/O before it becomes the next bottleneck [HIGH]

**Finding:** `data_time` is 0.75 s/step on 2-node — 41% of the 1-node step time.
Currently hidden behind FSDP communication overhead; will surface once item 1 is fixed.

**Action:**
- Pre-copy dataset to local node storage (`/local1`) at job start
- Switch to LMDB format for faster random-access reads
- Reduce `NUM_WORKERS` in `_common.sh` if NFS contention persists

---

## 4. Re-check unaccounted overhead after item 1 [MEDIUM]

**Finding (260419):** ~2.1 s/step unaccounted on 2-node (11% of step time).
Likely: MPI barriers + ZeRO-2 post-optimizer parameter all-gather.

**Action:** Re-measure after FSDP grouping (item 1). If it persists above 1 s,
add timing instrumentation around the optimizer step and parameter all-gather.

---

## 5. Step time variance [LOW — monitor]

**Finding:** Step time CV is 37% on 2-node. Expected to improve after item 1
reduces the number of NCCL collectives.

---

## Completed

- **analyze.py** updated to use per-rank logs for true wall-clock step time.
- **NCCL IB RDMA confirmed** active on all runs via `NCCL_DEBUG=INFO`.
- **diag.sh** added to collect network/NCCL diagnostics from any node.
- **Two-HCA discovery:** `mlx5_0` + `mlx5_1`; fixed by setting `NCCL_IB_HCA=mlx5_0,mlx5_1`.
- **GDR confirmed active (260419-GDR):** `NCCL_NET_GDR_LEVEL=4` enabled GDR (`GDR 1`
  in logs, all channels show `GDRDMA`). No performance improvement — confirmed that
  the bottleneck is FSDP overhead, not network bandwidth.
- **nccl-tests run (260419):** Peak AllReduce bus bandwidth = 80 GB/s (80% of
  theoretical 100 GB/s). Fabric is healthy. Single 437 MB AllGather takes ~9 ms;
  FSDP2 overhead inflates this to ~175 ms per collective.
- **ECC error discovered on qh138** during nccl-tests (uncorrectable, at 512 MB
  message size). Needs hardware investigation.
