# Multinode Benchmark — TODO

Ordered by priority based on benchmark results.

Current best numbers:
- bench_1node: 1.82s/step  (baseline, 4 GPUs)
- bench_2node ZeRO-2 flat mesh:          ~18.8s/step  (4.8% scaling efficiency)
- bench_2node ZeRO-2 + fsdp_group_size=0: ~17.8s/step  (forward 7.4s, backward 6.0s)
- bench_2node ZeRO-2 + hybrid_shard:     ~17.4s/step  (forward 3.8s, backward 7.6s — crashed at step 12 due to ECC on qh129)
- bench_2node ZeRO-3 + hybrid_shard:     ~16s/step    (forward ~3–4s, backward ~7–8s — nodes qh139+qh140, clean 20-step run 260420_102244)

---

## 1. ZeRO-3 + hybrid shard + per-layer FSDP wrapping [NEXT RUN]

**Status:** Ready to submit. Script updated: removed `--fsdp-group-size 0` from bench_2node.sh.

**Root cause identified:** With `fsdp_group_size=0` the entire model is one FSDP unit. In backward: all gradients compute first, then one massive NVLink ReduceScatter, then one massive blocking IB AllReduce. No overlap with compute possible — all ~7s is pure IB wait.

**Fix:** Restore default per-layer FSDP wrapping (no `--fsdp-group-size` flag). With ZeRO-3 + per-layer wrapping, each layer's IB AllReduce pipelines with backward compute of the next layer — the standard FSDP gradient overlap pattern.

**Expected:** Backward time significantly reduced (IB AllReduce hidden behind compute). Forward may regress slightly (more per-layer AllGathers vs zero `fsdp_group_size=0`'s single AllGather), but net total should improve.

**Submit:** `qsub jobs/bench_2node.sh` from the project root.

---

## 2. ECC hardware errors — report to admins [ACTION REQUIRED]

Two nodes confirmed with uncorrectable ECC errors:
- **qh138** — hit during nccl-tests (260419) at 512 MB AllReduce
- **qh129** — hit during training (260419) at step 12, GPU 1 (rank 5)

Both nodes are now guarded in `_common.sh` (job aborts at start if allocated). However the guard does not prevent mid-run failures if the GPU degrades after startup.

**Action:** Report both nodes to cluster admins with job logs. Request GPU memory diagnostic (`nvidia-smi -q | grep -A3 "ECC Errors"`). Avoid until confirmed repaired.

---

## 3. Data I/O [HIGH — after item 1]

`data_time` is ~0.02s/step in recent runs (group_by_length is sorting sequences well). No longer a bottleneck. Monitor after ZeRO-3 results.

---

## 4. Step time variance [MEDIUM — monitor]

Step time CV was 37% with original ZeRO-2 flat mesh. Expected to improve with ZeRO-3 + hybrid sharding (fewer cross-node collectives). Re-measure after item 1.

---

## Completed

- **FSDP layer grouping (`--fsdp-group-size 0`)** — reduces 34 per-layer LLM AllGathers to 1 top-level AllGather. Forward improved slightly but lost FSDP prefetch pipelining. Net: ~1s improvement in total.
- **Hybrid sharding (`--hybrid-shard`)** — HSDP 2D mesh (2 nodes × 4 GPUs). AllGathers stay on NVLink within each node. Forward time halved (7.4s → 3.8s). Backward regressed due to IB gradient AllReduce under ZeRO-2.
- **ZeRO-3 + hybrid sharding** — Measured 260420_102244 (qh139+qh140). ~16s/step, ~11% scaling efficiency. Modest improvement over ZeRO-2 + hybrid; backward still bottlenecked by IB AllReduce. Investigation continues (see item 1).
- **MAX_STEPS reduced to 20** — 10–20 stable steps is enough for benchmarking; jobs now finish in ~4 min.
- **ECC guard in `_common.sh`** — job aborts immediately if a known-bad node is allocated.
- **Allocated nodes printed** at job start for easy diagnosis.
- **NCCL IB RDMA confirmed** active on all runs via `NCCL_DEBUG=INFO`.
- **Two-HCA discovery:** `mlx5_0` + `mlx5_1`; fixed by setting `NCCL_IB_HCA=mlx5_0,mlx5_1`.
- **GDR confirmed active:** `NCCL_NET_GDR_LEVEL=4` enabled GDR (`GDR 1` in all channels).
- **nccl-tests (260419):** Peak AllReduce bus bandwidth = 80 GB/s. Fabric is healthy.
- **Root cause (260419):** Bottleneck is not network bandwidth but cross-node AllGather/AllReduce overhead. Hybrid sharding + ZeRO-3 is the principled fix.
