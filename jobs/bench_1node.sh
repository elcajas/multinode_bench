#!/bin/bash
# Benchmark: 1 node (single-node baseline)
# Submit with: qsub multinode_bench/jobs/bench_1node.sh
#              from /groups/input/internvideo25/internvideo_2_5/
#PBS -N bench_1node
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -j oe
# (no #PBS -o — _common.sh writes run.log to the timestamped output dir)

# Dataset — pick one that's available in your paths.json.
# Any size works: training stops at MAX_STEPS regardless of dataset length.
# Suggestions: ego4d_shortcap_lmdb, ek100_clipped, assembly101_clipped
DATASETS="handyvqa"

# Training config — keep identical across all bench_* scripts for fair comparison
MICRO_BATCH_SIZE=1
ACCUMULATIVE_COUNTS=4
EPOCHS=99       # large epoch count so MAX_STEPS is always the binding limit
MAX_STEPS=200   # stop after 200 optimizer steps; increase for more stable averages

# Uncomment to benchmark connector-only training (mirrors the actual multinode use case):
# EXTRA_ARGS="--freeze-vit --freeze-llm"

source "$(dirname $0)/../_common.sh"
run_training
