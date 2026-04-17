#!/bin/bash
# Benchmark: 4 nodes (32 GPUs total on rt_HF)
# Submit with: qsub multinode_bench/jobs/bench_4node.sh
#              from /groups/input/internvideo25/internvideo_2_5/
#PBS -N bench_4node
#PBS -q rt_HF
#PBS -l select=4
#PBS -l walltime=02:00:00
#PBS -j oe
# (no #PBS -o — _common.sh writes run.log to the timestamped output dir)

# Keep DATASETS and training config identical to bench_1node.sh for fair comparison
DATASETS="handyvqa"
MICRO_BATCH_SIZE=1
ACCUMULATIVE_COUNTS=4
EPOCHS=99
MAX_STEPS=200

# EXTRA_ARGS="--freeze-vit --freeze-llm"

source "$(dirname $0)/../_common.sh"
run_training
