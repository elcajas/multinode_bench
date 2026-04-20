#!/bin/bash
# Benchmark: 2 nodes (16 GPUs total on rt_HF)
# Submit with: qsub multinode_bench/jobs/bench_2node.sh
#              from /groups/input/internvideo25/internvideo_2_5/
#PBS -N bench_2node
#PBS -q rt_HF
#PBS -l select=2
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o logs/bench_2node/

# Keep DATASETS and training config identical to bench_1node.sh for fair comparison
DATASETS="handyvqa"
MICRO_BATCH_SIZE=1
ACCUMULATIVE_COUNTS=4
EPOCHS=99
MAX_STEPS=20

# GROUP_SIZE controls --fsdp-group-size (layers per FSDP unit).
# Override at submit time:  qsub -v GROUP_SIZE=4 jobs/bench_2node.sh
#   0  = whole LLM as one unit (no per-layer AllGather, one big IB AllReduce)
#   1  = per-layer (default; max pipelining but 32x AllReduce overhead)
#   4/8/16 = intermediate grouping
# Unset = no flag passed (framework default, equivalent to 1)
if [ -n "${GROUP_SIZE+x}" ]; then
    EXTRA_ARGS="--fsdp-group-size ${GROUP_SIZE} --hybrid-shard --shard-strategy full"
    JOB_NAME="bench_2node_gs${GROUP_SIZE}"
else
    EXTRA_ARGS="--hybrid-shard --shard-strategy full"
    JOB_NAME="bench_2node"
fi
cd "$PBS_O_WORKDIR"
source "${PBS_O_WORKDIR}/_common.sh"
run_training
