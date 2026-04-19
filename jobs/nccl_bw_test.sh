#!/bin/bash
# nccl_bw_test.sh — measure actual inter-node NCCL bandwidth using nccl-tests.
# Runs all_reduce_perf and allgather_perf across 2 nodes (8 ranks total).
# Results tell us whether the IB fabric is the bottleneck or FSDP serialization.
#
# Submit with: qsub jobs/nccl_bw_test.sh
# Results saved to: results/nccl_bw_test/<timestamp>/
#
#PBS -N nccl_bw_test
#PBS -q rt_HF
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o logs/nccl_bw_test/

BENCH_DIR="${PBS_O_WORKDIR}"
TIMESTAMP=$(date +"%y%m%d_%H%M%S")
OUTPUT_DIR="${BENCH_DIR}/results/nccl_bw_test/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "${BENCH_DIR}/logs/nccl_bw_test"

exec > >(tee "${OUTPUT_DIR}/run.log") 2>&1
echo "$(date) START — nccl bandwidth test"

# --- Environment ---
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.2
module load openmpi/4.1.7
conda activate iv25

# Same NCCL config as training jobs
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_SOCKET_IFNAME=ibp56s0
export NCCL_NET_GDR_LEVEL=4
export NCCL_DEBUG=WARN   # less verbose than INFO — we just want bandwidth numbers

# --- Node list ---
NODEFILE_UNIQUE=$(mktemp)
sort -u "$PBS_NODEFILE" > "$NODEFILE_UNIQUE"
NODES=$(wc -l < "$NODEFILE_UNIQUE")
GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
TOTAL_RANKS=$((NODES * GPUS_PER_NODE))

echo "Nodes: ${NODES}  GPUs/node: ${GPUS_PER_NODE}  Total ranks: ${TOTAL_RANKS}"
cat "$NODEFILE_UNIQUE"

# --- Build nccl-tests if needed ---
NCCL_TESTS_DIR="${BENCH_DIR}/nccl-tests"
NCCL_HOME="${CONDA_PREFIX}"   # PyTorch ships NCCL inside conda env
CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null || echo "/usr/local/cuda/bin/nvcc")))
MPI_HOME=$(dirname $(dirname $(which mpirun)))

if [ ! -f "${NCCL_TESTS_DIR}/build/all_reduce_perf" ]; then
    echo "--- Building nccl-tests ---"
    if [ ! -d "${NCCL_TESTS_DIR}" ]; then
        git clone https://github.com/NVIDIA/nccl-tests.git "${NCCL_TESTS_DIR}"
    fi
    make -C "${NCCL_TESTS_DIR}" \
        MPI=1 \
        MPI_HOME="${MPI_HOME}" \
        CUDA_HOME="${CUDA_HOME}" \
        NCCL_HOME="${NCCL_HOME}" \
        -j4
    echo "--- Build done ---"
else
    echo "nccl-tests already built at ${NCCL_TESTS_DIR}/build/"
fi

if [ ! -f "${NCCL_TESTS_DIR}/build/all_reduce_perf" ]; then
    echo "ERROR: nccl-tests build failed. Check ${OUTPUT_DIR}/run.log."
    exit 1
fi

run_test() {
    local binary="$1"
    local label="$2"
    echo ""
    echo "=============================="
    echo "TEST: ${label}"
    echo "=============================="
    mpirun \
        --np "${TOTAL_RANKS}" \
        --map-by node \
        --hostfile "${NODEFILE_UNIQUE}" \
        "${NCCL_TESTS_DIR}/build/${binary}" \
            -b 1M \
            -e 4G \
            -f 2 \
            -g 1 \
            -n 20
}

# AllReduce: measures gradient sync bandwidth (most relevant for training)
run_test all_reduce_perf "AllReduce (gradient sync)"

# AllGather: measures FSDP parameter AllGather bandwidth
run_test all_gather_perf "AllGather (FSDP param gather)"

echo ""
echo "$(date) DONE"
echo ""
echo "--- How to read the output ---"
echo "Column 'algbw' = algorithm bandwidth (data / time)"
echo "Column 'busbw' = bus bandwidth (accounts for ring factor — comparable to link speed)"
echo "For AllReduce over 8 ranks: busbw = algbw * 2*(N-1)/N = algbw * 1.75"
echo "Expected peak busbw with 2x400Gbps IB: ~90 GB/s"
echo "If busbw << 10 GB/s for large messages: IB fabric is the bottleneck"
echo "If busbw >> 10 GB/s: bottleneck is FSDP serialization (too many small collectives)"

rm -f "${NODEFILE_UNIQUE}"
