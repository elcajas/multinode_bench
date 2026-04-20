#!/bin/bash
# multinode_bench/_common.sh
# Shared setup for benchmark job scripts. Source this after setting DATASETS.
# Handles single-node (torchrun only) and multi-node (mpirun + torchrun) transparently.
#
# Required variable:
#   DATASETS — registry name(s) or path to a pre-generated JSON config
#
# Optional variables (all have defaults):
#   MICRO_BATCH_SIZE=1      — per-GPU batch size (kept low for benchmark consistency)
#   ACCUMULATIVE_COUNTS=4   — gradient accumulation steps
#   GPUS_PER_NODE=8         — override auto-detect (used when CUDA_VISIBLE_DEVICES is unset)
#   EPOCHS=1
#   EXTRA_ARGS=""           — any additional flags forwarded to the training script

# Derive paths relative to this script's location so the benchmark works
# wherever the parent directory is placed. Assumes internvideo25_hpc/ and
# multinode_bench/ are always siblings (both under the same parent dir).
BENCH_DIR="${PBS_O_WORKDIR}"
REPO_DIR="$(dirname "${PBS_O_WORKDIR}")/internvideo25_hpc"

cd "$REPO_DIR"

# --- Environment ---
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.2
module load openmpi/4.1.7    # provides mpirun for multi-node launch
conda activate iv25

# Prevent each torchrun worker from spawning extra OpenMP threads (avoids CPU contention)
export OMP_NUM_THREADS=1
# Disable cuBLAS LT addmm (can cause instability with some CUDA/driver versions)
export DISABLE_ADDMM_CUDA_LT=1
# Use cuDNN heuristic mode B for convolution algorithm selection
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
# Inter-node NCCL transport.
# Compute nodes have TWO IB HCAs: mlx5_0 and mlx5_1 (MT4129, 400 Gbps HDR each).
# IPoIB interface on compute nodes: ibp56s0 (10.0.13.x/16, MTU 2044).
# Note: login node (qes04) has different interface names (ibs1/ibp35s0) — do not
# use the login node to diagnose compute node network config.
#
# Pin both HCAs so NCCL can stripe AllReduce across both links (up to 800 Gbps).
# NCCL_SOCKET_IFNAME sets the IPoIB socket fallback interface (used for bootstrap
# and as fallback if IB RDMA is unavailable — not the primary data transport).
# NCCL_DEBUG=INFO prints the selected transport at startup — confirm "Using network IB".
# NCCL_NET_GDR_LEVEL: GPU-NIC PCIe topology on compute nodes is "NODE" level (two PCIe
# host bridges within the same NUMA node). NCCL's default GDR level is 3 (PHB = single
# host bridge), which silently disables GDR for NODE topology — confirmed by "GDR 0" in
# NCCL logs. Setting to 4 (NODE) enables GPU Direct RDMA so gradients go GPU→NIC directly
# instead of GPU→CPU RAM→NIC. Verify after re-run: "GDR 1" in NCCL logs.
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_SOCKET_IFNAME=ibp56s0
export NCCL_NET_GDR_LEVEL=4
export NCCL_DEBUG=INFO
# Keep Triton kernel cache local
export TRITON_CACHE_DIR="tmp/triton"
# Make repo importable without installing
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/../"
# Fixed rendezvous port — change if colliding with another job on the same node
export MASTER_PORT=34229
# Suppress TensorFlow log noise
export TF_CPP_MIN_LOG_LEVEL=3
# Prevent HuggingFace from making network requests (all weights are local)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Node and GPU detection ---
# PBS may list each node multiple times in PBS_NODEFILE (once per allocated slot).
# Deduplicate with sort -u to get the true node count.
NODEFILE_UNIQUE=$(mktemp)
sort -u "$PBS_NODEFILE" > "$NODEFILE_UNIQUE"
NODES=$(wc -l < "$NODEFILE_UNIQUE")
echo "Allocated nodes: $(tr '\n' ' ' < "$NODEFILE_UNIQUE")"

# Abort if any allocated node is known to have an uncorrectable ECC error.
# Add new bad nodes to this list as they are discovered.
BAD_NODES="qh138 qh129"
for bad in $BAD_NODES; do
    if grep -qw "$bad" "$NODEFILE_UNIQUE"; then
        echo "ERROR: allocated node $bad has a known ECC hardware fault. Aborting."
        echo "Please report $bad to cluster admins and resubmit."
        exit 1
    fi
done
MASTER_ADDR=$(head -n 1 "$NODEFILE_UNIQUE")

# GPU count auto-detection (in order of reliability):
#   1. CUDA_VISIBLE_DEVICES — set by PBS when GPUs are explicitly allocated
#   2. nvidia-smi           — always works on a GPU node
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
  GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
  GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
if [ "${GPUS_PER_NODE}" -eq 0 ]; then
  echo "ERROR: could not detect any GPUs. Check CUDA_VISIBLE_DEVICES or nvidia-smi."
  exit 1
fi

TOTAL_GPUS=$((GPUS_PER_NODE * NODES))

# --- Output directory ---
# Each run gets its own timestamped subdirectory to avoid overwriting previous results.
JOB_NAME=${JOB_NAME:-$(basename "$0" .sh)}
TIMESTAMP=$(date +"%y%m%d_%H%M%S")
OUTPUT_DIR="${BENCH_DIR}/results/${JOB_NAME}/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "${BENCH_DIR}/logs/${JOB_NAME}"

# Save a copy of this job script alongside the outputs for reproducibility
cp "$0" "${OUTPUT_DIR}/${JOB_NAME}.sh"

# Redirect all stdout/stderr to run.log while also printing to the PBS log
exec > >(tee "${OUTPUT_DIR}/run.log") 2>&1

echo "$(date +"%y%m%d_%H%M%S") START"
echo "Nodes: ${NODES}  GPUs/node: ${GPUS_PER_NODE}  Total GPUs: ${TOTAL_GPUS}"
echo "Master addr: ${MASTER_ADDR}"
echo "Output dir: ${OUTPUT_DIR}"

# --- Dataset resolution ---
# If DATASETS is a registry name (or space-separated names), generate a JSON config.
# If it already points to an existing JSON file, use it directly.
if [ ! -f "${DATASETS}" ]; then
  _datasets_json="${OUTPUT_DIR}/datasets.json"
  python3 scripts/make_train_json.py ${DATASETS} -o "${_datasets_json}"
  DATASETS="${_datasets_json}"
fi

# --- Parameter defaults ---
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
ACCUMULATIVE_COUNTS=${ACCUMULATIVE_COUNTS:-4}
NUM_WORKERS=8
EPOCHS=${EPOCHS:-1}

GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * TOTAL_GPUS * ACCUMULATIVE_COUNTS))

echo "Config: micro_batch=${MICRO_BATCH_SIZE}  accum=${ACCUMULATIVE_COUNTS}  global_batch=${GLOBAL_BATCH_SIZE}  epochs=${EPOCHS}"

# Benchmark-specific overrides:
#   - No checkpoints (we are measuring throughput, not training to convergence)
#   - Log every step for fine-grained per-step timing
#   - Stop after MAX_STEPS steps regardless of dataset size
CHECKPOINT_INTERVAL=999999
LOG_INTERVAL=1
MAX_STEPS=${MAX_STEPS:-200}    # controls benchmark duration; increase for more stable averages

# --- Training launch ---
# run_training() transparently handles single-node and multi-node:
#
#   Single-node:  torchrun --nproc_per_node=N  [train args]
#   Multi-node:   mpirun --np NODES --map-by node  torchrun --nproc_per_node=N --nnodes=NODES  [train args]
#
# --map-by node is required so mpirun places exactly one torchrun process per node.
# Without it, MPI may place all processes on the head node.
#
# The c10d rendezvous backend auto-assigns node ranks — no --node_rank needed.
#
# backward_time in the logs includes the NCCL AllReduce (gradient sync across GPUs/nodes).
# Comparing backward_time between single-node and multi-node runs reveals communication overhead.

run_training() {
  if [ "${NODES}" -gt 1 ]; then
    echo "Launch mode: MULTI-NODE (mpirun + torchrun)"
    mpirun \
      --np "${NODES}" \
      --map-by node \
      --hostfile "${NODEFILE_UNIQUE}" \
    torchrun \
      --nproc_per_node="${GPUS_PER_NODE}" \
      --nnodes="${NODES}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      --rdzv_id="${PBS_JOBID}" \
    unify_internvl2_train_r16.py \
      --model ./models/ \
      --datasets "${DATASETS}" \
      --num-workers ${NUM_WORKERS} \
      --micro-batch-size ${MICRO_BATCH_SIZE} \
      --global-batch-size ${GLOBAL_BATCH_SIZE} \
      --vit_lr 2e-6 \
      --connector_lr 1e-5 \
      --lr 1e-5 \
      --wd 0.0 \
      --use-fast-tokenizer \
      --warmup-ratio 0.03 \
      --work-dir "${OUTPUT_DIR}" \
      --log-interval ${LOG_INTERVAL} \
      --seed 42 \
      --checkpoint-interval ${CHECKPOINT_INTERVAL} \
      --checkpoint-drop-optimizer \
      --no-save \
      --shard-strategy 'zero2' \
      --group-by-length \
      --frame_sampling_method middle \
      --min_num_frames 16 \
      --max_num_frames 16 \
      --local_num_frames 1 \
      --num_tome_tokens 16 \
      --epochs ${EPOCHS} \
      --max-steps ${MAX_STEPS} \
      ${EXTRA_ARGS}
  else
    echo "Launch mode: SINGLE-NODE (torchrun)"
    torchrun \
      --nproc_per_node="${GPUS_PER_NODE}" \
      --master_port="${MASTER_PORT}" \
    unify_internvl2_train_r16.py \
      --model ./models/ \
      --datasets "${DATASETS}" \
      --num-workers ${NUM_WORKERS} \
      --micro-batch-size ${MICRO_BATCH_SIZE} \
      --global-batch-size ${GLOBAL_BATCH_SIZE} \
      --vit_lr 2e-6 \
      --connector_lr 1e-5 \
      --lr 1e-5 \
      --wd 0.0 \
      --use-fast-tokenizer \
      --warmup-ratio 0.03 \
      --work-dir "${OUTPUT_DIR}" \
      --log-interval ${LOG_INTERVAL} \
      --seed 42 \
      --checkpoint-interval ${CHECKPOINT_INTERVAL} \
      --checkpoint-drop-optimizer \
      --no-save \
      --shard-strategy 'zero2' \
      --group-by-length \
      --frame_sampling_method middle \
      --min_num_frames 16 \
      --max_num_frames 16 \
      --local_num_frames 1 \
      --num_tome_tokens 16 \
      --epochs ${EPOCHS} \
      --max-steps ${MAX_STEPS} \
      ${EXTRA_ARGS}
  fi

  echo "$(date +"%y%m%d_%H%M%S") END"
  rm -f "${NODEFILE_UNIQUE}"
}
