#!/bin/bash
# Submit a sweep of bench_2node jobs with varying fsdp_group_size.
# Run from the project root:  bash jobs/submit_sweep.sh
#
# Each variant writes to results/bench_2node_gs<N>/ for easy identification.
# After all jobs complete, compare with:
#   python3 parse_run.py results/bench_2node_gs*/*/run.log

set -euo pipefail
cd "$(dirname "$0")/.."

# GROUP_SIZE values to sweep.
# 0  = whole model as one FSDP unit (one big IB AllReduce)
# 4  = 4 layers per group  (~8 IB AllReduces)
# 8  = 8 layers per group  (~4 IB AllReduces)
# 16 = 16 layers per group (~2 IB AllReduces)
# 1  = per-layer (already measured: ~10s backward)
GROUP_SIZES=(0 4 8 16)

for gs in "${GROUP_SIZES[@]}"; do
    job_id=$(qsub -v GROUP_SIZE="${gs}" jobs/bench_2node.sh)
    echo "Submitted GROUP_SIZE=${gs}  →  ${job_id}"
done

echo ""
echo "Monitor: qstat"
echo "Analyze after completion:"
echo "  python3 parse_run.py results/bench_2node_gs*/*/run.log"
