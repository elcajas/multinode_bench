#!/bin/bash
# diag.sh — collect network and NCCL diagnostic info from this node.
# Run on both the login node and a compute node and compare outputs.
#
# Usage:
#   bash diag.sh              # prints to stdout
#   bash diag.sh > diag_$(hostname).txt   # save to file

echo "======================================================"
echo "HOST: $(hostname)   DATE: $(date)"
echo "======================================================"

echo ""
echo "[1. Network interfaces]"
ip link show | grep -E "^[0-9]+: "

echo ""
echo "[2. IPoIB / IB interfaces detail]"
for iface in ibp56s0 ibp35s0 ibs1 ib0; do
    result=$(ip addr show "$iface" 2>/dev/null)
    if [ -n "$result" ]; then
        echo "--- $iface ---"
        echo "$result"
    else
        echo "$iface : not found"
    fi
done

echo ""
echo "[3. IB HCA status]"
ibstat 2>/dev/null | grep -E "hca_id|CA type|Port State|Rate|Link layer|Active MTU" \
    || echo "ibstat not available"

echo ""
echo "[4. ibv_devinfo]"
ibv_devinfo 2>/dev/null | grep -E "hca_id|port_state|active_mtu|active_speed|link_layer" \
    || echo "ibv_devinfo not available"

echo ""
echo "[5. NCCL env vars (system + current session)]"
grep -rh "NCCL" /etc/profile.d/ 2>/dev/null | sed 's/^/  [profile.d] /'
grep -rh "NCCL" /etc/environment 2>/dev/null | sed 's/^/  [environment] /'
env | grep NCCL | sed 's/^/  [env] /' || echo "  (none set)"

echo ""
echo "[6. PyTorch + NCCL version]"
python3 -c "
import torch
print('PyTorch :', torch.__version__)
print('CUDA    :', torch.version.cuda)
try:
    print('NCCL    :', torch.cuda.nccl.version())
except Exception as e:
    print('NCCL    : error -', e)
" 2>/dev/null || echo "python3/torch not available (run after: conda activate iv25)"

echo ""
echo "[7. GPU topology (NVLink / PCIe)]"
nvidia-smi topo -m 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "[8. GPU Direct RDMA (GDR)]"
# nv_peer_mem or nvidia_peermem kernel module enables GDR
lsmod 2>/dev/null | grep -E "nv_peer_mem|nvidia_peermem" \
    && echo "  GDR kernel module loaded" \
    || echo "  nv_peer_mem / nvidia_peermem NOT found — GDR likely disabled"
# GDR copy device
ls /dev/gdrdrv 2>/dev/null && echo "  /dev/gdrdrv present" || echo "  /dev/gdrdrv not present"

echo ""
echo "[9. Loaded modules]"
module list 2>&1 | grep -v "^$" || echo "(module not available)"

echo ""
echo "======================================================"
echo "DONE"
echo "======================================================"
