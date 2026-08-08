#!/bin/bash
#SBATCH --job-name=gemm_mi210
# This source copy lives on gn02's node-local /rhome, so keep the job on gn02.
#SBATCH --partition=jobgn02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# No --output/--error path is specified here. Slurm will create its standard
# slurm-<job-id>.out file in the directory from which sbatch is invoked. Submit
# from the gn02 benchmark directory so the source and log stay together.

set -euo pipefail

# The job is submitted from the cloned matrix_tiling directory on gn02.
BENCHMARK_DIR="${SLURM_SUBMIT_DIR}"
cd "${BENCHMARK_DIR}"

# The cluster notes list ROCm 6.0.6 but not the environment-module name.
# Try the common module names, followed by the standard /opt/rocm location.
if ! command -v hipcc >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load rocm/6.0.6 2>/dev/null || \
        module load rocm 2>/dev/null || true
fi

if ! command -v hipcc >/dev/null 2>&1 && [[ -x /opt/rocm/bin/hipcc ]]; then
    export PATH="/opt/rocm/bin:${PATH}"
    export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
fi

if ! command -v hipcc >/dev/null 2>&1; then
    echo "ERROR: hipcc was not found. Available ROCm modules follow:" >&2
    module avail 2>&1 | grep -i rocm || true
    exit 1
fi

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "cpus=${SLURM_CPUS_PER_TASK}"
echo "visible_gpu=${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-not-set}}"
echo
hipcc --version
echo
rocm-smi --showproductname --showmeminfo vram 2>/dev/null || true

# MI210 uses the CDNA2 gfx90a target. -x hip lets hipcc compile the existing
# .cu filename as HIP while retaining CUDA compatibility for nvcc users.
hipcc -O3 -std=c++17 -x hip --offload-arch=gfx90a \
    cuda_gemm_approaches.cu -lhipblas -o hip_gemm

echo
echo "Starting GEMM sweep"
for size in 256 512 1024 2048; do
    repeats=20
    if [[ "${size}" -eq 2048 ]]; then
        repeats=10
    fi

    echo
    echo "===== M=N=K=${size}, repeats=${repeats} ====="
    srun ./hip_gemm "${size}" "${size}" "${size}" "${repeats}"
done
