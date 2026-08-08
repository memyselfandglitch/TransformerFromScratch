# GPU GEMM layout experiment

`cuda_gemm_approaches.cu` implements the six optimization stages from
[Mini Project: How to program a GPU?](https://www.youtube.com/watch?v=GetaI7KhbzM):

1. Basic CUDA: one output element per thread with deliberately uncoalesced
   thread-to-output mapping.
2. Memory coalescing: adjacent threads access adjacent `B` and `C` values.
3. Shared-memory tiling: blocks reuse tiles of `A` and `B` from on-chip memory.
4. 1D register tiling: each thread computes eight outputs in one column.
5. 2D register tiling: each thread accumulates an `8 x 8` output tile.
6. Vectorized loading: the 2D kernel moves aligned groups of four `float`
   values with `float4` loads.

The program also runs cuBLAS (NVIDIA) or hipBLAS (AMD) as a correctness and
performance reference. All
custom kernels receive the same row-major matrices and compute the same FP32
GEMM, so only the execution and memory-access strategy changes.

## Google Colab

Select an NVIDIA GPU runtime and run:

```bash
nvcc -O3 -std=c++17 cuda_gemm_approaches.cu -lcublas -o cuda_gemm
./cuda_gemm 512 512 512 20
```

The four arguments are `M N K repeats`, corresponding to:

```text
A[M, K] * B[K, N] = C[M, N]
```

For a larger experiment:

```bash
./cuda_gemm 1024 1024 1024 20
```

The table reports kernel-only mean latency, GFLOP/s, speedup relative to the
basic kernel, maximum absolute error relative to cuBLAS, and validation status.
Host-to-device copies, initialization, validation, and logging are outside the
timed region.

## IISc AMD MI210 cluster

The MI210 nodes use ROCm rather than CUDA. Submit the included Slurm job from
the directory containing the source and script:

```bash
cd ~/matrix_tiling
sbatch run_mi210_gemm.slurm
```

Record the job ID printed by `sbatch`, then inspect it with:

```bash
squeue -j JOB_ID
tail -f /data/scratch/$USER/TransformerFromScratch/matrix_tiling/gemm_mi210_JOB_ID.out
cat /data/scratch/$USER/TransformerFromScratch/matrix_tiling/gemm_mi210_JOB_ID.err
```

Cancel a queued or running job with `scancel JOB_ID`. The job requests one GPU
from the cluster's `GPU` partition, writes logs beside the benchmark under
`/data/scratch/$USER/TransformerFromScratch/matrix_tiling`, compiles for the
MI210 `gfx90a` target, and runs all six kernels plus hipBLAS for square sizes
256, 512, 1024, and 2048.

To compile and run one case inside an interactive allocation instead:

```bash
hipcc -O3 -std=c++17 -x hip --offload-arch=gfx90a \
  cuda_gemm_approaches.cu -lhipblas -o hip_gemm
./hip_gemm 1024 1024 1024 20
```

## Why this matters for KV-cache layout

Steps 1 and 2 isolate the effect of physical address ordering. Steps 3 through
6 show how a kernel changes its tile shape, shared-memory layout, register
reuse, and vector width around that ordering. The same method can later be
applied to NHD, HND, blocked-head-major, and blocked-block-major K/V caches:
keep the attention math fixed, change the address mapping, and measure latency,
effective bandwidth, and cache behavior.
