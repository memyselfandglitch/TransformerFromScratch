#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

// Keep the benchmark implementation backend-neutral. HIP supports the CUDA
// kernel language; these aliases adapt only the runtime and BLAS API names.
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDestroy hipEventDestroy
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaFree hipFree

#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define cublasHandle_t hipblasHandle_t
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSgemm hipblasSgemm
#define CUBLAS_OP_N HIPBLAS_OP_N
#else
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error = (call);                                              \
        if (error != cudaSuccess) {                                              \
            std::cerr << "GPU runtime error at " << __FILE__ << ':'           \
                      << __LINE__                                               \
                      << ": " << cudaGetErrorString(error) << '\n';             \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (false)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = (call);                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "GPU BLAS error at " << __FILE__ << ':' << __LINE__   \
                      << ": status " << static_cast<int>(status) << '\n';       \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (false)

namespace {

constexpr int kBasicTile = 32;

#if defined(__HIP_PLATFORM_AMD__)
constexpr const char* kBackendName = "HIP/ROCm";
constexpr const char* kBlasName = "hipBLAS";
#else
constexpr const char* kBackendName = "CUDA";
constexpr const char* kBlasName = "cuBLAS";
#endif

// Step 1: consecutive x-threads compute different rows. In row-major storage,
// their B and C addresses are far apart, so warp memory accesses are uncoalesced.
__global__ void gemm_basic(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Step 2: consecutive x-threads now compute adjacent columns. For each k, the
// warp reads adjacent B values and writes adjacent C values.
__global__ void gemm_coalesced(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Step 3: each thread block cooperatively loads one A tile and one B tile into
// shared memory. Every loaded value is reused by multiple threads.
__global__ void gemm_shared_tiled(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    __shared__ float A_tile[kBasicTile][kBasicTile];
    __shared__ float B_tile[kBasicTile][kBasicTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kBasicTile + ty;
    const int col = blockIdx.x * kBasicTile + tx;

    float sum = 0.0f;
    const int tile_count = (K + kBasicTile - 1) / kBasicTile;

    for (int tile = 0; tile < tile_count; ++tile) {
        const int A_col = tile * kBasicTile + tx;
        const int B_row = tile * kBasicTile + ty;

        A_tile[ty][tx] = (row < M && A_col < K)
                             ? A[row * K + A_col]
                             : 0.0f;
        B_tile[ty][tx] = (B_row < K && col < N)
                             ? B[B_row * N + col]
                             : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kBasicTile; ++k) {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

constexpr int k1dBlockM = 64;
constexpr int k1dBlockN = 64;
constexpr int k1dBlockK = 8;
constexpr int k1dThreadM = 8;
constexpr int k1dThreads = k1dBlockM * k1dBlockN / k1dThreadM;

// Step 4: each thread computes eight output rows for one output column. The B
// value is held in a register and reused across all eight accumulators.
__global__ void gemm_register_1d(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    __shared__ float A_tile[k1dBlockM][k1dBlockK];
    __shared__ float B_tile[k1dBlockK][k1dBlockN];

    const int tid = threadIdx.x;
    const int local_col = tid % k1dBlockN;
    const int local_row = (tid / k1dBlockN) * k1dThreadM;
    const int block_row = blockIdx.y * k1dBlockM;
    const int block_col = blockIdx.x * k1dBlockN;
    float sums[k1dThreadM] = {0.0f};

    for (int tile_k = 0; tile_k < K; tile_k += k1dBlockK) {
        const int A_row = tid / k1dBlockK;
        const int A_col = tid % k1dBlockK;
        const int B_row = tid / k1dBlockN;
        const int B_col = tid % k1dBlockN;

        A_tile[A_row][A_col] =
            (block_row + A_row < M && tile_k + A_col < K)
                ? A[(block_row + A_row) * K + tile_k + A_col]
                : 0.0f;
        B_tile[B_row][B_col] =
            (tile_k + B_row < K && block_col + B_col < N)
                ? B[(tile_k + B_row) * N + block_col + B_col]
                : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < k1dBlockK; ++k) {
            const float B_value = B_tile[k][local_col];
#pragma unroll
            for (int r = 0; r < k1dThreadM; ++r) {
                sums[r] += A_tile[local_row + r][k] * B_value;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < k1dThreadM; ++r) {
        const int row = block_row + local_row + r;
        const int col = block_col + local_col;
        if (row < M && col < N) {
            C[row * N + col] = sums[r];
        }
    }
}

constexpr int k2dBlockM = 128;
constexpr int k2dBlockN = 128;
constexpr int k2dBlockK = 8;
constexpr int k2dThreadM = 8;
constexpr int k2dThreadN = 8;
constexpr int k2dThreads =
    (k2dBlockM / k2dThreadM) * (k2dBlockN / k2dThreadN);

template <bool VectorizedLoads>
__global__ void gemm_register_2d(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    __shared__ float A_tile[k2dBlockM][k2dBlockK];
    __shared__ float B_tile[k2dBlockK][k2dBlockN];

    const int tid = threadIdx.x;
    const int thread_tiles_per_row = k2dBlockN / k2dThreadN;
    const int local_row = (tid / thread_tiles_per_row) * k2dThreadM;
    const int local_col = (tid % thread_tiles_per_row) * k2dThreadN;
    const int block_row = blockIdx.y * k2dBlockM;
    const int block_col = blockIdx.x * k2dBlockN;
    float sums[k2dThreadM][k2dThreadN] = {0.0f};

    for (int tile_k = 0; tile_k < K; tile_k += k2dBlockK) {
        if constexpr (!VectorizedLoads) {
            for (int index = tid; index < k2dBlockM * k2dBlockK;
                 index += blockDim.x) {
                const int row = index / k2dBlockK;
                const int col = index % k2dBlockK;
                A_tile[row][col] =
                    (block_row + row < M && tile_k + col < K)
                        ? A[(block_row + row) * K + tile_k + col]
                        : 0.0f;
            }
            for (int index = tid; index < k2dBlockK * k2dBlockN;
                 index += blockDim.x) {
                const int row = index / k2dBlockN;
                const int col = index % k2dBlockN;
                B_tile[row][col] =
                    (tile_k + row < K && block_col + col < N)
                        ? B[(tile_k + row) * N + block_col + col]
                        : 0.0f;
            }
        } else {
            // Step 6: each thread moves four adjacent floats per instruction.
            // Misaligned or boundary tiles fall back to four scalar accesses.
            const int A_vector_index = tid * 4;
            const int A_row = A_vector_index / k2dBlockK;
            const int A_col = A_vector_index % k2dBlockK;
            float4 A_values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (block_row + A_row < M && tile_k + A_col + 3 < K &&
                K % 4 == 0) {
                A_values = *reinterpret_cast<const float4*>(
                    &A[(block_row + A_row) * K + tile_k + A_col]);
            } else if (block_row + A_row < M) {
                float* values = reinterpret_cast<float*>(&A_values);
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    if (tile_k + A_col + x < K) {
                        values[x] = A[(block_row + A_row) * K +
                                      tile_k + A_col + x];
                    }
                }
            }
            *reinterpret_cast<float4*>(&A_tile[A_row][A_col]) = A_values;

            const int B_vector_index = tid * 4;
            const int B_row = B_vector_index / k2dBlockN;
            const int B_col = B_vector_index % k2dBlockN;
            float4 B_values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (tile_k + B_row < K && block_col + B_col + 3 < N &&
                N % 4 == 0) {
                B_values = *reinterpret_cast<const float4*>(
                    &B[(tile_k + B_row) * N + block_col + B_col]);
            } else if (tile_k + B_row < K) {
                float* values = reinterpret_cast<float*>(&B_values);
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    if (block_col + B_col + x < N) {
                        values[x] = B[(tile_k + B_row) * N +
                                      block_col + B_col + x];
                    }
                }
            }
            *reinterpret_cast<float4*>(&B_tile[B_row][B_col]) = B_values;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < k2dBlockK; ++k) {
            float A_values[k2dThreadM];
            float B_values[k2dThreadN];
#pragma unroll
            for (int r = 0; r < k2dThreadM; ++r) {
                A_values[r] = A_tile[local_row + r][k];
            }
#pragma unroll
            for (int c = 0; c < k2dThreadN; ++c) {
                B_values[c] = B_tile[k][local_col + c];
            }
#pragma unroll
            for (int r = 0; r < k2dThreadM; ++r) {
#pragma unroll
                for (int c = 0; c < k2dThreadN; ++c) {
                    sums[r][c] += A_values[r] * B_values[c];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < k2dThreadM; ++r) {
#pragma unroll
        for (int c = 0; c < k2dThreadN; ++c) {
            const int row = block_row + local_row + r;
            const int col = block_col + local_col + c;
            if (row < M && col < N) {
                C[row * N + col] = sums[r][c];
            }
        }
    }
}

enum class Kernel {
    Basic,
    Coalesced,
    SharedTiled,
    Register1D,
    Register2D,
    Vectorized,
};

void launch_kernel(Kernel kernel, const float* A, const float* B, float* C,
                   int M, int N, int K) {
    switch (kernel) {
        case Kernel::Basic: {
            const dim3 block(kBasicTile, kBasicTile);
            const dim3 grid((M + block.x - 1) / block.x,
                            (N + block.y - 1) / block.y);
            gemm_basic<<<grid, block>>>(A, B, C, M, N, K);
            break;
        }
        case Kernel::Coalesced: {
            const dim3 block(kBasicTile, kBasicTile);
            const dim3 grid((N + block.x - 1) / block.x,
                            (M + block.y - 1) / block.y);
            gemm_coalesced<<<grid, block>>>(A, B, C, M, N, K);
            break;
        }
        case Kernel::SharedTiled: {
            const dim3 block(kBasicTile, kBasicTile);
            const dim3 grid((N + kBasicTile - 1) / kBasicTile,
                            (M + kBasicTile - 1) / kBasicTile);
            gemm_shared_tiled<<<grid, block>>>(A, B, C, M, N, K);
            break;
        }
        case Kernel::Register1D: {
            const dim3 grid((N + k1dBlockN - 1) / k1dBlockN,
                            (M + k1dBlockM - 1) / k1dBlockM);
            gemm_register_1d<<<grid, k1dThreads>>>(A, B, C, M, N, K);
            break;
        }
        case Kernel::Register2D: {
            const dim3 grid((N + k2dBlockN - 1) / k2dBlockN,
                            (M + k2dBlockM - 1) / k2dBlockM);
            gemm_register_2d<false><<<grid, k2dThreads>>>(A, B, C, M, N, K);
            break;
        }
        case Kernel::Vectorized: {
            const dim3 grid((N + k2dBlockN - 1) / k2dBlockN,
                            (M + k2dBlockM - 1) / k2dBlockM);
            gemm_register_2d<true><<<grid, k2dThreads>>>(A, B, C, M, N, K);
            break;
        }
    }
}

float time_gpu_operation(const std::function<void()>& operation,
                         int warmups, int repeats) {
    for (int run = 0; run < warmups; ++run) {
        operation();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int run = 0; run < repeats; ++run) {
        operation();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / repeats;
}

struct ErrorSummary {
    float max_absolute = 0.0f;
    float max_relative = 0.0f;
    bool passed = true;
};

ErrorSummary compare_results(const std::vector<float>& actual,
                             const std::vector<float>& reference) {
    constexpr float absolute_tolerance = 1e-2f;
    constexpr float relative_tolerance = 1e-2f;
    ErrorSummary summary;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float absolute = std::abs(actual[i] - reference[i]);
        const float relative = absolute / std::max(std::abs(reference[i]), 1e-6f);
        summary.max_absolute = std::max(summary.max_absolute, absolute);
        summary.max_relative = std::max(summary.max_relative, relative);
        if (absolute > absolute_tolerance +
                           relative_tolerance * std::abs(reference[i])) {
            summary.passed = false;
        }
    }
    return summary;
}

struct Result {
    std::string name;
    float milliseconds;
    double gflops;
    ErrorSummary error;
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [M N K repeats]\n"
              << "Example: " << program << " 1024 1024 1024 20\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return 0;
    }
    if (argc != 1 && argc != 5) {
        print_usage(argv[0]);
        return 1;
    }

    int M = 512;
    int N = 512;
    int K = 512;
    int repeats = 20;
    if (argc == 5) {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
        repeats = std::stoi(argv[4]);
    }
    if (M <= 0 || N <= 0 || K <= 0 || repeats <= 0) {
        std::cerr << "M, N, K, and repeats must all be positive.\n";
        return 1;
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::cerr << "[INFO] Backend: " << kBackendName << '\n';
    std::cerr << "[INFO] GPU: " << properties.name << '\n';
    std::cerr << "[INFO] GEMM: (" << M << "x" << K << ") * ("
              << K << "x" << N << "), repeats=" << repeats << '\n';

    const std::size_t A_count = static_cast<std::size_t>(M) * K;
    const std::size_t B_count = static_cast<std::size_t>(K) * N;
    const std::size_t C_count = static_cast<std::size_t>(M) * N;
    std::vector<float> host_A(A_count);
    std::vector<float> host_B(B_count);
    std::vector<float> host_C(C_count);
    std::vector<float> reference_C(C_count);

    std::mt19937 generator(2026);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::generate(host_A.begin(), host_A.end(),
                  [&] { return distribution(generator); });
    std::generate(host_B.begin(), host_B.end(),
                  [&] { return distribution(generator); });

    float* device_A = nullptr;
    float* device_B = nullptr;
    float* device_C = nullptr;
    CUDA_CHECK(cudaMalloc(&device_A, A_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_B, B_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_C, C_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_A, host_A.data(), A_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_B, host_B.data(), B_count * sizeof(float),
                          cudaMemcpyHostToDevice));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
#if !defined(__HIP_PLATFORM_AMD__)
    // Keep NVIDIA's reference on the strict FP32 path instead of TF32.
    CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH));
#endif
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto launch_cublas = [&] {
        // The BLAS API is column-major. This computes C^T = B^T A^T while the
        // underlying buffers remain ordinary row-major A, B, and C matrices.
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha,
                                 device_B, N, device_A, K, &beta, device_C, N));
    };

    std::cerr << "[INFO] Computing the " << kBlasName
              << " correctness reference\n";
    launch_cublas();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(reference_C.data(), device_C,
                          C_count * sizeof(float), cudaMemcpyDeviceToHost));

    const std::vector<std::pair<std::string, Kernel>> kernels = {
        {"1 basic GPU", Kernel::Basic},
        {"2 coalesced", Kernel::Coalesced},
        {"3 shared-memory tiled", Kernel::SharedTiled},
        {"4 1D register tile", Kernel::Register1D},
        {"5 2D register tile", Kernel::Register2D},
        {"6 vectorized loads", Kernel::Vectorized},
    };
    std::vector<Result> results;
    constexpr int warmups = 2;

    for (const auto& [name, kernel] : kernels) {
        std::cerr << "[INFO] Benchmarking " << name << '\n';
        auto operation = [&] {
            launch_kernel(kernel, device_A, device_B, device_C, M, N, K);
        };
        const float milliseconds =
            time_gpu_operation(operation, warmups, repeats);
        CUDA_CHECK(cudaMemcpy(host_C.data(), device_C,
                              C_count * sizeof(float), cudaMemcpyDeviceToHost));
        const ErrorSummary error = compare_results(host_C, reference_C);
        const double operations = 2.0 * M * N * K;
        const double gflops = operations / (milliseconds * 1.0e6);
        results.push_back({name, milliseconds, gflops, error});
    }

    std::cerr << "[INFO] Benchmarking " << kBlasName << '\n';
    const float cublas_ms = time_gpu_operation(launch_cublas, warmups, repeats);
    const double cublas_gflops = 2.0 * M * N * K / (cublas_ms * 1.0e6);
    const float basic_ms = results.front().milliseconds;

    std::cout << '\n'
              << std::left << std::setw(27) << "approach"
              << std::right << std::setw(12) << "mean ms"
              << std::setw(14) << "GFLOP/s"
              << std::setw(12) << "speedup"
              << std::setw(14) << "max abs err"
              << std::setw(10) << "valid" << '\n';
    std::cout << std::string(89, '-') << '\n';
    std::cout << std::fixed << std::setprecision(3);
    for (const Result& result : results) {
        std::cout << std::left << std::setw(27) << result.name
                  << std::right << std::setw(12) << result.milliseconds
                  << std::setw(14) << result.gflops
                  << std::setw(11) << basic_ms / result.milliseconds << 'x'
                  << std::scientific << std::setprecision(2)
                  << std::setw(14) << result.error.max_absolute
                  << std::fixed << std::setprecision(3)
                  << std::setw(10) << (result.error.passed ? "yes" : "NO")
                  << '\n';
    }
    std::cout << std::left << std::setw(27)
              << (std::string(kBlasName) + " reference")
              << std::right << std::setw(12) << cublas_ms
              << std::setw(14) << cublas_gflops
              << std::setw(11) << basic_ms / cublas_ms << 'x'
              << std::setw(14) << "-"
              << std::setw(10) << "yes" << '\n';

    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaFree(device_A));
    CUDA_CHECK(cudaFree(device_B));
    CUDA_CHECK(cudaFree(device_C));
    return 0;
}
