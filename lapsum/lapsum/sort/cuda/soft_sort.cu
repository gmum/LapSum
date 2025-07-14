//
// Created by lstruski
//

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>


const int threads = 1024; // Number of threads per block

// Template implementation of logaddexp for numeric types.
template <typename T>
__device__ T logaddexp(T a, T b) {
    if (a > b) {
        return a + log1p(exp(b - a));
    }
    return b + log1p(exp(a - b));
}

/*
 * Function to compute the probability density function (PDF) of a Laplace distribution.
 * @param x: Input value.
 * @param alpha: Scale parameter of the Laplace distribution.
 * @return: PDF value at x.
 */
template <typename T>
__device__ T pdfLap(T x, T alpha) {
    return exp(-fabs(x / alpha)) / (2 * alpha);
}

/*
 * Function to compute the cumulative distribution function (CDF) of a Laplace distribution.
 * @param x: Input value.
 * @param alpha: Scale parameter of the Laplace distribution.
 * @return: CDF value at x.
 */
template <typename T>
__device__ T cdfLap(T x, T alpha) {
    T val = x / alpha;
    if (val <= 0) {
        return static_cast<T>(0.5) * exp(val);
    }
    return static_cast<T>(1) - static_cast<T>(0.5) * exp(-val);
}

/*
 * Function to solve the equation exp(a)/2*exp(x-s)-exp(b)/2*exp(r-x)+c=w.
 * @param r, s: Input parameters.
 * @param a, b, c, w: Equation coefficients.
 * @param alpha: Scale parameter of the Laplace distribution.
 * @return: Solution to the equation.
 */
template <typename T>
__device__ T solveExp(T r, T s, T a, T b, T c, T w, T alpha) {
    if (w > c && a > 0) {
        T diff = w - c;
        return s - alpha * log(a) + alpha * log(
            diff + sqrt(diff * diff + exp((r - s) / alpha) * a * b));
    }
    if (w == c && a > 0 && b > 0) {
        return static_cast<T>(0.5) * (r + s + alpha * (log(b) - log(a)));
    }
    if (w < c && b > 0) {
        T diff = c - w;
        return r + alpha * log(b) - alpha * log(
            diff + sqrt(diff * diff + exp((r - s) / alpha) * a * b));
    }
    return static_cast<T>(0.5) * (r + s);
}

/*
 * CUDA kernel to compute A and B tensors.
 * This kernel computes intermediate values A and B for each batch and column.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param A: Output tensor A of shape (batch_size, n).
 * @param B: Output tensor B of shape (batch_size, n).
 * @param alpha: Scale parameter of the Laplace distribution.
 */
template <typename T>
__global__ void computeABKernel(
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> A,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> B, const T alpha) {
    //batch index
    const int bs = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs < input.size(0)) {
        const int n = input.size(1);
        int idx;
        B[bs][0] = 0;
        A[bs][n - 1] = 0;
        for (int i = 0; i < n - 1; ++i) {
            B[bs][i + 1] = exp((input[bs][i] - input[bs][i + 1]) / alpha) * (1 + B[bs][i]);
            idx = n - i - 1;
            A[bs][idx - 1] = exp((input[bs][idx - 1] - input[bs][idx]) / alpha) * (1 + A[bs][idx]);
        }
    }
}


/*
 * CUDA kernel to compute the results tensor.
 * This kernel computes the final results based on input, A, B tensors.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param A: Input tensor A of shape (batch_size, n).
 * @param B: Input tensor B of shape (batch_size, n).
 * @param results: Output tensor results of shape (batch_size, n).
 * @param alpha: Scale parameter of the Laplace distribution.
 */
template <typename T>
__global__ void computeResultsKernel(
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> A,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> B,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> results, const T alpha) {
    const int bs = blockIdx.y;
    const int th = threadIdx.x;
    const int n = input.size(1);

    if (bs < input.size(0)) {
        // Ensure bs is within range
        int i = blockIdx.x * blockDim.x + th; // Global thread index for n
        int l = 0;

        while (i < n) {
            T w_val = i + 0.5;

            if (l == 0) {
                if (w_val <= 0.5 * (A[bs][0] - B[bs][0] + 1)) {
                    results[bs][i] = solveExp(
                        input[bs][0], input[bs][0], 1 + A[bs][0],
                        static_cast<T>(0), static_cast<T>(0), w_val, alpha
                    );
                    i += gridDim.x * blockDim.x; // Advance by total number of threads
                } else {
                    ++l;
                }
            }
            if (l > 0 && l < n) {
                if (0.5 * (A[bs][l - 1] - B[bs][l - 1] + 1) + static_cast<T>((l - 1) % n < w_val && w_val <= 0.5 * (A[bs][l] - B[bs][l] + 1) + static_cast<T>(l % n))) {
                    results[bs][i] = solveExp(
                        input[bs][l - 1], input[bs][l], 1 + A[bs][l],
                        1 + B[bs][l - 1], static_cast<T>(l), w_val, alpha
                    );
                    i += gridDim.x * blockDim.x; // Advance by total number of threads
                } else {
                    ++l;
                }
            }
            if (l == n) {
                if (w_val > 0.5 * (A[bs][n - 1] - B[bs][n - 1] + 1) + static_cast<T>((n - 1) % n)) {
                    results[bs][i] = solveExp(
                        input[bs][n - 1], input[bs][n - 1], static_cast<T>(0),
                        1 + B[bs][n - 1], static_cast<T>(n), w_val, alpha
                    );
                    i += gridDim.x * blockDim.x; // Advance by total number of threads
                }
            }
        }
    }
}


/*********************************************************************************************/
/*********************************************************************************************/
/*********************************************************************************************/

/**
 * Forward pass for the Soft-Soft algorithm.
 *
 * @param input:  Sorted input tensor of shape (batch_size, n).
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A vector containing the results tensor of Lap-Sort function.
 */
at::Tensor forward_cuda(at::Tensor input, const float alpha) {
    const auto batch_size = input.size(0);
    const auto n = input.size(1);

    // Allocate tensors
    auto results = at::empty_like(input);
    auto A = at::empty_like(input);
    auto B = at::empty_like(input);

    // Define thread block and grid dimensions
    const int blocks_bs = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "AB_cuda", ([&] {
        computeABKernel<scalar_t><<<blocks_bs, threads>>>(
            input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            A.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            B.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            static_cast<scalar_t>(alpha));
    }));

    const dim3 blocks_n((n + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "results_cuda", ([&] {
        computeResultsKernel<scalar_t><<<blocks_n, threads>>>(
            input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            A.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            B.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            results.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            static_cast<scalar_t>(alpha));
    }));

    return results;
}

/**
 **************************************************************************************************************
 *                                            Calculate gradients
 **************************************************************************************************************
**/


/**
 * Compute the normalization factor for a Laplace-like distribution.
 *
 * This function calculates the normalization factors in a batched and parallelized manner on the GPU.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n), sorted by value.
 * @param output: Output tensor of shape (batch_size, n).
 * @param nU: Normalization log tensor of shape (batch_size, n).
 * @param nL: Temporary tensor of shape (batch_size, n).
 * @param n: Number of elements in each batch of the sorted input array.
 * @param alpha: Scaling factor for normalization.
 */
template <typename scalar_t>
__global__ void calculation_normalization_factor_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> sorted_input,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> output,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> nU,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> nL,
    const int n, const scalar_t alpha) {
    const int& m = n;

    const int bs = blockIdx.x; // Batch index
    const int tid = threadIdx.x; // Thread index

    // Compute nU values
    if (tid == 0) {
        int ju = 0;
        while (ju < m && output[bs][ju] - sorted_input[bs][0] <= 0) {
            ++ju;
        }
        if (ju < n) {
            nU[bs][ju] = (sorted_input[bs][0] - output[bs][ju]) / alpha;
            int i = 1;
            for (int j = ju; j < m; ++j) {
                if (j > ju) {
                    nU[bs][j] = (output[bs][j - 1] - output[bs][j]) / alpha + nU[bs][j - 1];
                }
                while (i < n && output[bs][j] - sorted_input[bs][i] > 0) {
                    nU[bs][j] = logaddexp(nU[bs][j], (sorted_input[bs][i] - output[bs][j]) / alpha);
                    ++i;
                }
            }
        }
    }

    // Compute nL values
    if (tid == 1) {
        int jl = m - 1;
        while (jl > 0 && output[bs][jl] - sorted_input[bs][n - 1] >= 0) {
            --jl;
        }
        if (jl > 0) {
            nL[bs][jl] = (output[bs][jl] - sorted_input[bs][n - 1]) / alpha;
            int i = n - 2;
            for (int j = jl; j >= 0; --j) {
                if (j < jl) {
                    nL[bs][j] = (output[bs][j] - output[bs][j + 1]) / alpha + nL[bs][j + 1];
                }
                while (i >= 0 && output[bs][j] - sorted_input[bs][i] <= 0) {
                    nL[bs][j] = logaddexp(nL[bs][j], (output[bs][j] - sorted_input[bs][i]) / alpha);
                    --i;
                }
            }
        }
    }

    __syncthreads();

    // Combine nU and nL
    if (tid == 2) {
        int ju = 0;
        while (ju < m && output[bs][ju] - sorted_input[bs][0] <= 0) {
            ++ju;
        }
        int jl = m - 1;
        while (jl > 0 && output[bs][jl] - sorted_input[bs][n - 1] >= 0) {
            --jl;
        }
        int i = (ju < jl + 1) ? ju : jl + 1;
        int j = (ju < jl + 1) ? jl + 1 : ju;
        for (int l = 0; l < m; ++l) {
            if (l < i) {
                nU[bs][l] = nL[bs][l];
            } else if (l >= i && l < j) {
                nU[bs][l] = logaddexp(nL[bs][l], nU[bs][l]);
            }
        }
    }
}


/**
 * Compute the Jacobian-vector product (JVP) for mapping (x, w) -> output.
 *
 * This function calculates the gradient of the output with respect to the input, given the
 * Laplace distribution parameter alpha and vector v. The operation is batched and parallelized on GPU.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n), sorted by value.
 * @param output: Output tensor of shape (batch_size, n).
 * @param normLog: Normalization log tensor of shape (batch_size, n).
 * @param nvU: Gradient output tensor of shape (batch_size, n).
 * @param nvL: Temporary tensor of shape (batch_size, n).
 * @param sorted_v: Vector tensor of shape (batch_size, n), aligned with sorted_input.
 * @param n: Number of input elements per batch.
 * @param alpha: Scale parameter for the Laplace distribution.
 */
 template <typename scalar_t>
__global__ void solveLap_jvp_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> sorted_input,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> output,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> normLog,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> nvU,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> nvL,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> sorted_v,
    const int n, const scalar_t alpha) {
    const int& m = n;

    const int bs = blockIdx.x; // Batch index
    const int tid = threadIdx.x; // Thread index

    // Compute forward contributions to nvU
    if (tid == 0) {
        int i = 0, j = 0;
        while (i < n && output[bs][j] - sorted_input[bs][i] > 0) {
            nvU[bs][j] += exp((sorted_input[bs][i] - output[bs][j]) / alpha - normLog[bs][j]) * sorted_v[bs][i];
            ++i;
        }
        for (j = 1; j < m; ++j) {
            nvU[bs][j] = exp((output[bs][j - 1] - output[bs][j]) / alpha - (normLog[bs][j] - normLog[bs][j - 1])) * nvU[bs][j - 1];
            while (i < n && output[bs][j] - sorted_input[bs][i] > 0) {
                nvU[bs][j] += exp((sorted_input[bs][i] - output[bs][j]) / alpha - normLog[bs][j]) * sorted_v[bs][i];
                ++i;
            }
        }
    }

    // Compute backward contributions to nvL
    if (tid == 1) {
        int i = n - 1, j = m - 1;
        while (i >= 0 && output[bs][j] - sorted_input[bs][i] <= 0) {
            nvL[bs][j] += exp((output[bs][j] - sorted_input[bs][i]) / alpha - normLog[bs][j]) * sorted_v[bs][i];
            --i;
        }
        for (j = m - 2; j >= 0; --j) {
            nvL[bs][j] = exp((output[bs][j] - output[bs][j + 1]) / alpha - (normLog[bs][j] - normLog[bs][j + 1])) * nvL[bs][j + 1];
            while (i >= 0 && output[bs][j] - sorted_input[bs][i] <= 0) {
                nvL[bs][j] += exp((output[bs][j] - sorted_input[bs][i]) / alpha - normLog[bs][j]) * sorted_v[bs][i];
                --i;
            }
        }
    }

    __syncthreads();

    // Combine nvU and nvL
    if (tid == 2) {
        for (int j = 0; j < m; ++j) {
            nvU[bs][j] += nvL[bs][j];
        }
    }
}

/**
 * Jacobian-vector product (JVP) function for mapping (x, w) -> output.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n).
 * @param output: Output tensor of shape (batch_size, n).
 * @param sorted_v: Vector tensor of shape (batch_size, n), aligned with sorted_input.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, n).
 */
at::Tensor jvp_cuda(
        at::Tensor sorted_input, at::Tensor output, at::Tensor sorted_v, const float alpha) {
    // Extract dimensions
    const auto batch_size = sorted_input.size(0);
    const auto n = sorted_input.size(1);

    // Allocate output tensors
    auto normLog = at::zeros_like(output);
    auto temp = at::zeros_like(output);

    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "calculation_normalization_factor_kernel", ([&] {
        calculation_normalization_factor_kernel<scalar_t><<<batch_size, 3>>>(
            sorted_input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            normLog.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            temp.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            n, static_cast<scalar_t>(alpha));
    }));

    // Synchronize to ensure normalization computation is complete
    cudaDeviceSynchronize();

    auto grad = at::zeros_like(output);
    temp.zero_();

    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "solveLap_jvp_kernel", ([&] {
        solveLap_jvp_kernel<scalar_t><<<batch_size, 3>>>(
            sorted_input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            normLog.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            grad.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            temp.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            sorted_v.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            n, static_cast<scalar_t>(alpha));
    }));

    // Synchronize after kernel launch
    cudaDeviceSynchronize();

    return grad;
}

/**
 * CUDA Kernel: Compute the Vector-Jacobian Product (VJP) for the Laplace-like distribution.
 *
 * This kernel computes the VJP using the sorted input, output, normalization factors (normLog),
 * and the input vector (v). It uses shared memory to store intermediate results (kU).
 *
 * @param sorted_input: Input tensor of shape (batch_size, n), sorted by value.
 * @param output: Output tensor of shape (batch_size, n).
 * @param normLog: Normalization log tensor of shape (batch_size, n).
 * @param kU: Gradient output tensor of shape (batch_size, n).
 * @param kL: Temporary tensor of shape (batch_size, n).
 * @param v: Vector tensor of shape (batch_size, n).
 * @param n: Number of input elements per batch.
 * @param batch_size: Number of batch.
 * @param alpha: Scaling factor for the Laplace distribution.
 */
template <typename scalar_t>
__global__ void solveLap_vjp_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> sorted_input,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> output,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> normLog,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> kU,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> kL,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> v,
    const int n, const scalar_t alpha) {
    const int& m = n;

    const int bs = blockIdx.x; // Batch index
    const int tid = threadIdx.x; // Thread index

    // Compute forward contributions to kU
    if (tid == 0) {
        int i = 0, j = 0;
        while (i < m && output[bs][i] - sorted_input[bs][j] <= 0) {
            kU[bs][j] += exp((output[bs][i] - sorted_input[bs][j]) / alpha - normLog[bs][i]) * v[bs][i];
            ++i;
        }
        for (j = 1; j < n; ++j) {
            kU[bs][j] = exp((sorted_input[bs][j - 1] - sorted_input[bs][j]) / alpha) * kU[bs][j - 1];
            while (i < m && output[bs][i] - sorted_input[bs][j] <= 0) {
                kU[bs][j] += exp((output[bs][i] - sorted_input[bs][j]) / alpha - normLog[bs][i]) * v[bs][i];
                ++i;
            }
        }
    }

    // Compute backward contributions to kL
    if (tid == 1) {
        int i = m - 1, j = n - 1;
        while (i >= 0 && output[bs][i] - sorted_input[bs][j] > 0) {
            kL[bs][j] += exp((sorted_input[bs][j] - output[bs][i]) / alpha - normLog[bs][i]) * v[bs][i];
            --i;
        }
        for (j = n - 2; j >= 0; --j) {
            kL[bs][j] = exp((sorted_input[bs][j] - sorted_input[bs][j + 1]) / alpha) * kL[bs][j + 1];
            while (i >= 0 && output[bs][i] - sorted_input[bs][j] > 0) {
                kL[bs][j] += exp((sorted_input[bs][j] - output[bs][i]) / alpha - normLog[bs][i]) * v[bs][i];
                --i;
            }
        }
    }

    __syncthreads();

    // Combine kU and kL
    if (tid == 2) {
        for (int j = 0; j < n; ++j) {
            kU[bs][j] += kL[bs][j];
        }
    }
}


/**
 * CUDA Wrapper Function: Compute the Vector-Jacobian Product (VJP) for the Laplace-like distribution.
 *
 * This function orchestrates the CUDA kernel for computing the VJP. It handles memory allocation,
 * kernel launches, and synchronization.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n), sorted by value.
 * @param w_outputout: Output tensor of shape (batch_size, n).
 * @param v: Vector tensor of shape (batch_size, n).
 * @param alpha: Scaling factor for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, n).
 */
at::Tensor vjp_cuda(
            at::Tensor sorted_input, at::Tensor output, at::Tensor v, const float alpha) {
    // Extract dimensions
    const auto batch_size = sorted_input.size(0);
    const auto n = sorted_input.size(1);

    // Allocate output tensors
    auto normLog = at::zeros_like(output);
    auto temp = at::zeros_like(sorted_input);

    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "calculation_normalization_factor_kernel", ([&] {
        calculation_normalization_factor_kernel<scalar_t><<<batch_size, 3>>>(
            sorted_input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            normLog.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            temp.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            n, static_cast<scalar_t>(alpha));
    }));

    // Synchronize to ensure normalization computation is complete
    cudaDeviceSynchronize();

    auto grad = at::zeros_like(sorted_input);
    temp.zero_();

    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "solveLap_vjp_kernel", ([&] {
        solveLap_vjp_kernel<scalar_t><<<batch_size, 3>>>(
            sorted_input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            normLog.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            grad.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            temp.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            n, static_cast<scalar_t>(alpha));
    }));

    // Synchronize after kernel launch
    cudaDeviceSynchronize();

    return grad;
}
