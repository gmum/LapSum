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
 * CUDA kernel to compute the temp tensor.
 * This kernel computes intermediate values for the temp tensor based on A and B.
 *
 * @param A: Input tensor A of shape (batch_size, n).
 * @param B: Input tensor B of shape (batch_size, n).
 * @param temp: Output tensor temp of shape (batch_size, n).
 */
template <typename T>
__global__ void computeTempKernel(
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> A,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> B,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> temp) {
    //batch index
    const int bs = blockIdx.y;
    // column index
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = A.size(1);
    if (i < n) {
        temp[bs][i] = 0.5 * (A[bs][i] - B[bs][i]) + static_cast<T>(i % n) + 0.5;
    }
}

/*
 * CUDA kernel to compute the results tensor.
 * This kernel computes the final results based on input, w, A, B tensors.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param w: Input tensor w of shape (batch_size, m).
 * @param A: Input tensor A of shape (batch_size, n).
 * @param B: Input tensor B of shape (batch_size, n).
 * @param results: Output tensor results of shape (batch_size, m).
 * @param alpha: Scale parameter of the Laplace distribution.
 */
template <typename T>
__global__ void computeResultsKernel(
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> w,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> A,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> B,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> results, const T alpha) {
    const int bs = blockIdx.y;
    const int th = threadIdx.x;
    const int n = input.size(1);
    const int m = w.size(1);

    if (bs < input.size(0)) {
        // Ensure bs is within range
        int i = blockIdx.x * blockDim.x + th; // Global thread index for m
        int l = 0;

        while (i < m) {
            T w_val = w[bs][i];

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
                if (0.5 * (A[bs][l - 1] - B[bs][l - 1] + 1) + static_cast<T>((l - 1) % n) < w_val && w_val <= 0.5 * (A[bs][l] - B[bs][l] + 1) + static_cast<T>(l % n)) {
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
 * solveLap mapping (x, w) -> b.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n).
 * @param w: Weight tensor of shape (batch_size, m). Must be sorted in ascending order as sorted_input.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: result tensor b.
 */
at::Tensor solveLap_cuda(at::Tensor sorted_input, at::Tensor w, const float alpha) {
    const auto batch_size = sorted_input.size(0);
    const auto n = sorted_input.size(1);
    const auto m = w.size(1);

    // Allocate tensors
    auto results = at::empty_like(w);

    {
        auto A = at::empty_like(sorted_input);
        auto B = at::empty_like(sorted_input);

        // Define thread block and grid dimensions
        const int blocks_bs = (batch_size + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "AB_cuda", ([&] {
            computeABKernel<scalar_t><<<blocks_bs, threads>>>(
                sorted_input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                A.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                B.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                static_cast<scalar_t>(alpha));
        }));

        const dim3 blocks_m((m + threads - 1) / threads, batch_size);

        AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "results_cuda", ([&] {
            computeResultsKernel<scalar_t><<<blocks_m, threads>>>(
                sorted_input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                w.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                A.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                B.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                results.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                static_cast<scalar_t>(alpha));
        }));
    }

    return results;
}

/**
 * Computes the difference of the CDF of the Laplace distribution at two points.
 *
 * @param a The lower bound.
 * @param b The upper bound (a < b).
 * @param alpha The scale parameter of the Laplace distribution.
 * @return The difference CDF(b, alpha) - CDF(a, alpha).
 */
template <typename scalar_t>
__device__ scalar_t diffCDF(scalar_t a, scalar_t b, scalar_t alpha) {
    if (a >= b) {
        return 0.0; // Handle error in a way suitable for your application.
    }

    if (b <= 0) {
        return -0.5 * exp(b / alpha) * expm1((a - b) / alpha);
    }
    if (a >= 0) {
        return -0.5 * exp(-a / alpha) * expm1((a - b) / alpha);
    }
    // a < 0 < b
    return -0.5 * (expm1(-b / alpha) + expm1(a / alpha));
}

/**
 * CUDA kernel to compute the double stochastic matrix mapping (x, w) -> output.
 *
 * This kernel computes a double stochastic matrix for each batch of input and output elements.
 * The matrix has dimensions (n x (m + 1)) for each batch.
 *
 * @param input Input tensor of shape (batch_size, n).
 * @param output Output tensor of shape (batch_size, m), containing output values.
 * @param dStochMat Output tensor of shape (batch_size, n, m + 1), to store the double stochastic matrix.
 * @param n Number of input elements per batch.
 * @param m Number of output elements per batch.
 * @param batch_size Number of batches.
 * @param alpha Scale parameter for the Laplace distribution.
 */
template <typename scalar_t>
__global__ void doubleStochasticMatrixKernel(
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> input,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> output,
    at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> dStochMat,
    const int n, const int m, const int batch_size, const scalar_t alpha) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * n) return;

    int bs = idx / n; // Batch index
    int i = idx % n;  // Input element index within the batch

    scalar_t x = input[bs][i];

    // Compute the first element in the row
    dStochMat[bs][i][0] = cdfLap(output[bs][0] - x, alpha);

    // Compute the differences for intermediate columns
    for (int j = 1; j < m; ++j) {
        dStochMat[bs][i][j] = diffCDF(output[bs][j - 1] - x, output[bs][j] - x, alpha);
    }

    // Compute the last element in the row
    dStochMat[bs][i][m] = cdfLap(x - output[bs][m - 1], alpha);
}

/**
 * Function to compute the double stochastic matrix mapping (x, w) -> output.
 *
 * @param input Input tensor of shape (batch_size, n).
 * @param output Output tensor of shape (batch_size, m), containing output values.
 * @param dStochMat Output tensor of shape (batch_size, n, m + 1), to store the double stochastic matrix.
 * @param alpha Scale parameter for the Laplace distribution.
 */
at::Tensor doubleStochasticMatrix_cuda(at::Tensor input, at::Tensor output, const float alpha) {
    // Extract dimensions
    const auto batch_size = input.size(0);
    const auto n = input.size(1);
    const auto m = output.size(1);

    // Allocate output tensor
    auto dStochMat = at::empty({batch_size, n, m + 1}, input.options());

    // Launch the CUDA kernel
    const int blocks = (batch_size * n + threads - 1) / threads; // Number of blocks

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "doubleStochasticMatrixKernel", ([&] {
        doubleStochasticMatrixKernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            dStochMat.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),
            n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    return dStochMat;
}

/**
 * Compute the logarithm of the CDF of the Laplace distribution at a given point.
 *
 * @param a The input value.
 * @param alpha The scale parameter of the Laplace distribution.
 * @return log(CDF(a))
 */
 template <typename scalar_t>
__device__ scalar_t logCDFl(scalar_t a, scalar_t alpha) {
    if (a <= 0) {
        return a / alpha - log(2);
    }
    return log(1 - 0.5 * exp(-a / alpha));
}

/**
 * Compute the logarithm of one minus the CDF of the Laplace distribution at a given point.
 *
 * @param b The input value.
 * @param alpha The scale parameter of the Laplace distribution.
 * @return log(1 - CDF(b))
 */
 template <typename scalar_t>
__device__ scalar_t logCDFu(scalar_t b, scalar_t alpha) {
    if (b >= 0) {
        return -b / alpha - log(2);
    }
    return log(1 - 0.5 * exp(b / alpha));
}

/**
 * Compute the logarithm of the difference between the CDFs of the Laplace distribution at two points.
 * Assumes a < b.
 *
 * @param a The lower bound.
 * @param b The upper bound (a < b).
 * @param alpha The scale parameter of the Laplace distribution.
 * @return log(CDF(b) - CDF(a))
 */
template <typename scalar_t>
__device__ scalar_t logCDF(scalar_t a, scalar_t b, scalar_t alpha) {
    if (b <= 0) {
        return b / alpha - log(2) + log(-expm1((a - b) / alpha));
    }
    if (a >= 0) {
        return -a / alpha - log(2) + log(-expm1((a - b) / alpha));
    }
    // a < 0 < b
    return log(-expm1(-b / alpha) - expm1(a / alpha)) - log(2);
}

/**
 * Compute the cross-entropy loss for a batch of input-output pairs.
 *
 * @tparam T The data type (e.g., float or double).
 * @param input Pointer to the input tensor of shape (batch_size, n).
 * @param output Pointer to the output tensor of shape (batch_size, m).
 * @param target Pointer to the target tensor of shape (batch_size, n).
 *               Each value is an index indicating the target class for the corresponding input.
 * @param loss Pointer to the output loss array of shape (batch_size).
 * @param n The number of input elements per batch.
 * @param m The number of output elements per batch.
 * @param batch_size The number of batches.
 * @param alpha The scale parameter for the Laplace distribution.
 */
template <typename scalar_t>
__global__ void lossCE_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> input,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> output,
    const at::PackedTensorAccessor32<long, 2, at::RestrictPtrTraits> target,
    at::PackedTensorAccessor32<scalar_t, 1, at::RestrictPtrTraits> loss,
    const int n, const int m, const int batch_size, const scalar_t alpha) {

    const int bs = blockIdx.y; // Batch index
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // Element index within the batch

    if (bs >= batch_size) return;

    if (i < n) {
        scalar_t batch_loss = 0;
        int t = target[bs][i];
        scalar_t input_value = input[bs][i];

        if (t == 0) {
            // First class: logCDF from -∞ to output[0]
            batch_loss -= logCDFl(output[bs][t] - input_value, alpha);
        } else if (t == n - 1) {
            // Last class: logCDF from output[m-1] to +∞
            batch_loss -= logCDFu(output[bs][t - 1] - input_value, alpha);
        } else {
            // Intermediate classes: logCDF difference
            batch_loss -= logCDF(output[bs][t - 1] - input_value, output[bs][t] - input_value, alpha);
        }

        // Atomic add to accumulate the loss for the batch
        atomicAdd(&loss[bs], batch_loss);
    }
}


/**
 * Cross-entropy loss permutation.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Weight tensor of shape (batch_size, n - 1). Output of solveLap function.
 * @param target: Indices tensor of shape (batch_size, n). Permutation of the indices for each input.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: result loss values for each batch.
 */
at::Tensor lossCE_cuda(at::Tensor input, at::Tensor output, at::Tensor target, const float alpha) {
    // Extract dimensions
    auto batch_size = input.size(0);
    auto n = input.size(1);
    auto m = output.size(1);

    TORCH_CHECK(m == n - 1, "The size of the second dimension of the 'output' should be smaller by 1 than the size of the second dimension of the 'input'");

    // Allocate output tensor
    auto loss = at::zeros({batch_size}, input.options());

    // Launch the CUDA kernel
    const dim3 blocks((n + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "lossCE_kernel", ([&] {
        lossCE_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            target.packed_accessor32<long, 2, at::RestrictPtrTraits>(),
            loss.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
            n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    return loss;
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
 * @param output: Output tensor of shape (batch_size, m).
 * @param nU: Normalization log tensor of shape (batch_size, m).
 * @param nL: Temporary tensor of shape (batch_size, m).
 * @param n: Number of elements in each batch of the sorted input array.
 * @param m: Number of elements in each batch of the output array.
 * @param alpha: Scaling factor for normalization.
 */
template <typename scalar_t>
__global__ void calculation_normalization_factor_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> sorted_input,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> output,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> nU,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> nL,
    const int n, const int m, const scalar_t alpha) {

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
 * @param output: Output tensor of shape (batch_size, m).
 * @param normLog: Normalization log tensor of shape (batch_size, m).
 * @param nvU: Gradient output tensor of shape (batch_size, m).
 * @param nvL: Temporary tensor of shape (batch_size, m).
 * @param sorted_v: Vector tensor of shape (batch_size, n), aligned with sorted_input.
 * @param n: Number of input elements per batch.
 * @param m: Number of output elements per batch.
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
    const int n, const int m, const scalar_t alpha) {

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
 * @param output: Output tensor of shape (batch_size, m).
 * @param sorted_v: Vector tensor of shape (batch_size, n), aligned with sorted_input.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, m).
 */
std::vector<at::Tensor> solveLap_jvp_cuda(
        at::Tensor sorted_input, at::Tensor output, at::Tensor sorted_v, const float alpha) {
    // Extract dimensions
    const auto batch_size = sorted_input.size(0);
    const auto n = sorted_input.size(1);
    const auto m = output.size(1);

    // Allocate output tensors
    auto normLog = at::zeros_like(output);
    auto temp = at::zeros_like(output);

    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "calculation_normalization_factor_kernel", ([&] {
        calculation_normalization_factor_kernel<scalar_t><<<batch_size, 3>>>(
            sorted_input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            normLog.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            temp.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            n, m, static_cast<scalar_t>(alpha));
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
            n, m, static_cast<scalar_t>(alpha));
    }));

    // Synchronize after kernel launch
    cudaDeviceSynchronize();

    return {grad, normLog};
}

/**
 * CUDA Kernel: Compute the Vector-Jacobian Product (VJP) for the Laplace-like distribution.
 *
 * This kernel computes the VJP using the sorted input, output, normalization factors (normLog),
 * and the input vector (v). It uses shared memory to store intermediate results (kU).
 *
 * @param sorted_input: Input tensor of shape (batch_size, n), sorted by value.
 * @param output: Output tensor of shape (batch_size, m).
 * @param normLog: Normalization log tensor of shape (batch_size, m).
 * @param kU: Gradient output tensor of shape (batch_size, n).
 * @param kL: Temporary tensor of shape (batch_size, n).
 * @param v: Vector tensor of shape (batch_size, m).
 * @param n: Number of input elements per batch.
 * @param m: Number of output elements per batch.
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
    const int n, const int m, const scalar_t alpha) {

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
 * @param w_outputout: Output tensor of shape (batch_size, m).
 * @param v: Vector tensor of shape (batch_size, m).
 * @param alpha: Scaling factor for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, n).
 */
at::Tensor solveLap_vjp_cuda(
            at::Tensor sorted_input, at::Tensor output, at::Tensor v, const float alpha) {
    // Extract dimensions
    const auto batch_size = sorted_input.size(0);
    const auto n = sorted_input.size(1);
    const auto m = output.size(1);

    TORCH_CHECK(m <= n, "Dimension of 'output' cannot be greater than dimensional of 'sorted_input'");

    // Allocate output tensors
    auto normLog = at::zeros_like(output);
    auto temp = at::zeros_like(sorted_input);

    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "calculation_normalization_factor_kernel", ([&] {
        calculation_normalization_factor_kernel<scalar_t><<<batch_size, 3>>>(
            sorted_input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            normLog.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            temp.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            n, m, static_cast<scalar_t>(alpha));
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
            n, m, static_cast<scalar_t>(alpha));
    }));

    // Synchronize after kernel launch
    cudaDeviceSynchronize();

    return grad;
}

/**
 * Compute the ratio (PDF(b, alpha) - PDF(a, alpha)) / (CDF(b, alpha) - CDF(a, alpha)).
 *
 * Assumes a < b.
 *
 * @tparam T Numeric type (e.g., float, double).
 * @param a The lower bound.
 * @param b The upper bound.
 * @param alpha The scale parameter of the Laplace distribution.
 * @return The computed ratio.
 */
template <typename scalar_t>
__device__ scalar_t PDFdivCDF(scalar_t a, scalar_t b, scalar_t alpha) {
    if (b <= 0) {
        return static_cast<scalar_t>(1);
    }
    if (a >= 0) {
        return static_cast<scalar_t>(-1);
    }
    // a < 0 < b
    scalar_t exp_a = expm1(a / alpha);
    scalar_t exp_neg_b = expm1(-b / alpha);
    return (exp_a - exp_neg_b) / (exp_a + exp_neg_b);
}

/**
 * Compute the ratio PDF(a, alpha) / (CDF(b, alpha) - CDF(a, alpha)).
 *
 * Assumes a < b.
 *
 * @tparam T Numeric type (e.g., float, double).
 * @param a The lower bound.
 * @param b The upper bound.
 * @param alpha The scale parameter of the Laplace distribution.
 * @return The computed ratio.
 */
template <typename scalar_t>
__device__ scalar_t PDFadivCDF(scalar_t a, scalar_t b, scalar_t alpha) {
    if (b <= 0) {
        return -exp((a - b) / alpha) / expm1((a - b) / alpha);
    }
    if (a >= 0) {
        return static_cast<scalar_t>(-1) / expm1((a - b) / alpha);
    }
    // a < 0 < b
    scalar_t exp_a = exp(a / alpha);
    scalar_t exp_neg_b = expm1(-b / alpha);
    scalar_t exp_a_over_alpha = expm1(a / alpha);
    return -exp_a / (exp_neg_b + exp_a_over_alpha);
}

/**
 * Compute the ratio PDF(b, alpha) / (CDF(b, alpha) - CDF(a, alpha)).
 *
 * Assumes a < b.
 *
 * @tparam T Numeric type (e.g., float, double).
 * @param a The lower bound.
 * @param b The upper bound.
 * @param alpha The scale parameter of the Laplace distribution.
 * @return The computed ratio.
 */
template <typename scalar_t>
__device__ scalar_t PDFbdivCDF(scalar_t a, scalar_t b, scalar_t alpha) {
    if (b <= 0) {
        return static_cast<scalar_t>(-1) / expm1((a - b) / alpha);
    }
    if (a >= 0) {
        return -exp((a - b) / alpha) / expm1((a - b) / alpha);
    }
    // a < 0 < b
    scalar_t exp_neg_b = exp(-b / alpha);
    scalar_t exp_neg_b_expm1 = expm1(-b / alpha);
    scalar_t exp_a_over_alpha = expm1(a / alpha);
    return -exp_neg_b / (exp_neg_b_expm1 + exp_a_over_alpha);
}

/*
 * Auxiliary function to compute a value based on the input x.
 * @param x: Input value.
 * @return: Computed value based on the input x.
 */
template <typename scalar_t>
__device__ scalar_t auxiliary_function(scalar_t x) {
    return (x > 0.5) ? (1 - x) / x : static_cast<scalar_t>(1);
}

/**
 * Compute the partial derivatives of the loss function with respect to input, output, and alpha.
 *
 * @tparam T The data type (e.g., float, double).
 * @param input Pointer to the input tensor of shape (batch_size, n).
 * @param output Pointer to the output tensor of shape (batch_size, m).
 * @param target Pointer to the target tensor of shape (batch_size, n).
 * @param D_input Pointer to the gradient tensor w.r.t. input (batch_size, n).
 * @param D_output Pointer to the gradient tensor w.r.t. output (batch_size, m).
 * @param D_alpha Reference to the gradient w.r.t. alpha, tensor of shape (batch_size).
 * @param n Number of input elements per batch.
 * @param m Number of output elements per batch.
 * @param batch_size Number of batches.
 * @param alpha Scale parameter for the Laplace distribution.
 */
template <typename scalar_t>
__global__ void derivative_lossCE_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> input,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> output,
    const at::PackedTensorAccessor32<long, 2, at::RestrictPtrTraits> target,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> D_input,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> D_output,
    at::PackedTensorAccessor32<scalar_t, 1, at::RestrictPtrTraits> D_alpha,
    const int n, const int m, const int batch_size, const scalar_t alpha) {

    const int bs = blockIdx.y; // Batch index
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // Element index within the batch
    if (bs >= batch_size) return;

    if (i < n) {
        scalar_t r, a, b, diff, pdf_a_div_cdf, pdf_b_div_cdf, pdf_div_cdf, common_term, extra_term;
        int target_index = target[bs][i];
        r = input[bs][i];

        if (target_index == 0) {
            // First class: derivative of -log(CDF(output[0] - r, alpha))
            diff = output[bs][0] - r;
            a = cdfLap(diff, alpha);
            b = auxiliary_function(a);

            atomicAdd(&D_input[bs][i], b / alpha);
            atomicAdd(&D_output[bs][0], -b / alpha);
            atomicAdd(&D_alpha[bs], diff * b / (alpha * alpha));
        } else if (target_index == n - 1) {
            // Last class: derivative of -log(1 - CDF(r - output[m-1], alpha))
            diff = r - output[bs][target_index - 1];
            a = cdfLap(diff, alpha);
            b = auxiliary_function(a);

            atomicAdd(&D_input[bs][i], -b / alpha);
            atomicAdd(&D_output[bs][target_index - 1], b / alpha);
            atomicAdd(&D_alpha[bs], diff * b / (alpha * alpha));
        } else {
            // Intermediate classes: derivative of -log(CDF(b - r, alpha) - CDF(a - r, alpha))
            a = output[bs][target_index - 1];
            b = output[bs][target_index];

            pdf_a_div_cdf = PDFadivCDF(a - r, b - r, alpha);
            pdf_b_div_cdf = PDFbdivCDF(a - r, b - r, alpha);
            pdf_div_cdf = PDFdivCDF(a - r, b - r, alpha);

            common_term = (b - r) * pdf_div_cdf / (alpha * alpha);
            extra_term = (b - a) * pdf_a_div_cdf / (alpha * alpha);

            atomicAdd(&D_input[bs][i], pdf_div_cdf / alpha);
            atomicAdd(&D_output[bs][target_index - 1], pdf_a_div_cdf / alpha);
            atomicAdd(&D_output[bs][target_index], -pdf_b_div_cdf / alpha);
            atomicAdd(&D_alpha[bs], common_term + extra_term);
        }
    }
}

/**
 * Compute the partial derivatives of the loss CE function with respect to input, output, and alpha.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Weight tensor of shape (batch_size, n - 1). Output of solveLap function.
 * @param target: Indices tensor of shape (batch_size, n). Permutation of the indices for each input.
 * @param alpha: Scale parameter for the Laplace distribution.
 */
std::vector<at::Tensor> derivative_lossCE_cuda(
        at::Tensor input, at::Tensor output, at::Tensor target, const float alpha) {
    // Extract dimensions
    auto batch_size = input.size(0);
    auto n = input.size(1);
    auto m = output.size(1);
    TORCH_CHECK(m == n - 1, "The size of the second dimension of the 'output' should be smaller by 1 than the size of the second dimension of the 'input'");

    // Allocate output tensor
    auto D_input = at::zeros_like(input);
    auto D_output = at::zeros_like(output);
    auto D_alpha = at::zeros({batch_size}, input.options());

    // Define the number of threads per block
    const dim3 blocks((n + threads - 1) / threads, batch_size);

    // Launch the kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "derivative_lossCE_kernel", ([&] {
        derivative_lossCE_kernel<<<blocks, threads>>>(
            input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            target.packed_accessor32<long, 2, at::RestrictPtrTraits>(),
            D_input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            D_output.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            D_alpha.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
            n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    return {D_input, D_output, D_alpha};
}
