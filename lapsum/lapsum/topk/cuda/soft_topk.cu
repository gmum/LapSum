//
// Created by lstruski
//

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cmath>


const int threads = 256; // Number of threads per block

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
    if (w < c) {
        T diff = c - w;
        return r + alpha * log(b) - alpha * log(
            diff + sqrt(diff * diff + exp((r - s) / alpha) * a * b));
    }
    return static_cast<T>(0.5) * (r + s);
}

// Helper function for warp-wide reduction
template <typename T>
__device__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


// Helper function for block-wide reduction
template <typename T>
__device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}


template <typename T>
__global__ void compute_a_sums_kernel(
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
    at::PackedTensorAccessor32<T, 1, at::RestrictPtrTraits> results,
    const int reference_idx, const int end_idx, const T alpha) {

    extern __shared__ char shared_mem[];
    T* shared_data = reinterpret_cast<T*>(shared_mem);
    T& shared_val = shared_data[0];
    T& block_results = shared_data[1];

    const int bs = blockIdx.y;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Initialize block result and load reference value
    if (tid == 0) {
        shared_val = input[bs][reference_idx];
        block_results = 0;
    }
    __syncthreads();

    // Rest of the kernel remains the same...
    // Calculate global index
    const int i = reference_idx + 1 + blockIdx.x * block_size + tid;
    T thread_sum = 0;

    if (bs < input.size(0)) {
        // Process elements with coalesced memory access
        for (int idx = i; idx <= end_idx; idx += block_size * gridDim.x) {
            const T diff = shared_val - input[bs][idx];
            thread_sum += exp(diff / alpha);
        }
    }

    // Block-wide reduction to avoid multiple atomicAdd operations
    thread_sum = blockReduceSum(thread_sum);

    if (tid == 0) {
        atomicAdd(&results[bs], thread_sum);
    }
}

// Similarly modify compute_b_sums_kernel
template <typename T>
__global__ void compute_b_sums_kernel(
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
    at::PackedTensorAccessor32<T, 1, at::RestrictPtrTraits> results,
    const int reference_idx, const int end_idx, const T alpha) {

    extern __shared__ char shared_mem[];
    T* shared_data = reinterpret_cast<T*>(shared_mem);
    T& shared_val = shared_data[0];
    T& block_results = shared_data[1];

    // Rest of the kernel remains the same...
    const int bs = blockIdx.y;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (tid == 0) {
        shared_val = input[bs][end_idx];
        block_results = 0;
    }
    __syncthreads();

    const int i = reference_idx + blockIdx.x * block_size + tid;
    T thread_sum = 0;

    if (bs < input.size(0)) {
        for (int idx = i; idx < end_idx; idx += block_size * gridDim.x) {
            const T diff = input[bs][idx] - shared_val;
            thread_sum += exp(diff / alpha);
        }
    }

    thread_sum = blockReduceSum(thread_sum);

    if (tid == 0) {
        atomicAdd(&results[bs], thread_sum);
    }
}

template <typename T>
__global__ void compute_sums_kernel(
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
    at::PackedTensorAccessor32<T, 1, at::RestrictPtrTraits> results,
    const at::PackedTensorAccessor32<T, 1, at::RestrictPtrTraits> temp,
    const int reference_idx0, const int reference_idx1, const T alpha) {

    const int bs = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs < input.size(0)) {
        const T diff = input[bs][reference_idx0] - input[bs][reference_idx1];
        const T val = exp(diff / alpha) * temp[bs];
        atomicAdd(&results[bs], val);
    }
}




template <typename T>
__device__ T compute_am(const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
                       int bs, int mid, int r, T a_e, T alpha) {
    T a_m = 0;
    for (int j = mid + 1; j <= r; ++j) {
        a_m += exp((input[bs][mid] - input[bs][j]) / alpha);
    }
    a_m += exp((input[bs][mid] - input[bs][r]) / alpha) * a_e;
    return a_m;
}

template <typename T>
__device__ T compute_bm(const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
                       int bs, int l, int mid, T b_s, T alpha) {
    T b_m = 0;
    for (int j = l; j < mid; ++j) {
        b_m += exp((input[bs][j] - input[bs][mid]) / alpha);
    }
    b_m += exp((input[bs][l] - input[bs][mid]) / alpha) * b_s;
    return b_m;
}

template <typename T>
__global__ void compute_results_kernel(
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> w,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> results,
    const at::PackedTensorAccessor32<T, 1, at::RestrictPtrTraits> a_start,
    const at::PackedTensorAccessor32<T, 1, at::RestrictPtrTraits> a_mid,
    const at::PackedTensorAccessor32<T, 1, at::RestrictPtrTraits> b_mid,
    const at::PackedTensorAccessor32<T, 1, at::RestrictPtrTraits> b_end,
    const int start, const int middle, const int end, const T alpha) {

    const int bs = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs >= input.size(0) || i >= w.size(1)) return;

    if (w[bs][i] <= 0.5 * (a_start[bs] + 1)) {
        results[bs][i] = solveExp(
            input[bs][0], input[bs][0], 1 + a_start[bs],
            static_cast<T>(0), static_cast<T>(0), w[bs][i], alpha);
    } else if (w[bs][i] >= 0.5 * (1 - b_end[bs]) + static_cast<T>(end)) {
        results[bs][i] = solveExp(
            input[bs][input.size(1)-1], input[bs][input.size(1)-1], static_cast<T>(0),
            1 + b_end[bs], static_cast<T>(input.size(1)), w[bs][i], alpha);
    } else {
        int l = start, mid = middle, r = end;
        T a_s = a_start[bs], b_s = 0;
        T a_m = a_mid[bs], b_m = b_mid[bs];
        T a_e = 0, b_e = b_end[bs];

        while (l + 1 < r) {
            if (w[bs][i] <= 0.5 * (a_m - b_m + 1) + static_cast<T>(mid)) {
                r = mid;
                a_e = a_m;
                b_e = b_m;
            } else {
                l = mid;
                a_s = a_m;
                b_s = b_m;
            }

            mid = (l + r) / 2;
            a_m = compute_am(input, bs, mid, r, a_e, alpha);
            b_m = compute_bm(input, bs, l, mid, b_s, alpha);
        }

        results[bs][i] = solveExp(
            input[bs][l], input[bs][r], 1 + a_e, 1 + b_s,
            static_cast<T>(1 + l), w[bs][i], alpha);
    }
}




template <typename T>
__global__ void probability_kernel(
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> output,
    at::PackedTensorAccessor32<T, 3, at::RestrictPtrTraits> prob, const T alpha) {
    const int bs = blockIdx.y;
    const int th = threadIdx.x;
    const int n = input.size(1);
    const int m = output.size(1);

    int index = blockIdx.x * blockDim.x + th;

    if (bs < input.size(0) && index < n * m) {
        int l = index % n;
        int j = index / n;
        prob[bs][j][l] = cdfLap(output[bs][j] - input[bs][l], alpha);
    }
}



/*********************************************************************************************/
/*********************************************************************************************/
/*********************************************************************************************/

/**
 * Forward pass for the Soft-Top-k algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param sorted_input: Sorted input tensor of shape (batch_size, n).
 * @param w: Weight tensor of shape (batch_size, m).
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A vector containing the results tensor of Lap-Sum function and the probability tensor.
 */
std::vector<at::Tensor> forward_cuda(
        at::Tensor input, at::Tensor sorted_input, at::Tensor w, const float alpha) {

    const auto batch_size = input.size(0);
    const auto n = input.size(1);
    const auto m = w.size(1);

    auto results = at::empty_like(w);

    // Allocate temporary tensors for intermediate results
    auto a_start = at::zeros({batch_size}, input.options());
    auto a_mid = at::zeros({batch_size}, input.options());
    auto b_mid = at::zeros({batch_size}, input.options());
    auto b_end = at::zeros({batch_size}, input.options());

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dispatch based on input type
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "forward_wrapper", ([&] {
        auto input_a = sorted_input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto a_mid_a = a_mid.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>();
        auto a_start_a = a_start.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>();
        auto b_mid_a = b_mid.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>();
        auto b_end_a = b_end.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>();

        // Calculate optimal block size (typically 128-256 threads)
        const int threads_per_block = threads;
        const int middle = (n - 1) / 2;
        const size_t shared_mem_size = 2 * sizeof(scalar_t); // For shared_val and block_results

        // Compute a_mid and b_end (processing elements from middle+1 to n-1)
        int elements_to_process = n - 1 - (middle + 1) + 1;
        int blocks_per_batch = (elements_to_process + threads_per_block - 1) / threads_per_block;
        dim3 grid1(blocks_per_batch, batch_size);
        compute_a_sums_kernel<<<grid1, threads_per_block, shared_mem_size, stream>>>(
            input_a, a_mid_a, middle, n - 1, static_cast<scalar_t>(alpha));
        compute_b_sums_kernel<<<grid1, threads_per_block, shared_mem_size, stream>>>(
            input_a, b_end_a, middle, n - 1, static_cast<scalar_t>(alpha));

        // Compute a_start and b_mid (processing elements from 0 to middle-1)
        elements_to_process = middle - 0;
        blocks_per_batch = (elements_to_process + threads_per_block - 1) / threads_per_block;
        dim3 grid2(blocks_per_batch, batch_size);
        compute_a_sums_kernel<<<grid2, threads_per_block, shared_mem_size, stream>>>(
            input_a, a_start_a, 0, middle, static_cast<scalar_t>(alpha));
        compute_b_sums_kernel<<<grid2, threads_per_block, shared_mem_size, stream>>>(
            input_a, b_mid_a, 0, middle, static_cast<scalar_t>(alpha));

        // Final computations
        int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
        compute_sums_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            input_a, a_start_a, a_mid_a, 0, middle, static_cast<scalar_t>(alpha));
        compute_sums_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            input_a, b_end_a, b_mid_a, middle, n - 1, static_cast<scalar_t>(alpha));



        // Compute final results using the optimized kernel
        auto w_a = w.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto results_a = results.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();

        const int blocks_per_m = (m + threads_per_block - 1) / threads_per_block;
        dim3 grid3(blocks_per_m, batch_size);

        compute_results_kernel<<<grid3, threads_per_block>>>(
            input_a, w_a, results_a, a_start_a, a_mid_a, b_mid_a, b_end_a,
            0, middle, n - 1, static_cast<scalar_t>(alpha));

    }));

    // Allocate tensors
    auto prob = at::empty({batch_size, m, n}, input.options());

    const dim3 num_blocks((n * m + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "prob_cuda", ([&] {
        probability_kernel<scalar_t><<<num_blocks, threads>>>(
            input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            results.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            prob.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),
            static_cast<scalar_t>(alpha));
    }));

    return {results, prob};
}






/**
 **************************************************************************************************************
 *                                            Calculate gradients
 **************************************************************************************************************
**/


/**
 * CUDA kernel to calculate components to Jacobian-vector product (JVP) for the Soft-Top-k algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Output tensor of shape (batch_size, m).
 * @param v: Tensor of shape (batch_size, m, n) used in the JVP computation.
 * @param sum: Temporary tensor of shape (batch_size, m).
 * @param scalar_prod: Temporary tensor of shape (batch_size, m).
 * @param scalar_prod_input: Temporary tensor of shape (batch_size, m).
 * @param alpha: Scale parameter for the Laplace distribution.
 * @param grad: Tensor of shape (batch_size, m, n) contains components to calculate gradient.
 */
template <typename T>
__global__ void derivative_components_jvp_kernel(
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> output,
    const at::PackedTensorAccessor32<T, 3, at::RestrictPtrTraits> v,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> sum,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> scalar_prod,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> scalar_prod_input,
    at::PackedTensorAccessor32<T, 3, at::RestrictPtrTraits> grad, const T alpha) {
    const int bs = blockIdx.y;
    const int n = input.size(1);
    const int m = output.size(1);

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs < input.size(0) && index < n * m) {
        int l = index % n;
        int j = index / n;

        T value = pdfLap(output[bs][j] - input[bs][l], alpha);
        grad[bs][j][l] = value;

        // Atomic addition to ensure thread safety for the sum array
        atomicAdd(&sum[bs][j], value);

        T value1 = value * input[bs][l];
        value *= v[bs][j][l];
        atomicAdd(&scalar_prod[bs][j], value);
        atomicAdd(&scalar_prod_input[bs][j], value1);
    }
}

/**
 * CUDA kernel to calculate Jacobian-vector product (JVP) for the Soft-Top-k algorithm.
 *
 * @param grad: Gradient tensor of shape (batch_size, m, n) that will be return.
 * @param grad_w: Gradient tensor of shape (batch_size, m) that will be return.
 * @param grad_alpha: Gradient tensor of shape (1) that will be return.
 * @param sum: Temporary tensor of shape (batch_size, m).
 * @param scalar_prod: Temporary tensor of shape (batch_size, m).
 * @param scalar_prod_input: Temporary tensor of shape (batch_size, m).
 * @param v: Tensor of shape (batch_size, m, n) used in the JVP computation.
 * @param input: Input tensor with shape (batch_size, n).
 * @param alpha: Scale parameter for the Laplace distribution.
 */
template <typename T>
__global__ void derivative_jvp_kernel(
    at::PackedTensorAccessor32<T, 3, at::RestrictPtrTraits> grad,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> grad_w,
    at::PackedTensorAccessor32<T, 1, at::RestrictPtrTraits> grad_alpha,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> sum,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> scalar_prod,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> scalar_prod_input,
    const at::PackedTensorAccessor32<T, 3, at::RestrictPtrTraits> v,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input, const T alpha) {
    const int bs = blockIdx.y;
    const int m = grad.size(1);
    const int n = grad.size(2);

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs < grad.size(0) && index < n * m) {
        int l = index % n;
        int j = index / n;

        T _sum = sum[bs][j];
        _sum = (_sum != 0 ? _sum : 1); // Avoid division by zero

        T value = input[bs][l] - scalar_prod_input[bs][j] / _sum;
        value = (grad[bs][j][l] * value * v[bs][j][l]) / alpha;
        atomicAdd(&grad_alpha[0], value);

        value = scalar_prod[bs][j] / _sum;
        grad_w[bs][j] = value;

        value -= v[bs][j][l];
        grad[bs][j][l] *= value;
    }
}

/**
 * Jacobian-vector product (JVP) function for the Soft-Top-k algorithm and
 * the gradient of the log probability with respect to all parameters.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Output tensor of Lap-Sum function with shape (batch_size, m).
 * @param v: Tensor of shape (batch_size, m, n) used in the JVP computation.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A tuple of gradient tensors with shapes:
 *          - Gradient with respect to input: (batch_size, n)
 *          - Gradient with respect to alpha: (1)
 */
std::vector<at::Tensor> jvp_cuda(at::Tensor input, at::Tensor output, at::Tensor v, const float alpha) {
    const auto batch_size = input.size(0);
    const auto n = input.size(1);
    const auto m = output.size(1);

    // Allocate output tensors
    auto grad = at::empty({batch_size, m, n}, input.options());
    auto sum = at::zeros_like(output);
    auto scalar_prod = at::zeros_like(output);
    auto scalar_prod_input = at::zeros_like(output);
    auto grad_w = at::empty_like(output);
    auto grad_alpha = at::zeros({1}, input.options());

    const dim3 num_blocks((n * m + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "components_cuda", ([&] {
        derivative_components_jvp_kernel<scalar_t><<<num_blocks, threads>>>(
            input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),
            sum.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            scalar_prod.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            scalar_prod_input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            grad.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),
            static_cast<scalar_t>(alpha));
    }));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grad_cuda", ([&] {
        derivative_jvp_kernel<scalar_t><<<num_blocks, threads>>>(
            grad.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),
            grad_w.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            grad_alpha.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
            sum.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            scalar_prod.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            scalar_prod_input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),
            input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            static_cast<scalar_t>(alpha));
    }));

    return {grad.sum(1), grad_w, grad_alpha};
}

/**
 * Vector-Jacobin product (VJP) function for the Soft-Top-k algorithm and
 * the gradient of the log probability with respect to all parameters.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Output tensor of Lap-Sum function with shape (batch_size, m).
 * @param v: Tensor of shape (batch_size, m, n) used in the VJP computation.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A tuple of gradient tensors with shapes:
 *          - Gradient with respect to input: (batch_size, n)
 *          - Gradient with respect to alpha: (1)
 */
std::vector<at::Tensor> vjp_cuda(at::Tensor input, at::Tensor output, at::Tensor v, const float alpha) {
    // Reuse the JVP function for VJP
    return jvp_cuda(input, output, v, alpha);
}

