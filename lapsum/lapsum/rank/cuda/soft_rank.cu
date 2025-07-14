//
// Created by lstruski
//

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>


const int threads = 1024; // Number of threads per block

/*
 * CUDA kernel to compute the Soft-Rank tensor.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param temp: Output tensor A of shape (batch_size, n).
 * @param alpha: Scale parameter of the Laplace distribution.
 */
template <typename T>
__global__ void computeRankKernelA(
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> temp, const T alpha) {
    //batch index
    const int bs = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs < input.size(0)) {
        const int n = input.size(1);
        T val = 0;

        for (int i = n - 1; i > 0; --i) {
            val = exp((input[bs][i - 1] - input[bs][i]) / alpha) * (1 + val);
            atomicAdd(&temp[bs][i - 1], 0.5 * val);
        }
    }
}

/*
 * CUDA kernel to compute the Soft-Rank tensor.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param temp: Output tensor B of shape (batch_size, n).
 * @param alpha: Scale parameter of the Laplace distribution.
 */
template <typename T>
__global__ void computeRankKernelB(
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> temp, const T alpha) {
    //batch index
    const int bs = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs < input.size(0)) {
        const int n = input.size(1);
        T val = 0;

        for (int i = 0; i < n - 1; ++i) {
            val = exp((input[bs][i] - input[bs][i + 1]) / alpha) * (1 + val);
            atomicAdd(&temp[bs][i], i);
            atomicAdd(&temp[bs][i + 1], -0.5 * val);
        }
        atomicAdd(&temp[bs][n - 1], n - 1);
    }
}

/*********************************************************************************************/
/*********************************************************************************************/
/*********************************************************************************************/

/**
 * Forward pass for the Soft-Rank algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A vector containing the results tensor of Lap-Rank function.
 */
at::Tensor forward_cuda(at::Tensor input, const float alpha) {
    const auto batch_size = input.size(0);
    const auto n = input.size(1);

    // Allocate tensors
    auto results = at::zeros_like(input);

    const int blocks_bs = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rank_cuda", ([&] {
        computeRankKernelB<scalar_t><<<blocks_bs, threads>>>(
            input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            results.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            static_cast<scalar_t>(alpha));
    }));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rank_cuda", ([&] {
        computeRankKernelA<scalar_t><<<blocks_bs, threads>>>(
            input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
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
 * CUDA kernel to calculate components to Jacobian-vector product (JVP) for the Soft-Rank algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param v: Tensor of shape (batch_size, n) used in the JVP computation.
 * @param grad: Tensor of shape (batch_size, n) contains components to calculate gradient.
 * @param grad_alpha: Tensor of shape (1) contains components to calculate gradient alpha.
 * @param alpha: Scale parameter for the Laplace distribution.
 */
template <typename T>
__global__ void derivative_jvp_kernel(
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> input,
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> v,
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> grad,
    at::PackedTensorAccessor32<T, 1, at::RestrictPtrTraits> grad_alpha, const T alpha) {
    const int bs = blockIdx.y;
    const int n = input.size(1);

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs < input.size(0) && i < n) {
        T temp1, temp2, sum_row = 0, value = 0, value_alpha = 0;

        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            temp1 = input[bs][i] - input[bs][j];
            temp2 = exp(-abs(temp1 / alpha));
            sum_row += temp2;
            value -= temp2 * v[bs][j];
            value_alpha += temp1 * temp2 * v[bs][j];
        }
        value += sum_row * v[bs][i];
        grad[bs][i] = value / (2 * alpha);
        atomicAdd(&grad_alpha[0], value_alpha / (2 * alpha * alpha));
    }
}


/**
 * Jacobian-vector product (JVP) function for the Soft-Rank algorithm
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param v: Tensor of shape (batch_size, m, n) used in the JVP computation.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A tuple of gradient tensors with shapes:
 *          - Gradient with respect to input: (batch_size, n)
 *          - Gradient with respect to alpha: (1)
 */
std::vector<at::Tensor> jvp_cuda(at::Tensor input, at::Tensor v, const float alpha) {
    const auto batch_size = input.size(0);
    const auto n = input.size(1);

    // Allocate output tensors
    auto grad = at::empty_like(input);
    auto grad_alpha = at::zeros({1}, input.options());

    const dim3 num_blocks((n + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grad_cuda", ([&] {
        derivative_jvp_kernel<scalar_t><<<num_blocks, threads>>>(
            input.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            grad.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            grad_alpha.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
            static_cast<scalar_t>(alpha));
    }));

    return {grad, grad_alpha};
}

/**
 * Vector-Jacobin product (VJP) function for the Soft-Rank algorithm
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param v: Tensor of shape (batch_size, m, n) used in the VJP computation.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A tuple of gradient tensors with shapes:
 *          - Gradient with respect to input: (batch_size, n)
 *          - Gradient with respect to alpha: (1)
 */
std::vector<at::Tensor> vjp_cuda(at::Tensor input, at::Tensor v, const float alpha) {
    // Reuse the JVP function for VJP
    return jvp_cuda(input, v, alpha);
}

