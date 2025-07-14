//
// Created by lstruski
//

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <vector>
#include <cmath>

// Macros to ensure tensors are on the CPU and contiguous
#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

/**
 **************************************************************************************************************
 *                                            Basic operations
 **************************************************************************************************************
**/

/*
 * Forward wrapper function to compute intermediate values and results.
 * @param input: Input array.
 * @param results: Output array to store results.
 * @param n: Number of elements in the input array.
 * @param batch_size: Number of batches.
 * @param alpha: Scale parameter of the Laplace distribution.
 */
template <typename T>
void forward_wrapper(const T* input, T* results,
                     const int n, const int batch_size, const T alpha) {
    int idx, idx_tmp;
    T val;

    for (int bs = 0; bs < batch_size; ++bs) {
        idx = bs * n;

        val = 0;
        for (int i = 0; i < n - 1; ++i) {
            idx_tmp = idx + i;
            val = std::exp((input[idx_tmp] - input[idx_tmp + 1]) / alpha) * (1 + val);
            results[idx_tmp] += i;
            results[idx_tmp + 1] -= 0.5 * val;
        }
        results[idx + n - 1] += n - 1;

        val = 0;
        for (int i = idx + n - 1; i > idx; --i) {
            val = std::exp((input[i - 1] - input[i]) / alpha) * (1 + val);
            results[i - 1] += 0.5 * val;
        }
    }
}

/**
 **************************************************************************************************************
 *                                               Forward pass
 **************************************************************************************************************
**/

/**
 * Performs the forward pass for the Soft-Rank algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param alpha: Scale parameter for the Laplace distribution, controlling the sharpness of the distribution.
 * @return: A tuple containing:
 *          - The output tensor of the Lap-Rank function with shape (batch_size, n).
 */
at::Tensor forward(at::Tensor input, const float alpha = 1) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input and w are CPU tensors and contiguous
    CHECK_INPUT(input);

    // Extract dimensions
    auto batch_size = input.size(0);
    auto n = input.size(1);

    // Ensure input and w have the correct dimensions
    TORCH_CHECK(input.dim() == 2, "Tensor 'input' must have 2 dimensions");
    
    // Sort input and w tensors
//    at::Tensor sorted_input, indices;
//    std::tie(sorted_input, indices) = at::sort(input, /*dim=*/1, /*descending=*/(alpha < 0));
    at::Tensor indices;
    indices = at::argsort(input, /*dim=*/1, /*descending=*/(alpha < 0));
    input = at::gather(input, /*dim=*/1, indices);

    // Allocate output tensors
    auto results = at::zeros_like(input);

    // Dispatch based on the input type (float or double)
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "forward_wrapper", ([&] {
        const auto input_ptr = input.data_ptr<scalar_t>();
//        const auto sorted_input_ptr = sorted_input.data_ptr<scalar_t>();
        auto results_ptr = results.data_ptr<scalar_t>();

        forward_wrapper<scalar_t>(input_ptr, results_ptr,
                                  n, batch_size, static_cast<scalar_t>(alpha));
    }));

    return at::empty_like(results).scatter_(1, indices, results);
    // results = at::gather(results, 1, indices);
}


/**
 **************************************************************************************************************
 *                                            Calculate gradients
 **************************************************************************************************************
**/

/**
 * Computes the Jacobian-vector product (JVP) for the Soft-Rank algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param v: Tensor of shape (batch_size, n) used in the JVP computation.
 * @param grad: Gradient tensor with respect to the input, of shape (batch_size, n).
 * @param grad_alpha: Gradient tensor with respect to the parameter alpha, of shape (1).
 * @param n: Number of elements in the input dimension.
 * @param batch_size: Number of samples in the batch.
 * @param alpha: Scale parameter for the Laplace distribution.
 */
template <typename T>
void derivative_jvp(const T* input, const T* v, T* grad, T* grad_alpha,
    const int n, const int batch_size, const T alpha) {
    int in_idx, out_idx, idx;
    T sum_row, temp1, temp2, value;

    grad_alpha[0] = 0;
    for (int bs = 0; bs < batch_size; ++bs) {
        idx = bs * n;
        for (int i = 0; i < n; ++i) {
            in_idx = idx + i;

            value = 0, sum_row = 0;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                out_idx = idx + j;
                temp1 = input[in_idx] - input[out_idx];
                temp2 = std::exp(-std::abs(temp1 / alpha));
                sum_row += temp2;
                value -= temp2 * v[out_idx];
                grad_alpha[0] += temp1 * temp2 * v[out_idx];
            }
            value += sum_row * v[in_idx];
            grad[in_idx] = value / (2 * alpha);
        }
    }
    grad_alpha[0] /= 2 * alpha * alpha;
}

/**
 * Computes the Jacobian-vector product (JVP) for the Soft-Rank algorithm and the derivatives
 * of the probability with respect to the input, weights (w), and the scale parameter (alpha).
 * This function is used to efficiently compute gradients for the input, weights, and alpha
 * in the context of the Laplace distribution-based Soft-Rank operation.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param v: Tensor of shape (batch_size, m, n) used in the JVP computation.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A tuple of gradient tensors with shapes:
 *          - Gradient with respect to input: (batch_size, n)
 *          - Gradient with respect to alpha: (1)
 */
std::vector<at::Tensor> jvp(at::Tensor input, at::Tensor v, const float alpha = 1) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(input);
    CHECK_INPUT(v);

    // Check tensor dimensions
    TORCH_CHECK(input.dim() == 2, "'input' tensor must have 2 dimensions");
    TORCH_CHECK(input.sizes() == v.sizes(), "Tensors must have the same shape");

    // Extract dimensions
    auto batch_size = input.size(0);
    auto n = input.size(1);
    
    auto grad = at::empty_like(input);
    auto grad_alpha = at::empty({1}, input.options());
    
    // Dispatch based on the input type (float or double)
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "derivative", ([&] {
        const auto input_ptr = input.data_ptr<scalar_t>();
        const auto v_ptr = v.data_ptr<scalar_t>();
        auto grad_ptr = grad.data_ptr<scalar_t>();
        auto grad_alpha_ptr = grad_alpha.data_ptr<scalar_t>();

        derivative_jvp<scalar_t>(input_ptr, v_ptr, grad_ptr, grad_alpha_ptr,
            n, batch_size, static_cast<scalar_t>(alpha));
    }));

    return {grad, grad_alpha};
}

/**
 * Computes the vector-Jacobian product (VJP) for the Soft-Rank algorithm and the derivative
 * of the probability with respect to the scale parameter (alpha). This function is used to
 * efficiently compute gradients for the input tensor in the context of the Laplace
 * distribution-based Soft-Rank operation.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param v: Tensor of shape (batch_size, n) used in the VJP computation.
 * @param alpha: Scale parameter for the Laplace distribution, controlling the sharpness of the distribution.
 * @return: A tuple of gradient tensors with shapes:
 *          - Gradient with respect to input: (batch_size, n)
 *          - Gradient with respect to alpha: (1)
 */
std::vector<at::Tensor> vjp(at::Tensor input, at::Tensor v, const float alpha = 1) {
    return jvp(input, v, alpha);
}

/**
 * Pybind11 module definition for the Soft-Top-k extension.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Soft-Rank forward (CPU)");
    m.def("jvp", &jvp, "Soft-Rank jvp (CPU)");
    m.def("vjp", &vjp, "Soft-Rank vjp -- backward (CPU)");
}
