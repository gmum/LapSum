//
// Created by lstruski
//

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> forward_cuda(at::Tensor input, at::Tensor sorted_input, at::Tensor w, float alpha);
std::vector<at::Tensor> jvp_cuda(at::Tensor input, at::Tensor output, at::Tensor log_prob, at::Tensor v, float alpha);
std::vector<at::Tensor> vjp_cuda(at::Tensor input, at::Tensor output, at::Tensor log_prob, at::Tensor v, float alpha);

// C++ interface

// Macros to ensure tensors are on the CUDA device and contiguous
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/**
 * Forward pass for the logarithm of Soft-Top-k algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param w: Weight tensor of shape (batch_size, m).
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A vector containing the results tensor and the logarithm of probability tensor.
 */
std::vector<at::Tensor> forward(at::Tensor input, at::Tensor w, const float alpha = -1) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input and w are CUDA tensors and contiguous
    CHECK_INPUT(input);
    CHECK_INPUT(w);

    // Ensure the input tensor is 2-dimensional
    TORCH_CHECK(input.dim() == 2, "'input' tensor must have 2 dimensions");
    TORCH_CHECK(w.dim() == 2, "'w' tensor must have 2 dimensions");
    TORCH_CHECK(input.size(0) == w.size(0), "The first dimensions of 'input' and 'w' should be equal.");

    // Sort input data and w
    at::Tensor sorted_input, indices;
    std::tie(sorted_input, indices) = at::sort(input, /*dim=*/1, /*descending=*/(alpha < 0));

    // Call the CUDA implementation of the forward pass
    return forward_cuda(input, sorted_input, w, alpha);
}

/**
 * Jacobian-vector product (JVP) function for the logarithm of Soft-Top-k algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Output tensor of Lap-Sum function with shape (batch_size, m).
 * @param log_prob: Logarithmic of probability tensor of shape (batch_size, m, n).
 * @param v: Tensor of shape (batch_size, m, n) used in the JVP computation.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, m, n).
 */
std::vector<at::Tensor> jvp(
    at::Tensor input, at::Tensor output, at::Tensor log_prob, at::Tensor v, const float alpha = -1) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(log_prob);
    CHECK_INPUT(v);

    // Ensure the input tensor is 2-dimensional
    TORCH_CHECK(input.dim() == 2, "'input' tensor must have 2 dimensions");
    TORCH_CHECK(output.dim() == 2, "'output' tensor must have 2 dimensions");
    TORCH_CHECK(input.size(0) == output.size(0), "The first dimensions of 'input' and 'output' should be equal.");
    TORCH_CHECK(log_prob.sizes() == v.sizes(), "The sizes of 'log_prob' and 'v' should be equal.");

    // Call the CUDA implementation of the JVP
    return jvp_cuda(input, output, log_prob, v, alpha);
}

/**
 * Vector-Jacobian product (VJP) function for the logarithm of Soft-Top-k algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Output tensor of Lap-Sum function with shape (batch_size, m).
 * @param log_prob: Logarithmic of probability tensor of shape (batch_size, m, n).
 * @param v: Tensor of shape (batch_size, m, n) used in the VJP computation.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, m, n).
 */
std::vector<at::Tensor> vjp(
    at::Tensor input, at::Tensor output, at::Tensor log_prob, at::Tensor v, const float alpha = -1) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(log_prob);
    CHECK_INPUT(v);

    // Ensure the input tensor is 2-dimensional
    TORCH_CHECK(input.dim() == 2, "'input' tensor must have 2 dimensions");
    TORCH_CHECK(output.dim() == 2, "'output' tensor must have 2 dimensions");
    TORCH_CHECK(input.size(0) == output.size(0), "The first dimensions of 'input' and 'output' should be equal.");
    TORCH_CHECK(log_prob.sizes() == v.sizes(), "The sizes of 'log_prob' and 'v' should be equal.");

    // Call the CUDA implementation of the VJP
    return vjp_cuda(input, output, log_prob, v, alpha);
}

/**
 * Pybind11 module definition for the logarithm of Soft-Top-k extension.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Soft-Top-k forward (CUDA)");
    m.def("jvp", &jvp, "Soft-Top-k jvp (CUDA)");
    m.def("vjp", &vjp, "Soft-Top-k vjp -- backward (CUDA)");
}
