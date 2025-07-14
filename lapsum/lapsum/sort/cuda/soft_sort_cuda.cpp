//
// Created by lstruski
//

#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

at::Tensor forward_cuda(at::Tensor input, const float alpha);
at::Tensor jvp_cuda(at::Tensor input, at::Tensor output, at::Tensor v, float alpha);
at::Tensor vjp_cuda(at::Tensor input, at::Tensor output, at::Tensor v, float alpha);

// C++ interface

// Macros to ensure tensors are on the CUDA device and contiguous
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/**
 * Forward pass for the Soft Top-K algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A tensor of Lap-Sort function.
 */
at::Tensor forward(at::Tensor input, const float alpha = 1) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");
    // Ensure input and w are CUDA tensors and contiguous
    CHECK_INPUT(input);

    // Ensure the input tensor is 2-dimensional
    TORCH_CHECK(input.dim() == 2, "'input' tensor must have 2 dimensions");

    // Call the CUDA implementation of the forward pass
    return forward_cuda(input, alpha);
}

/**
 * Jacobin-vector product (JVP) function for the Soft Top-K algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Output tensor of Lap-Sum function with shape (batch_size, n).
 * @param v: Tensor of shape (batch_size, n) used in the JVP computation.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, n).
 */
at::Tensor jvp(at::Tensor input, at::Tensor output, at::Tensor v, const float alpha = 1) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(v);

    // Ensure the input tensor is 2-dimensional
    TORCH_CHECK(input.dim() == 2, "'input' tensor must have 2 dimensions");
    TORCH_CHECK(input.sizes() == output.sizes(), "The shape of 'input' and 'output' should be equal.");
    TORCH_CHECK(input.sizes() == v.sizes(), "The shape of 'input' and 'v' should be equal.");

    // Call the CUDA implementation of the JVP
    return jvp_cuda(input, output, v, alpha);
}

/**
 * Vector-Jacobin product (VJP) function for the Soft-Top-k algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Output tensor of Lap-Sum function with shape (batch_size, n).
 * @param v: Tensor of shape (batch_size, n) used in the VJP computation.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, n).
 */
at::Tensor vjp(at::Tensor input, at::Tensor output, at::Tensor v, const float alpha = 1) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(v);

    // Ensure the input tensor is 2-dimensional
    TORCH_CHECK(input.dim() == 2, "'input' tensor must have 2 dimensions");
    TORCH_CHECK(input.sizes() == output.sizes(), "The shape of 'input' and 'output' should be equal.");
    TORCH_CHECK(input.sizes() == v.sizes(), "The shape of 'input' and 'v' should be equal.");

    // Call the CUDA implementation of the VJP
    return vjp_cuda(input, output, v, alpha);
}

/**
 * Pybind11 module definition for the Soft-Sort extension.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Soft-Top-k forward (CUDA)");
    m.def("jvp", &jvp, "Soft-Top-k jvp (CUDA)");
    m.def("vjp", &vjp, "Soft-Top-k vjp -- backward (CUDA)");
}
