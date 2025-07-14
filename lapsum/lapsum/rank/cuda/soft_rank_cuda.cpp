//
// Created by lstruski
//

#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

at::Tensor forward_cuda(at::Tensor input, float alpha);
std::vector<at::Tensor> jvp_cuda(at::Tensor input, at::Tensor v, float alpha);
std::vector<at::Tensor> vjp_cuda(at::Tensor input, at::Tensor v, float alpha);

// C++ interface

// Macros to ensure tensors are on the CUDA device and contiguous
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/**
 * Forward pass for the Soft-Rank algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A vector containing the results tensor of Lap-Rank function.
 */
at::Tensor forward(at::Tensor input, const float alpha = 1) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");
    // Ensure input and w are CUDA tensors and contiguous
    CHECK_INPUT(input);

    // Ensure the input tensor is 2-dimensional
    TORCH_CHECK(input.dim() == 2, "'input' tensor must have 2 dimensions");

    // Sort input data and w
    at::Tensor indices;
    indices = at::argsort(input, /*dim=*/1, /*descending=*/(alpha < 0));
    input = at::gather(input, /*dim=*/1, indices);

    // Call the CUDA implementation of the forward pass
    at::Tensor results = forward_cuda(input, alpha);
    return at::empty_like(results).scatter_(1, indices, results);
}

/**
 * Jacobin-vector product (JVP) function for the Soft-Rank algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param v: Tensor of shape (batch_size, n) used in the JVP computation.
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

    // Ensure the input tensor is 2-dimensional
    TORCH_CHECK(input.dim() == 2, "'input' tensor must have 2 dimensions");
    TORCH_CHECK(input.sizes() == v.sizes(), "Tensors must have the same shape");

    // Call the CUDA implementation of the JVP
    return jvp_cuda(input, v, alpha);
}

/**
 * Vector-Jacobin product (VJP) function for the Soft-Rank algorithm.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param v: Tensor of shape (batch_size, n) used in the VJP computation.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A tuple of gradient tensors with shapes:
 *          - Gradient with respect to input: (batch_size, n)
 *          - Gradient with respect to alpha: (1)
 */
std::vector<at::Tensor> vjp(at::Tensor input, at::Tensor v, const float alpha = 1) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(input);
    CHECK_INPUT(v);

    // Ensure the input tensor is 2-dimensional
    TORCH_CHECK(input.dim() == 2, "'input' tensor must have 2 dimensions");
    TORCH_CHECK(input.sizes() == v.sizes(), "Tensors must have the same shape");

    // Call the CUDA implementation of the VJP
    return vjp_cuda(input, v, alpha);
}

/**
 * Pybind11 module definition for the Soft-Rank CUDA extension.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Soft-Rank forward (CUDA)");
    m.def("jvp", &jvp, "Soft-Rank jvp (CUDA)");
    m.def("vjp", &vjp, "Soft-Rank vjp -- backward (CUDA)");
}
