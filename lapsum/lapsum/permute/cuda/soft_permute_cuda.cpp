//
// Created by lstruski
//

#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

at::Tensor solveLap_cuda(at::Tensor sorted_input, at::Tensor w, float alpha);
std::vector<at::Tensor> solveLap_jvp_cuda(at::Tensor sorted_input, at::Tensor w_out, at::Tensor sorted_v, float alpha);
at::Tensor solveLap_vjp_cuda(at::Tensor sorted_input, at::Tensor w_out, at::Tensor v, float alpha);
at::Tensor doubleStochasticMatrix_cuda(at::Tensor input, at::Tensor output, float alpha);
at::Tensor lossCE_cuda(at::Tensor input, at::Tensor output, at::Tensor target, float alpha);
std::vector<at::Tensor> derivative_lossCE_cuda(at::Tensor input, at::Tensor output, at::Tensor target, float alpha);

// C++ interface

// Macros to ensure tensors are on the CUDA device and contiguous
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/**
 * solveLap mapping (x, w) -> b.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n).
 * @param w: Weight tensor of shape (batch_size, m). Must be sorted in ascending order as sorted_input.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A vector containing the results tensor and the probability tensor.
 */
at::Tensor solveLap(at::Tensor sorted_input, at::Tensor w, const float alpha) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");
    // Ensure input and w are CUDA tensors and contiguous
    CHECK_INPUT(sorted_input);
    CHECK_INPUT(w);

    // Ensure the input tensor is 2-dimensional
    TORCH_CHECK(sorted_input.dim() == 2, "'sorted_input' tensor must have 2 dimensions");
    TORCH_CHECK(w.dim() == 2, "'w' tensor must have 2 dimensions");
    TORCH_CHECK(sorted_input.size(0) == w.size(0), "The first dimensions of 'sorted_input' and 'w' should be equal.");

    // Call the CUDA implementation of the forward pass
    return solveLap_cuda(sorted_input, w, alpha);
}

/**
 * Jacobian-vector product (JVP) function for mapping (x, w) -> b.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n).
 * @param b_out: Output tensor of shape (batch_size, m).
 * @param sorted_v: Vector tensor of shape (batch_size, n), aligned with sorted_input.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, m).
 */
std::vector<at::Tensor> solveLap_jvp(at::Tensor sorted_input, at::Tensor b_out, at::Tensor sorted_v, const float alpha) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(sorted_input);
    CHECK_INPUT(b_out);
    CHECK_INPUT(sorted_v);

    // Ensure the input tensor is 2-dimensional
    TORCH_CHECK(sorted_input.dim() == 2, "'sorted_input' tensor must have 2 dimensions");
    TORCH_CHECK(b_out.dim() == 2, "'b_out' tensor must have 2 dimensions");
    TORCH_CHECK(sorted_input.sizes() == sorted_v.sizes(), "Tensor 'sorted_v' must have the same shape as 'sorted_input'.");
    TORCH_CHECK(sorted_input.size(0) == b_out.size(0), "The first dimensions of 'input' and 'b_out' should be equal.");

    // Call the CUDA implementation of the JVP
    return solveLap_jvp_cuda(sorted_input, b_out, sorted_v, alpha);
}

/**
 * Vector-Jacobian product (VJP) function for mapping (x, w) -> b.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n).
 * @param b_out: Output tensor of shape (batch_size, m).
 * @param v: Vector tensor of shape (batch_size, m).
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, n).
 */
at::Tensor solveLap_vjp(at::Tensor sorted_input, at::Tensor b_out, at::Tensor v, const float alpha) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(sorted_input);
    CHECK_INPUT(b_out);
    CHECK_INPUT(v);

    // Ensure the input tensor is 2-dimensional
    TORCH_CHECK(sorted_input.dim() == 2, "'sorted_input' tensor must have 2 dimensions");
    TORCH_CHECK(b_out.dim() == 2, "'b_out' tensor must have 2 dimensions");
    TORCH_CHECK(v.sizes() == b_out.sizes(), "Tensor 'v' must have the same shape as 'b_out'.");
    TORCH_CHECK(sorted_input.size(0) == b_out.size(0), "The first dimensions of 'sorted_input' and 'b_out' should be equal.");

    // Call the CUDA implementation of the VJP
    return solveLap_vjp_cuda(sorted_input, b_out, v, alpha);
}

/**
 * Compute the double stochastic matrix mapping (x, w) -> output.
 *
 * This function computes a double stochastic matrix for each batch of input and output elements.
 * The matrix has dimensions (n x (m + 1)) for each batch.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param b_out: Weight tensor of shape (batch_size, m). Output of solveLap function.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: stochastic matrix that has shape (batch_size, n, (m + 1)).
 */

at::Tensor stochastic_matrix(at::Tensor input, at::Tensor b_out, const float alpha) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(input);
    CHECK_INPUT(b_out);

    // Ensure the input tensor is 2-dimensional
    TORCH_CHECK(input.dim() == 2, "'sorted_input' tensor must have 2 dimensions");
    TORCH_CHECK(b_out.dim() == 2, "'b_out' tensor must have 2 dimensions");
    TORCH_CHECK(input.size(0) == b_out.size(0), "The first dimensions of 'input' and 'b_out' should be equal.");

    return doubleStochasticMatrix_cuda(input, b_out, alpha);
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
at::Tensor lossCE(at::Tensor input, at::Tensor output, at::Tensor target, const float alpha) {
    // Ensure alpha is not zero
    TORCH_CHECK(alpha != 0, "Parameter 'alpha' cannot be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(target);

    // Ensure input and output have the correct dimensions
    TORCH_CHECK(input.dim() == 2, "Tensor 'input' must have 2 dimensions.");
    TORCH_CHECK(output.dim() == 2, "Tensor 'output' must have 2 dimensions.");
    TORCH_CHECK(target.sizes() == input.sizes(), "Tensor 'target' must have the same shape as 'input'.");
    TORCH_CHECK(input.size(0) == output.size(0), "The first dimensions of 'input' and 'output' should be equal.");

    return lossCE_cuda(input, output, target, alpha);
}

std::vector<at::Tensor> derivative_lossCE(at::Tensor input, at::Tensor output, at::Tensor target, const float alpha) {
    // Ensure alpha is not zero
    TORCH_CHECK(alpha != 0, "Parameter 'alpha' cannot be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(target);

    // Ensure input and output have the correct dimensions
    TORCH_CHECK(input.dim() == 2, "Tensor 'input' must have 2 dimensions.");
    TORCH_CHECK(output.dim() == 2, "Tensor 'output' must have 2 dimensions.");
    TORCH_CHECK(target.sizes() == input.sizes(), "Tensor 'target' must have the same shape as 'input'.");
    TORCH_CHECK(input.size(0) == output.size(0), "The first dimensions of 'input' and 'output' should be equal.");

    return derivative_lossCE_cuda(input, output, target, alpha);
}


/**
 * Pybind11 module definition for the Soft Permute extension.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solveLap", &solveLap, "solveLap forward (CUDA)");
    m.def("solveLap_jvp", &solveLap_jvp, "solveLap jvp (CUDA)");
    m.def("solveLap_vjp", &solveLap_vjp, "solveLap vjp (CUDA)");
    m.def("stochastic_matrix", &stochastic_matrix, "Double Stochastic Matrix (CUDA)");
    m.def("lossCE", &lossCE, "Cross-Entropy loss (CUDA)");
    m.def("DlossCE", &derivative_lossCE, "derivative of Cross-Entropy loss (CUDA)");
}
