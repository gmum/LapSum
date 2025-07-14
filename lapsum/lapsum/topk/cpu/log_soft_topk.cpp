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
 * Function to compute the probability density function (PDF) of a Laplace distribution.
 * @param x: Input value.
 * @param alpha: Scale parameter of the Laplace distribution.
 * @return: PDF value at x.
 */
template <typename T>
T pdfLap(T x, T alpha) {
    return std::exp(-std::abs(x / alpha)) / (2 * alpha);
}

/*
 * Function to compute the logarithm of the cumulative distribution function (CDF) of a Laplace distribution.
 * @param x: Input value.
 * @param alpha: Scale parameter of the Laplace distribution.
 * @return: Logarithm of CDF value at x.
 */
template <typename T>
T log_cdfLap(T x, T alpha) {
    T val = x / alpha;
    if (val <= 0) {
        return std::log(static_cast<T>(0.5)) + val;
    }
    return std::log(static_cast<T>(2) - std::exp(-val)) + std::log(static_cast<T>(0.5));
}

/*
 * Function to solve the equation exp(a)/2*exp(x-s)-exp(b)/2*exp(r-x)+c=w.
 * @param r, s: Input parameters.
 * @param a, b, c, w: Equation coefficients.
 * @param alpha: Scale parameter of the Laplace distribution.
 * @return: Solution to the equation.
 */
template <typename T>
T solveExp(T r, T s, T a, T b, T c, T w, T alpha) {
    if (w > c && a > 0) {
        T diff = w - c;
        return s - alpha * std::log(a) + alpha * std::log(
            diff + std::sqrt(diff * diff + std::exp((r - s) / alpha) * a * b));
    }
    if (w == c && a > 0 && b > 0) {
        return 0.5 * (r + s + alpha * (std::log(b) - std::log(a)));
    }
    if (w < c) {
        T diff = c - w;
        return r + alpha * std::log(b) - alpha * std::log(
            diff + std::sqrt(diff * diff + std::exp((r - s) / alpha) * a * b));
    }
    return 0.5 * (r + s);
}

/*
 * Forward wrapper function to compute intermediate values and results.
 * @param input: Input array.
 * @param w: Weight array.
 * @param results: Output array to store results.
 * @param n: Number of elements in the input array.
 * @param m: Number of elements in the weight array.
 * @param batch_size: Number of batches.
 * @param alpha: Scale parameter of the Laplace distribution.
 */
template <typename T>
void forward_wrapper(const T* input, const T* w, T* results,
                     const int n, const int m, const int batch_size, const T alpha) {
    for (int bs = 0; bs < batch_size; ++bs) {
        int middle, start = 0, end = n - 1;
        middle = (start + end) / 2;

        T a_start = 0, a_mid = 0, a_end = 0;
        for (int i = middle + 1; i <= end; ++i) {
            a_mid += std::exp((input[bs * n + middle] - input[bs * n + i]) / alpha);
        }
        for (int i = start + 1; i <= middle; ++i) {
            a_start += std::exp((input[bs * n + start] - input[bs * n + i]) / alpha);
        }
        a_start += std::exp((input[bs * n + start] - input[bs * n + middle]) / alpha) * a_mid;

        T b_start = 0, b_mid = 0, b_end = 0;
        for (int i = start; i < middle; ++i) {
            b_mid += std::exp((input[bs * n + i] - input[bs * n + middle]) / alpha);
        }
        for (int i = middle; i < end; ++i) {
            b_end += std::exp((input[bs * n + i] - input[bs * n + end]) / alpha);
        }
        b_end += std::exp((input[bs * n + middle] - input[bs * n + end]) / alpha) * b_mid;

        for (int i = 0; i < m; ++i) {
            if (w[bs * m + i] <= 0.5 * (a_start - b_start + 1)) {
                results[bs * m + i] = solveExp(
                        input[bs * n], input[bs * n], 1 + a_start,
                        static_cast<T>(0), static_cast<T>(0), w[bs * m + i], alpha
                    );
            } else if (w[bs * m + i] >= 0.5 * (a_end - b_end + 1) + static_cast<T>(end)) {
                results[bs * m + i] = solveExp(
                        input[(bs + 1) * n - 1], input[(bs + 1) * n - 1], static_cast<T>(0),
                        1 + b_end, static_cast<T>(n), w[bs * m + i], alpha
                    );
            } else {
                int l = start, mid = middle, r = end;
                T a_s = a_start, b_s = b_start, a_m = a_mid, b_m = b_mid, a_e = a_end, b_e = b_end;
                while (l + 1 < r) {
                    if (w[bs * m + i] <= 0.5 * (a_m - b_m + 1) + static_cast<T>(mid)) {
                        r = mid;
                        a_e = a_m;
                        b_e = b_m;
                    } else {
                        l = mid;
                        a_s = a_m;
                        b_s = b_m;
                    }
                    mid = (l + r) / 2;
                    a_m = 0;
                    for (int j = mid + 1; j <= r; ++j) {
                        a_m += std::exp((input[bs * n + mid] - input[bs * n + j]) / alpha);
                    }
                    a_m += std::exp((input[bs * n + mid] - input[bs * n + r]) / alpha) * a_e;
                    b_m = 0;
                    for (int j = l; j < mid; ++j) {
                        b_m += std::exp((input[bs * n + j] - input[bs * n + mid]) / alpha);
                    }
                    b_m += std::exp((input[bs * n + l] - input[bs * n + mid]) / alpha) * b_s;
                }
                results[bs * m + i] = solveExp(
                    input[bs * n + l], input[bs * n + r], 1 + a_e, 1 + b_s,
                    static_cast<T>(1 + l), w[bs * m + i], alpha
                );
            }
        }
    }
}


/*
 * Function to calculate the logarithm of probability of the forward past.
 * @param input: Input array with shape (batch_size, n).
 * @param output: Output array of Lap-Sum function with shape (batch_size, m).
 * @param log_prob: Array to store logarithm of probabilities with shape (batch_size, m, n).
 * @param n: Number of elements in the input array.
 * @param m: Number of elements in the output array.
 * @param batch_size: Number of batches.
 * @param alpha: Scale parameter of the Laplace distribution.
 */
template <typename T>
void log_probability(const T* input, const T* output, T* log_prob,
                     const int n, const int m, const int batch_size, const T alpha) {
    int in_off, out_off, lp_off, out_idx, lp_row;
    for (int bs = 0; bs < batch_size; ++bs) {
        in_off = bs * n;
        out_off = bs * m;
        lp_off = bs * n * m;

        for (int i = 0; i < m; ++i) {
            out_idx = out_off + i;
            lp_row = lp_off + i * n;

            for (int j = 0; j < n; ++j) {
                log_prob[lp_row + j] = log_cdfLap(output[out_idx] - input[in_off + j], alpha);
            }
        }
    }
}

/**
 **************************************************************************************************************
 *                                               Forward pass
 **************************************************************************************************************
**/

/**
 * Performs the forward pass for the logarithm of Soft Top-K algorithm using the Lap-Sum function.
 * This function computes the output of the Lap-Sum operation and the associated probability tensor
 * based on the input tensor, weight tensor, and the Laplace distribution scale parameter.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param w: Weight tensor of shape (batch_size, m). Values must lie in the interval (0, n).
 * @param alpha: Scale parameter for the Laplace distribution, controlling the sharpness of the distribution.
 * @return: A tuple containing:
 *          - The output tensor of the Lap-Sum function with shape (batch_size, m).
 *          - The logarithm of probability tensor with shape (batch_size, m, n).
 */
std::vector<at::Tensor> forward(at::Tensor input, at::Tensor w, const float alpha = -1) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input and w are CPU tensors and contiguous
    CHECK_INPUT(input);
    CHECK_INPUT(w);

    // Extract dimensions
    auto batch_size = input.size(0);
    auto n = input.size(1);
    auto m = w.size(1);

    // Ensure input and w have the correct dimensions
    TORCH_CHECK(input.dim() == 2, "Tensor 'input' must have 2 dimensions");
    TORCH_CHECK(w.dim() == 2, "Tensor 'w' must have 2 dimensions");
    TORCH_CHECK(batch_size == w.size(0), "The first dimensions of 'input' and 'w' should be equal.");

    // Sort input and w tensors
    at::Tensor sorted_input, indices;
    std::tie(sorted_input, indices) = at::sort(input, /*dim=*/1, /*descending=*/(alpha < 0));

    // Allocate output tensors
    auto results = at::empty_like(w);
    auto log_prob = at::empty({batch_size, m, n}, input.options());

    // Dispatch based on the input type (float or double)
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "forward_wrapper", ([&] {
        const auto sorted_input_ptr = sorted_input.data_ptr<scalar_t>();
        const auto w_ptr = w.data_ptr<scalar_t>();
        auto results_ptr = results.data_ptr<scalar_t>();

        forward_wrapper<scalar_t>(sorted_input_ptr, w_ptr, results_ptr,
                                  n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    // Compute logarithm of probabilities
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_probability", ([&] {
        const auto input_ptr = input.data_ptr<scalar_t>();
        auto results_ptr = results.data_ptr<scalar_t>();
        auto log_prob_ptr = log_prob.data_ptr<scalar_t>();

        log_probability<scalar_t>(input_ptr, results_ptr, log_prob_ptr,
                                  n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    // Return the results and logarithm of probabilities
    return {results, log_prob};
}


/**
 **************************************************************************************************************
 *                                            Calculate gradients
 **************************************************************************************************************
**/

/*
 * Auxiliary function to compute a value based on the input x.
 * @param x: Input value of type T.
 * @return: Computed value based on the input x.
 */
template <typename T>
T auxiliary_function(T x) {
    return (x > 0.5) ? 1 / x - 1 : static_cast<T>(1);
}

/**
 * Computes the Jacobian-vector product (JVP) for the logarithm of Soft Top-K algorithm.
 * This function is designed to efficiently compute gradients for the input, weights (w),
 * and the scale parameter (alpha) of the Laplace distribution, while avoiding unnecessary
 * computations when `grad_w` and `grad_alpha` are not required.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Output tensor of Lap-Sum function with shape (batch_size, m).
 * @param log_prob: Logarithm of probability tensor of shape (batch_size, m, n).
 * @param v: Tensor of shape (batch_size, m, n) used in the JVP computation.
 * @param grad: Gradient tensor with respect to the input, of shape (batch_size, n).
 * @param grad_w: Gradient tensor with respect to the weights (w), of shape (batch_size, m).
 * @param grad_alpha: Gradient tensor with respect to the scale parameter (alpha), of shape (1).
 * @param n: Number of elements in the input dimension.
 * @param m: Number of elements in the output dimension.
 * @param batch_size: Number of samples in the batch.
 * @param alpha: Scale parameter for the Laplace distribution.
 */
template <typename T>
void derivative_jvp(
    const T* input, const T* output, const T* log_prob, const T* v,
    T* grad, T* grad_w, T* grad_alpha,
    const int n, const int m, const int batch_size, const T alpha) {
    T value, batch_sum, scalar_prod, scalar_prod_alpha;
    int in_off, out_off, der_off, in_idx, out_idx, idx;

    for (int bs = 0; bs < batch_size; ++bs) {
        in_off = bs * n;
        out_off = bs * m;
        der_off = bs * n * m;

        for (int j = 0; j < m; ++j) {
            out_idx = out_off + j;
            idx = der_off + j * n;
            batch_sum = 0;
            scalar_prod = 0;
            scalar_prod_alpha = 0;

            for (int i = 0; i < n; ++i) {
                in_idx = in_off + i;
                value = pdfLap(output[out_idx] - input[in_idx], alpha);
                batch_sum += value;
                scalar_prod += value * v[idx + i];
                scalar_prod_alpha += value * input[in_idx];
            }
            // Avoid division by zero
            batch_sum = (batch_sum != 0 ? batch_sum : 1);
            scalar_prod /= batch_sum;
            scalar_prod_alpha /= batch_sum;

            grad_w[out_idx] = 0;

            for (int i = 0; i < n; ++i) {
                in_idx = idx + i;

                value = std::exp(log_prob[in_idx]);
                value = auxiliary_function(value);

                grad[in_off + i] += value * (scalar_prod - v[in_idx]) / alpha;
                value *= v[in_idx];
                grad_w[out_idx] += value;
                grad_alpha[0] += value * (input[in_idx] - scalar_prod_alpha);
            }
            grad_w[out_idx] /= batch_sum * alpha;
        }
    }
    grad_alpha[0] /= alpha * alpha;
}

/**
 * Computes the vector-Jacobian product (VJP) for the logarithm of Soft Top-K algorithm and the derivative
 * of the probability with respect to the scale parameter (alpha). This function is used to
 * efficiently compute gradients for the input tensor in the context of the Laplace
 * distribution-based Soft-Top-k operation.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Output tensor of the Lap-Sum function with shape (batch_size, m).
 * @param log_prob: Logarithm of probability tensor of shape (batch_size, m, n).
 * @param v: Tensor of shape (batch_size, m, n) used in the VJP computation.
 * @param grad: Gradient tensor with respect to the input, of shape (batch_size, n).
 * @param grad_w: Gradient tensor with respect to the weights (w), of shape (batch_size, m).
 * @param grad_alpha: Gradient tensor with respect to the scale parameter (alpha), of shape (1).
 * @param n: Number of elements in the input dimension.
 * @param m: Number of elements in the output dimension.
 * @param batch_size: Number of samples in the batch.
 * @param alpha: Scale parameter for the Laplace distribution.
 */
template <typename T>
void derivative_vjp(const T* input, const T* output, const T* log_prob, const T* v,
    T* grad, T* grad_w, T* grad_alpha,
    const int n, const int m, const int batch_size, const T alpha) {
    // Temporary storage for values
    std::vector<T> temp(n);
    std::vector<T> _temp_reduce(n);

    T value, batch_sum, scalar_prod, scalar_prod_alpha;
    int in_off, out_off, der_off, in_idx, out_idx, idx;

    for (int bs = 0; bs < batch_size; ++bs) {
        in_off = bs * n;
        out_off = bs * m;
        der_off = bs * n * m;

        for (int j = 0; j < m; ++j) {
            out_idx = out_off + j;
            idx = der_off + j * n;
            batch_sum = 0;
            scalar_prod = 0;
            scalar_prod_alpha = 0;

            for (int i = 0; i < n; ++i) {
                in_idx = idx + i;
                value = std::exp(log_prob[in_idx]);
                value = auxiliary_function(value);

                _temp_reduce[i] = value;

                scalar_prod += value * v[in_idx];

                in_idx = in_off + i;
                value = pdfLap(output[out_idx] - input[in_idx], alpha);
                batch_sum += value;
                temp[i] = value;
                scalar_prod_alpha += value * input[in_idx];
            }

            // Avoid division by zero
            batch_sum = (batch_sum != 0 ? batch_sum : 1);
            scalar_prod /= batch_sum;
            scalar_prod_alpha /= batch_sum;

            grad_w[out_idx] = scalar_prod / alpha;

            // Update gradient
            for (int i = 0; i < n; ++i) {
                in_idx = idx + i;

                grad[in_off + i] += (scalar_prod * temp[i] - _temp_reduce[i] * v[in_idx]) / alpha;
                grad_alpha[0] += _temp_reduce[i] * (input[in_idx] - scalar_prod_alpha) * v[in_idx];
            }
        }
    }
    grad_alpha[0] /= alpha * alpha;
}



/**
 * Wrapper to compute the Jacobian-vector product (JVP) for the logarithm of Soft Top-K algorithm and the derivatives
 * of the logarithm of probability with respect to the input, weights (w), and the scale parameter (alpha).
 * This function is used to efficiently compute gradients for the input, weights, and alpha
 * in the context of the Laplace distribution-based Soft Top-K operation.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Output tensor of the Lap-Sum function with shape (batch_size, m).
 * @param log_prob: Logarithm of probability tensor of shape (batch_size, m, n).
 * @param v: Tensor of shape (batch_size, m, n) used in the JVP computation.
 * @param alpha: Scale parameter for the Laplace distribution..
 * @return: A tuple of gradient tensors with shapes:
 *          - Gradient with respect to input: (batch_size, n)
 *          - Gradient with respect to weights (w): (batch_size, m)
 *          - Gradient with respect to alpha: (1)
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

    // Check tensor dimensions
    TORCH_CHECK(input.dim() == 2, "'input' tensor must have 2 dimensions");
    TORCH_CHECK(output.dim() == 2, "'output' tensor must have 2 dimensions");
    TORCH_CHECK(input.size(0) == output.size(0), "The first dimensions of 'input' and 'output' should be equal.");
    TORCH_CHECK(log_prob.sizes() == v.sizes(), "The sizes of 'log_prob' and 'v' should be equal.");

    // Extract dimensions
    auto batch_size = input.size(0);
    auto n = input.size(1);
    auto m = output.size(1);

    /*
        Allocate output tensors.
        The case of reducing the dimensions of gradients.
    */
    auto grad = at::zeros_like(input);
    auto grad_w = at::empty_like(output);
    auto grad_alpha = at::zeros({1}, input.options());

    /*
        Allocate output tensors.
        The case of full gradient, without dimension reduction.
    */
//    auto grad = at::empty({batch_size, m, n}, input.options());
//    auto grad_w = at::empty({batch_size, m, n}, input.options());
//    auto grad_alpha = at::empty({batch_size, m, n}, input.options());

    // Dispatch based on the input type (float or double)
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "derivative", ([&] {
        const auto input_ptr = input.data_ptr<scalar_t>();
        const auto output_ptr = output.data_ptr<scalar_t>();
        const auto log_prob_ptr = log_prob.data_ptr<scalar_t>();
        const auto v_ptr = v.data_ptr<scalar_t>();
        auto grad_ptr = grad.data_ptr<scalar_t>();
        auto grad_w_ptr = grad_w.data_ptr<scalar_t>();
        auto grad_alpha_ptr = grad_alpha.data_ptr<scalar_t>();

        derivative_jvp<scalar_t>(input_ptr, output_ptr, log_prob_ptr, v_ptr, grad_ptr, grad_w_ptr, grad_alpha_ptr,
            n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    return {grad, grad_w, grad_alpha};
}

/**
 * Wrapper to calculate of Vector-Jacobian product (VJP) function for the logarithm of Soft Top-K algorithm and
 * the gradient of the logarithm of probability with respect to the alpha parameter.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Output tensor of Lap-Sum function with shape (batch_size, m).
 * @param log_prob: Logarithm of probability tensor of shape (batch_size, m, n).
 * @param v: Tensor of shape (batch_size, m, n) used in the VJP computation.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: A tuple of gradient tensors with shapes:
 *          - Gradient with respect to input: (batch_size, n)
 *          - Gradient with respect to weights (w): (batch_size, m)
 *          - Gradient with respect to alpha: (1)
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

    // Check tensor dimensions
    TORCH_CHECK(input.dim() == 2, "'input' tensor must have 2 dimensions");
    TORCH_CHECK(output.dim() == 2, "'output' tensor must have 2 dimensions");
    TORCH_CHECK(input.size(0) == output.size(0), "The first dimensions of 'input' and 'output' should be equal.");
    TORCH_CHECK(log_prob.sizes() == v.sizes(), "The sizes of 'log_prob' and 'output' should be equal.");

    // Extract dimensions
    auto batch_size = input.size(0);
    auto n = input.size(1);
    auto m = output.size(1);

    /*
        Allocate output tensors.
        The case of reducing the dimensions of gradients.
    */
    auto grad = at::zeros_like(input);
    auto grad_w = at::empty_like(output);
    auto grad_alpha = at::zeros({1}, input.options());

    /*
        Allocate output tensors.
        The case of full gradient, without dimension reduction.
    */
//    auto grad = at::empty({batch_size, m, n}, input.options());
//    auto grad_w = at::empty({batch_size, m, n}, input.options());
//    auto grad_alpha = at::empty({batch_size, m, n}, input.options());

    // Dispatch based on the input type (float or double)
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "derivative", ([&] {
        const auto input_ptr = input.data_ptr<scalar_t>();
        const auto output_ptr = output.data_ptr<scalar_t>();
        const auto log_prob_ptr = log_prob.data_ptr<scalar_t>();
        const auto v_ptr = v.data_ptr<scalar_t>();
        auto grad_ptr = grad.data_ptr<scalar_t>();
        auto grad_w_ptr = grad_w.data_ptr<scalar_t>();
        auto grad_alpha_ptr = grad_alpha.data_ptr<scalar_t>();

        derivative_vjp<scalar_t>(input_ptr, output_ptr, log_prob_ptr, v_ptr, grad_ptr, grad_w_ptr, grad_alpha_ptr,
             n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    return {grad, grad_w, grad_alpha};
}

/**
 * Pybind11 module definition for the Soft-Top-k extension.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Soft-Top-k forward (CPU)");
    m.def("jvp", &jvp, "Soft-Top-k jvp (CPU)");
    m.def("vjp", &vjp, "Soft-Top-k vjp -- backward (CPU)");
}
