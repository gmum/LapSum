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

// Template implementation of logaddexp for numeric types.
template <typename T>
T logaddexp(T x, T y) {
    // Handle the case where one of the inputs is negative infinity.
    if (x == -std::numeric_limits<T>::infinity()) return y;
    if (y == -std::numeric_limits<T>::infinity()) return x;

    // Ensure numerical stability by computing the result in a way
    // that avoids overflow and underflow issues.
    if (x > y) {
        return x + std::log1p(std::exp(y - x));
    } else {
        return y + std::log1p(std::exp(x - y));
    }
}

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
 * Function to compute the cumulative distribution function (CDF) of a Laplace distribution.
 * @param x: Input value.
 * @param alpha: Scale parameter of the Laplace distribution.
 * @return: CDF value at x.
 */
template <typename T>
T cdfLap(T x, T alpha) {
    T val = x / alpha;
    if (val <= 0) {
        return 0.5 * std::exp(val);
    }
    return 1 - 0.5 * std::exp(-val);
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
    if (w < c && b > 0) {
        T diff = c - w;
        return r + alpha * std::log(b) - alpha * std::log(
            diff + std::sqrt(diff * diff + std::exp((r - s) / alpha) * a * b));
    }
    return 0.5 * (r + s);
}

/*
 * Forward wrapper function to compute intermediate values and results.
 * @param input: Input array.
 * @param A, B: Intermediate arrays.
 * @param results: Output array to store results.
 * @param n: Number of elements in the input array.
 * @param batch_size: Number of batches.
 * @param alpha: Scale parameter of the Laplace distribution.
 */
template <typename T>
void forward_wrapper(const T* input, T* A, T* B, T* results,
                     const int n, const int batch_size, const T alpha) {
    int idx, a_idx, b_idx;
    for (int bs = 0; bs < batch_size; ++bs) {
        idx = bs * n;

        B[idx] = 0;
        A[idx + n - 1] = 0;
        for (int i = 0; i < n - 1; ++i) {
            b_idx = idx + i;
            a_idx = idx + n - i - 1;

            B[b_idx + 1] = std::exp((input[b_idx] - input[b_idx + 1]) / alpha) * (1 + B[b_idx]);
            A[a_idx - 1] = std::exp((input[a_idx - 1] - input[a_idx]) / alpha) * (1 + A[a_idx]);
        }
    }

    int i, l;
    T w_val;
    for (int j = 0; j < batch_size; ++j) {
        i = 0;
        l = 0;
        while (i < n) {
            w_val = i + 0.5;
            if (l == 0) {
                if (w_val <= 0.5 * (A[j * n] - B[j * n] + 1)) {
                    results[j * n + i] = solveExp(
                        input[j * n], input[j * n], 1 + A[j * n],
                        static_cast<T>(0), static_cast<T>(0), w_val, alpha
                    );
                    ++i;
                } else {
                    ++l;
                }
            }
            if (l > 0 && l < n) {
                if (0.5 * (A[j * n + l - 1] - B[j * n + l - 1] + 1) + static_cast<T>((j * n + l - 1) % n) < w_val && w_val <= 0.5 * (A[j * n + l] - B[j * n + l] + 1) + static_cast<T>((j * n + l) % n)) {
                    results[j * n + i] = solveExp(
                        input[j * n + l - 1], input[j * n + l],
                        1 + A[j * n + l], 1 + B[j * n + l - 1], static_cast<T>(l), w_val, alpha
                    );
                    ++i;
                } else {
                    ++l;
                }
            }
            if (l == n) {
                if (w_val > 0.5 * (A[(j + 1) * n - 1] - B[(j + 1) * n - 1] + static_cast<T>(((j + 1) * n - 1) % n))) {
                    results[j * n + i] = solveExp(
                        input[(j + 1) * n - 1], input[(j + 1) * n - 1], static_cast<T>(0),
                        1 + B[(j + 1) * n - 1], static_cast<T>(n), w_val, alpha
                    );
                    ++i;
                }
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
 * Performs the forward pass for the Soft-Sort algorithm.
 * This function computes the output of the Lap-Sort operation and the associated probability tensor
 * based on the input tensor, weight tensor, and the Laplace distribution scale parameter.
 *
 * @param input: Input tensor of shape (batch_size, n) that has the sorted values.
 * @param alpha: Scale parameter for the Laplace distribution, controlling the sharpness of the distribution.
 * @return: The output tensor of the Lap-Sort function with shape (batch_size, n).
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

    // Allocate output tensors
    auto results = at::empty_like(input);

    {
        // Allocate intermediate tensors
        auto A = at::empty_like(input);
        auto B = at::empty_like(input);

        // Dispatch based on the input type (float or double)
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "forward_wrapper", ([&] {
            const auto input_ptr = input.data_ptr<scalar_t>();
            auto A_ptr = A.data_ptr<scalar_t>();
            auto B_ptr = B.data_ptr<scalar_t>();
            auto results_ptr = results.data_ptr<scalar_t>();

            forward_wrapper<scalar_t>(input_ptr, A_ptr, B_ptr, results_ptr,
                                      n, batch_size, static_cast<scalar_t>(alpha));
        }));
    }

    return results;
}


/**
 **************************************************************************************************************
 *                                            Calculate gradients
 **************************************************************************************************************
**/

/**
 * Template function to compute the normalization factor.
 *
 * @param sorted_input: Pointer to the sorted input array of size (batch_size, n).
 * @param output: Pointer to the output array of size (batch_size, n).
 * @param normLog: Pointer to the result array of size (batch_size, n) to store normalization factors.
 * @param n: Number of elements in each batch of the sorted_input array.
 * @param batch_size: Number of batches to process.
 * @param alpha: Scaling factor for normalization.
**/
template <typename T>
void calculation_normalization_factor(
    const T* sorted_input, const T* output, T* normLog, const int n, const int batch_size, const T alpha) {
    const int& m = n;

    // Temporary storage for for intermediate values
    std::vector<T> nU(m);
    std::vector<T> nL(m);

    int ju, jl, i, j;
    for (int bs = 0; bs < batch_size; ++bs) {
        // Initialize nU and nL
        std::fill(nU.begin(), nU.end(), 0);
        std::fill(nL.begin(), nL.end(), 0);

        // Compute nU values
        ju = 0;
        while (ju < m && output[bs * m + ju] - sorted_input[bs * n] <= 0) {
            ++ju;
        }
        if (ju < n) {
            nU[ju] = (sorted_input[bs * n] - output[bs * m + ju]) / alpha;

            i = 1;
            while (i < n && output[bs * m + ju] - sorted_input[bs * n + i] > 0) {
                nU[ju] = logaddexp(nU[ju], (sorted_input[bs * n + i] - output[bs * m + ju]) / alpha);
                ++i;
            }

            for (j = ju + 1; j < m; ++j) {
                nU[j] = (output[bs * m + j - 1] - output[bs * m + j]) / alpha + nU[j - 1];
                while (i < n && output[bs * m + j] - sorted_input[bs * n + i] > 0) {
                    nU[j] = logaddexp(nU[j], (sorted_input[bs * n + i] - output[bs * m + j]) / alpha);
                    ++i;
                }
            }
        }

        // Compute nL values
        jl = m - 1;
        while (jl > 0 && output[bs * m + jl] - sorted_input[(bs + 1) * n - 1] >= 0) {
            --jl;
        }
        if (jl > 0) {
            nL[jl] = (output[bs * m + jl] - sorted_input[(bs + 1) * n - 1]) / alpha;

            i = n - 2;
            while (i >= 0 && output[bs * m + jl] - sorted_input[bs * n + i] <= 0) {
                nL[jl] = logaddexp(nL[jl], (output[bs * m + jl] - sorted_input[bs * n + i]) / alpha);
                --i;
            }

            for (j = jl - 1; j >= 0; --j) {
                nL[j] = (output[bs * m + j] - output[bs * m + j + 1]) / alpha + nL[j + 1];
                while (i >= 0 && output[bs * m + j] - sorted_input[bs * n + i] <= 0) {
                    nL[j] = logaddexp(nL[j], (output[bs * m + j] - sorted_input[bs * n + i]) / alpha);
                    --i;
                }
            }
        }

        // Combine nU and nL into normLog
        i = (ju < jl + 1) ? ju : jl + 1;
        j = (ju < jl + 1) ? jl + 1 : ju;
        for (int l = 0; l < m; ++l) {
            if (l < i) {
                normLog[bs * m + l] = nL[l];
            } else if (l >= j) {
                normLog[bs * m + l] = nU[l];
            } else {
                normLog[bs * m + l] = logaddexp(nL[l], nU[l]);
            }
        }
    }
}

/**
 * Compute the Jacobian-vector product (JVP)
 *
 * This function calculates the gradient of the output with respect to the input, given the
 * Laplace distribution parameter alpha and vector v. The operation is batched.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n), sorted by value.
 * @param output: Output tensor of shape (batch_size, n).
 * @param normLog: Normalization log tensor of shape (batch_size, n).
 * @param grad: Gradient output tensor of shape (batch_size, n).
 * @param sorted_v: Vector tensor of shape (batch_size, n), aligned with sorted_input.
 * @param n: Number of input elements per batch.
 * @param batch_size: Number of batches.
 * @param alpha: Scale parameter for the Laplace distribution.
 */
template <typename T>
void solveLap_jvp_wrapper(const T* sorted_input, const T* output, const T* normLog, T* grad, const T* sorted_v,
                    const int n, const int batch_size, const T alpha) {
    const int& m = n;
    int i, j;

    // Temporary storage for intermediate values
    std::vector<T> nvU(m);

    for (int bs = 0; bs < batch_size; ++bs) {
        // Initialize nvU
        std::fill(nvU.begin(), nvU.end(), 0);

        // Compute forward contributions to nvU
        i = 0;
        while (i < n && output[bs * m] - sorted_input[bs * n + i] > 0) {
            nvU[0] += std::exp((sorted_input[bs * n + i] - output[bs * m]) / alpha - normLog[bs * m]) * sorted_v[bs * n + i];;
            ++i;
        }
        for (j = 1; j < m; ++j) {
            nvU[j] = std::exp((output[bs * m + j - 1] - output[bs * m + j]) / alpha - (normLog[bs * m + j] - normLog[bs * m + j - 1])) * nvU[j - 1];
            while (i < n && output[bs * m + j] - sorted_input[bs * n + i] > 0) {
                nvU[j] += std::exp((sorted_input[bs * n + i] - output[bs * m + j]) / alpha - normLog[bs * m + j]) * sorted_v[bs * n + i];
                ++i;
            }
        }
        i = n - 1;
        // Compute backward contributions to nvL = grad
        while (i >= 0 && output[(bs + 1) * m - 1] - sorted_input[bs * n + i] <= 0) {
            grad[(bs + 1) * m - 1] += std::exp((output[(bs + 1) * m - 1] - sorted_input[bs * n + i]) / alpha - normLog[(bs + 1) * m - 1]) * sorted_v[bs * n + i];
            --i;
        }
        for (j = m - 2; j >= 0; --j) {
            grad[bs * m + j] = std::exp((output[bs * m + j] - output[bs * m + j + 1]) / alpha - (normLog[bs * m + j] - normLog[bs * m + j + 1])) * grad[bs * m + j + 1];
            while (i >= 0 && output[bs * m + j] - sorted_input[bs * n + i] <= 0) {
                grad[bs * m + j] += std::exp((output[bs * m + j] - sorted_input[bs * n + i]) / alpha - normLog[bs * m + j]) * sorted_v[bs * n + i];
                --i;
            }
        }

        for (j = 0; j < m; ++j) {
            grad[bs * m + j] += nvU[j];
        }
    }
}

/**
 * Compute the vector-Jacobian product (VJP)
 *
 * This function computes the VJP by leveraging the Laplace distribution for
 * smooth approximation and batching the operations for efficiency.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n), sorted by value.
 * @param output: Output tensor of shape (batch_size, n).
 * @param normLog: Normalization log tensor of shape (batch_size, n).
 * @param grad: Gradient output tensor of shape (batch_size, n).
 * @param v: Vector tensor of shape (batch_size, n).
 * @param n: Number of input elements per batch.
 * @param batch_size: Number of batches.
 * @param alpha: Scale parameter for the Laplace distribution.
 */
template <typename T>
void solveLap_vjp_wrapper(const T* sorted_input, const T* output, const T* normLog, T* grad, const T* v,
                    const int n, const int batch_size, const T alpha) {
    const int& m = n;
    int i, j;

    // Temporary storage for intermediate values
    std::vector<T> kU(n);

    for (int bs = 0; bs < batch_size; ++bs) {
        // Initialize nvU
        std::fill(kU.begin(), kU.end(), 0);

        // Compute forward contributions to kU
        i = 0;
        while (i < m && output[bs * m + i] - sorted_input[bs * n] <= 0) {
            kU[0] += std::exp((output[bs * m + i] - sorted_input[bs * n]) / alpha - normLog[bs * m + i]) * v[bs * m + i];
            ++i;
        }
        for (j = 1; j < n; ++j) {
            kU[j] = std::exp((sorted_input[bs * n + j - 1] - sorted_input[bs * n + j]) / alpha) * kU[j - 1];
            while (i < m && output[bs * m + i] - sorted_input[bs * n + j] <= 0) {
                kU[j] += std::exp((output[bs * m + i] - sorted_input[bs * n + j]) / alpha - normLog[bs * m + i]) * v[bs * m + i];
                ++i;
            }
        }

        i = m - 1;
        // Compute backward contributions to kL = grad
        while (i >= 0 && output[bs * m + i] - sorted_input[(bs + 1) * n - 1] > 0) {
            grad[(bs + 1) * n - 1] += std::exp((sorted_input[(bs + 1) * n - 1] - output[bs * m + i]) / alpha - normLog[bs * m + i]) * v[bs * m + i];
            --i;
        }
        for (j = n - 2; j >= 0; --j) {
            grad[bs * n + j] = std::exp((sorted_input[bs * n + j] - sorted_input[bs * n + j + 1]) / alpha) * grad[bs * n + j + 1];
            while (i >= 0 && output[bs * m + i] - sorted_input[bs * n + j] > 0) {
                grad[bs * n + j] += std::exp((sorted_input[bs * n + j] - output[bs * m + i]) / alpha - normLog[bs * m + i]) * v[bs * m + i];
                --i;
            }
        }

        for (j = 0; j < n; ++j) {
            grad[bs * n + j] += kU[j];
        }
    }
}


/**
 * Jacobian-vector product (JVP)
 *
 * @param sorted_input: Input tensor of shape (batch_size, n), sorted by value.
 * @param output: Output tensor of shape (batch_size, n).
 * @param sorted_v: Vector tensor of shape (batch_size, n), aligned with sorted_input.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, n).
 */
at::Tensor solveLap_jvp(at::Tensor sorted_input, at::Tensor output, at::Tensor sorted_v, const float alpha) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(sorted_input);
    CHECK_INPUT(output);
    CHECK_INPUT(sorted_v);

    // Check tensor dimensions
    TORCH_CHECK(sorted_input.dim() == 2, "'sorted_input' tensor must have 2 dimensions");
    TORCH_CHECK(sorted_input.sizes() == output.sizes(), "Tensor 'sorted_input' must have the same shape as 'output'.");
    TORCH_CHECK(sorted_v.sizes() == sorted_input.sizes(), "Tensor 'v' must have the same shape as 'sorted_input'.");

    // Extract dimensions
    auto batch_size = sorted_input.size(0);
    auto n = sorted_input.size(1);

    // Allocate output tensors
    auto normLog = at::empty_like(output);
    auto grad = at::zeros_like(output);

    // Dispatch based on the input type (float or double)
    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "calculation_normalization_factor", ([&] {
        const auto sorted_input_ptr = sorted_input.data_ptr<scalar_t>();
        const auto w_out_ptr = output.data_ptr<scalar_t>();
        auto normLog_ptr = normLog.data_ptr<scalar_t>();

        calculation_normalization_factor(sorted_input_ptr, w_out_ptr, normLog_ptr,
                                         n, batch_size, static_cast<scalar_t>(alpha));
    }));

    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "solveLap_jvp_wrapper", ([&] {
        const auto sorted_input_ptr = sorted_input.data_ptr<scalar_t>();
        const auto w_out_ptr = output.data_ptr<scalar_t>();
        const auto normLog_ptr = normLog.data_ptr<scalar_t>();
        auto grad_ptr = grad.data_ptr<scalar_t>();
        const auto sorted_v_ptr = sorted_v.data_ptr<scalar_t>();

        solveLap_jvp_wrapper<scalar_t>(sorted_input_ptr, w_out_ptr, normLog_ptr, grad_ptr, sorted_v_ptr,
                                       n, batch_size, static_cast<scalar_t>(alpha));
    }));

    return grad;
}

/**
 * Vector-Jacobian product (VJP)
 *
 * @param sorted_input: Input tensor of shape (batch_size, n), sorted by value.
 * @param output: Output tensor of shape (batch_size, n).
 * @param v: Vector tensor of shape (batch_size, n).
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, n).
 */
at::Tensor solveLap_vjp(at::Tensor sorted_input, at::Tensor output, at::Tensor v, const float alpha) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(sorted_input);
    CHECK_INPUT(output);
    CHECK_INPUT(v);

    // Check tensor dimensions
    TORCH_CHECK(sorted_input.dim() == 2, "'sorted_input' tensor must have 2 dimensions");
    TORCH_CHECK(sorted_input.sizes() == output.sizes(), "Tensor 'sorted_input' must have the same shape as 'output'.");
    TORCH_CHECK(v.sizes() == sorted_input.sizes(), "Tensor 'v' must have the same shape as 'sorted_input'.");

    // Extract dimensions
    auto batch_size = sorted_input.size(0);
    auto n = sorted_input.size(1);

    // Allocate output tensors
    auto normLog = at::empty_like(output);
    auto grad = at::zeros_like(sorted_input);

    // Dispatch based on the input type (float or double)
    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "calculation_normalization_factor", ([&] {
        const auto sorted_input_ptr = sorted_input.data_ptr<scalar_t>();
        const auto output_ptr = output.data_ptr<scalar_t>();
        auto normLog_ptr = normLog.data_ptr<scalar_t>();

        calculation_normalization_factor(sorted_input_ptr, output_ptr, normLog_ptr,
                                         n, batch_size, static_cast<scalar_t>(alpha));
    }));

    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "solveLap_vjp_wrapper", ([&] {
        const auto sorted_input_ptr = sorted_input.data_ptr<scalar_t>();
        const auto output_ptr = output.data_ptr<scalar_t>();
        const auto normLog_ptr = normLog.data_ptr<scalar_t>();
        auto grad_ptr = grad.data_ptr<scalar_t>();
        const auto v_ptr = v.data_ptr<scalar_t>();

        solveLap_vjp_wrapper<scalar_t>(sorted_input_ptr, output_ptr, normLog_ptr, grad_ptr, v_ptr,
                                       n, batch_size, static_cast<scalar_t>(alpha));
    }));

    return grad;
}


/**
 * Pybind11 module definition for the Soft-Sort extension.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Soft-Sort forward (CPU)");
    m.def("jvp", &solveLap_jvp, "Soft-Sort jvp (CPU)");
    m.def("vjp", &solveLap_vjp, "Sort-Soft vjp (CPU)");
}
