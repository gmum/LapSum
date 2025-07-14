//
// Created by lstruski
//

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>

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

/**
 **************************************************************************************************************
 *                                               Forward pass
 **************************************************************************************************************
**/

/*
 * solveLap wrapper function to compute intermediate values and results.
 * @param input: Input array.
 * @param w: Weight array.
 * @param A, B: Intermediate arrays.
 * @param results: Output array to store results.
 * @param n: Number of elements in the input array.
 * @param m: Number of elements in the weight array.
 * @param batch_size: Number of batches.
 * @param alpha: Scale parameter of the Laplace distribution.
 */
template <typename T>
void solveLap_wrapper(const T* input, const T* w, T* A, T* B, T* results,
                     const int n, const int m, const int batch_size, const T alpha) {
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
    for (int j = 0; j < batch_size; ++j) {
        i = 0;
        l = 0;
        while (i < m) {
            if (l == 0) {
                if (w[j * m + i] <= 0.5 * (A[j * n] - B[j * n] + 1)) {
                    results[j * m + i] = solveExp(
                        input[j * n], input[j * n], 1 + A[j * n],
                        static_cast<T>(0), static_cast<T>(0), w[j * m + i], alpha
                    );
                    ++i;
                } else {
                    ++l;
                }
            }
            if (l > 0 && l < n) {
                if (0.5 * (A[j * n + l - 1] - B[j * n + l - 1] + 1) + static_cast<T>((j * n + l - 1) % n) < w[j * m + i] && w[j * m + i] <= 0.5 * (A[j * n + l] - B[j * n + l] + 1) + static_cast<T>((j * n + l) % n)) {
                    results[j * m + i] = solveExp(
                        input[j * n + l - 1], input[j * n + l],
                        1 + A[j * n + l], 1 + B[j * n + l - 1], static_cast<T>(l), w[j * m + i], alpha
                    );
                    ++i;
                } else {
                    ++l;
                }
            }
            if (l == n) {
                if (w[j * m + i] > 0.5 * (A[(j + 1) * n - 1] - B[(j + 1) * n - 1] + static_cast<T>(((j + 1) * n - 1) % n))) {
                    results[j * m + i] = solveExp(
                        input[(j + 1) * n - 1], input[(j + 1) * n - 1], static_cast<T>(0),
                        1 + B[(j + 1) * n - 1], static_cast<T>(n), w[j * m + i], alpha
                    );
                    ++i;
                }
            }
        }
    }
}

/**
 * Computes the difference of the CDF of the Laplace distribution at two points.
 *
 * @param a The lower bound.
 * @param b The upper bound (a < b).
 * @param alpha The scale parameter of the Laplace distribution.
 * @return The difference CDF(b, alpha) - CDF(a, alpha).
 */
double diffCDF(double a, double b, double alpha) {
    if (a >= b) {
        throw std::invalid_argument("Invalid input: a (" + std::to_string(a) + ") must be less than b (" + std::to_string(b) + ").");
    }

    if (b <= 0) {
        return -0.5 * std::exp(b / alpha) * std::expm1((a - b) / alpha);
    }
    if (a >= 0) {
        return -0.5 * std::exp(-a / alpha) * std::expm1((a - b) / alpha);
    }
    // a < 0 < b
    return -0.5 * (std::expm1(-b / alpha) + std::expm1(a / alpha));
}

/**
 * Compute the double stochastic matrix mapping (x, w) -> output.
 *
 * This function computes a double stochastic matrix for each batch of input and output elements.
 * The matrix has dimensions (n x (m + 1)) for each batch.
 *
 * @param input Input array of shape (batch_size, n), containing sorted input values.
 * @param output Output array of shape (batch_size, m), containing output values.
 * @param dStochMat Output array of shape (batch_size, n, m + 1), to store the double stochastic matrix.
 * @param n Number of input elements per batch.
 * @param m Number of output elements per batch.
 * @param batch_size Number of batches.
 * @param alpha Scale parameter for the Laplace distribution.
 */
template <typename T>
void doubleStochasticMatrix_wrapper(const T* input, const T* output, T* dStochMat,
                    const int n, const int m, const int batch_size, const T alpha) {
    int input_offset, output_offset, matrix_offset, row_offset;
    T x;
    for (int bs = 0; bs < batch_size; ++bs) {
        input_offset = bs * n;
        output_offset = bs * m;
        matrix_offset = bs * n * (m + 1);

        for (int i = 0; i < n; ++i) {
            row_offset = matrix_offset + i * (m + 1);
            x = input[input_offset + i];

            // Compute the first element in the row
            dStochMat[row_offset] = cdfLap(output[output_offset] - x, alpha);

            // Compute the differences for intermediate columns
            for (int j = 1; j < m; ++j) {
                dStochMat[row_offset + j] = diffCDF(output[output_offset + j - 1] - x,
                                                   output[output_offset + j] - x, alpha);
            }

            // Compute the last element in the row
            dStochMat[row_offset + m] = cdfLap(x - output[output_offset + m - 1], alpha);
        }
    }
}


/**
 * solveLap mapping (x, w) -> b.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n).
 * @param w: Weight tensor of shape (batch_size, m). Must be sorted in ascending order as sorted_input.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: result tensor b.
 */
at::Tensor solveLap(at::Tensor sorted_input, at::Tensor w, const float alpha) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure sorted_input and w are CPU tensors and contiguous
    CHECK_INPUT(sorted_input);
    CHECK_INPUT(w);

    // Extract dimensions
    auto batch_size = sorted_input.size(0);
    auto n = sorted_input.size(1);
    auto m = w.size(1);

    // Ensure sorted_input and w have the correct dimensions
    TORCH_CHECK(sorted_input.dim() == 2, "Tensor 'sorted_input' must have 2 dimensions");
    TORCH_CHECK(w.dim() == 2, "Tensor 'w' must have 2 dimensions");
    TORCH_CHECK(batch_size == w.size(0), "The first dimensions of 'sorted_input' and 'w' should be equal.");

    // Allocate output tensor
    auto results = at::empty_like(w);

    {
        // Allocate intermediate tensors
        auto A = at::empty_like(sorted_input);
        auto B = at::empty_like(sorted_input);

        // Dispatch based on the sorted_input type (float or double)
        AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "solveLap_wrapper", ([&] {
            const auto sorted_input_ptr = sorted_input.data_ptr<scalar_t>();
            const auto w_ptr = w.data_ptr<scalar_t>();
            auto A_ptr = A.data_ptr<scalar_t>();
            auto B_ptr = B.data_ptr<scalar_t>();
            auto results_ptr = results.data_ptr<scalar_t>();

            solveLap_wrapper<scalar_t>(sorted_input_ptr, w_ptr, A_ptr, B_ptr, results_ptr,
                                      n, m, batch_size, static_cast<scalar_t>(alpha));
        }));
    }

    return results;
}


/**
 * Compute the double stochastic matrix mapping (x, w) -> output.
 *
 * This function computes a double stochastic matrix for each batch of input and output elements.
 * The matrix has dimensions (n x (m + 1)) for each batch.
 *
 * @param input: Input tensor of shape (batch_size, n).
 * @param output: Weight tensor of shape (batch_size, m). Output of solveLap function.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: stochastic matrix that has shape (batch_size, n, (m + 1)).
 */
at::Tensor doubleStochasticMatrix(at::Tensor input, at::Tensor output, const float alpha) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure sorted_input and w are CPU tensors and contiguous
    CHECK_INPUT(input);
    CHECK_INPUT(output);

    // Extract dimensions
    auto batch_size = input.size(0);
    auto n = input.size(1);
    auto m = output.size(1);

    // Ensure sorted_input and w have the correct dimensions
    TORCH_CHECK(input.dim() == 2, "Tensor 'input' must have 2 dimensions");
    TORCH_CHECK(output.dim() == 2, "Tensor 'output' must have 2 dimensions");
    TORCH_CHECK(batch_size == output.size(0), "The first dimensions of 'input' and 'output' should be equal.");

    // Allocate output tensor
    auto result = at::empty({batch_size, n, m + 1}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "doubleStochasticMatrix_wrapper", ([&] {
        const auto input_ptr = input.data_ptr<scalar_t>();
        const auto output_ptr = output.data_ptr<scalar_t>();
        auto result_ptr = result.data_ptr<scalar_t>();

        doubleStochasticMatrix_wrapper<scalar_t>(input_ptr, output_ptr, result_ptr,
                                                 n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    return result;
}

/**
 * Compute the logarithm of the CDF of the Laplace distribution at a given point.
 *
 * @param a The input value.
 * @param alpha The scale parameter of the Laplace distribution.
 * @return log(CDF(a))
 */
template <typename T>
T logCDFl(T a, T alpha) {
    if (a <= 0) {
        return a / alpha - std::log(2);
    }
    return std::log(1 - 0.5 * std::exp(-a / alpha));
}

/**
 * Compute the logarithm of one minus the CDF of the Laplace distribution at a given point.
 *
 * @param b The input value.
 * @param alpha The scale parameter of the Laplace distribution.
 * @return log(1 - CDF(b))
 */
template <typename T>
T logCDFu(T b, T alpha) {
    if (b >= 0) {
        return -b / alpha - std::log(2);
    }
    return std::log(1 - 0.5 * std::exp(b / alpha));
}

/**
 * Compute the logarithm of the difference between the CDFs of the Laplace distribution at two points.
 * Assumes a < b.
 *
 * @param a The lower bound.
 * @param b The upper bound.
 * @param alpha The scale parameter of the Laplace distribution.
 * @return log(CDF(b) - CDF(a))
 */
template <typename T>
T logCDF(T a, T b, T alpha) {
    if (a >= b) {
        throw std::invalid_argument("Invalid input: a (" + std::to_string(a) + ") must be less than b (" + std::to_string(b) + ").");
    }

    if (b <= 0) {
        return b / alpha - std::log(2) + std::log(-std::expm1((a - b) / alpha));
    }
    if (a >= 0) {
        return -a / alpha - std::log(2) + std::log(-std::expm1((a - b) / alpha));
    }
    // a < 0 < b
    return std::log(-std::expm1(-b / alpha) - std::expm1(a / alpha)) - std::log(2);
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
template <typename T>
void lossCE_wrapper(const T* input, const T* output, const long* target, T* loss,
            const int n, const int m, const int batch_size, const T alpha) {
    T batch_loss, input_value;
    int t;
    for (int bs = 0; bs < batch_size; ++bs) {
        batch_loss = 0;

        // Compute loss for each element in the batch
        for (int i = 0; i < n; ++i) {
            t = target[bs * n + i];
            input_value = input[bs * n + i];
            if (t == 0) {
                // First class: logCDF from -∞ to output[0]
                batch_loss -= logCDFl(output[bs * m + t] - input_value, alpha);
            } else if (t == n - 1) {
                // Last class: logCDF from output[m-1] to +∞
                batch_loss -= logCDFu(output[bs * m + t - 1] - input_value, alpha);
            } else {
                // Intermediate classes: logCDF difference
                batch_loss -= logCDF(output[bs * m + t - 1] - input_value, output[bs * m + t] - input_value, alpha);
            }
        }
        // Store the computed loss for the batch
        loss[bs] = batch_loss;
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
at::Tensor lossCE(at::Tensor input, at::Tensor output, at::Tensor target, const float alpha) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure sorted_input and w are CPU tensors and contiguous
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(target);

    // Extract dimensions
    auto batch_size = input.size(0);
    auto n = input.size(1);
    auto m = output.size(1);

    // Ensure sorted_input and w have the correct dimensions
    TORCH_CHECK(input.dim() == 2, "Tensor 'input' must have 2 dimensions");
    TORCH_CHECK(output.dim() == 2, "Tensor 'output' must have 2 dimensions");
    TORCH_CHECK(target.sizes() == input.sizes(), "Tensor 'target' must have the same shape as 'input'");
    TORCH_CHECK(batch_size == output.size(0), "The first dimensions of 'input' and 'output' should be equal.");
    TORCH_CHECK(m == n - 1, "The size of the second dimension of the 'output' should be smaller by 1 than the size of the second dimension of the 'input'");

    // Allocate output tensor
    auto result = at::empty({batch_size}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "doubleStochasticMatrix_wrapper", ([&] {
        const auto input_ptr = input.data_ptr<scalar_t>();
        const auto output_ptr = output.data_ptr<scalar_t>();
        const auto target_ptr = target.data_ptr<long>();
        auto result_ptr = result.data_ptr<scalar_t>();

        lossCE_wrapper<scalar_t>(input_ptr, output_ptr, target_ptr, result_ptr,
                                 n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    return result;
}


/**
 **************************************************************************************************************
 *                                            Calculate gradients
 **************************************************************************************************************
**/

/**
 * Template function to compute the normalization factor.
 *
 * @param sorted_input: Pointer to the sorted input array of size (batch_size * n).
 * @param output: Pointer to the output array of size (batch_size * m).
 * @param normLog: Pointer to the result array of size (batch_size * m) to store normalization factors.
 * @param n: Number of elements in each batch of the sorted_input array.
 * @param m: Number of elements in each batch of the output array.
 * @param batch_size: Number of batches to process.
 * @param alpha: Scaling factor for normalization.
**/
template <typename T>
void calculation_normalization_factor(
    const T* sorted_input, const T* output, T* normLog,
    const int n, const int m, const int batch_size, const T alpha) {

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
 * Compute the Jacobian-vector product (JVP) for mapping (x, w) -> output.
 *
 * This function calculates the gradient of the output with respect to the input, given the
 * Laplace distribution parameter alpha and vector v. The operation is batched.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n), sorted by value.
 * @param output: Output tensor of shape (batch_size, m).
 * @param normLog: Normalization log tensor of shape (batch_size, m).
 * @param grad: Gradient output tensor of shape (batch_size, m).
 * @param sorted_v: Vector tensor of shape (batch_size, n), aligned with sorted_input.
 * @param n: Number of input elements per batch.
 * @param m: Number of output elements per batch.
 * @param batch_size: Number of batches.
 * @param alpha: Scale parameter for the Laplace distribution.
 */
template <typename T>
void solveLap_jvp_wrapper(const T* sorted_input, const T* output, const T* normLog, T* grad, const T* sorted_v,
                    const int n, const int m, const int batch_size, const T alpha) {
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
 * Compute the vector-Jacobian product (VJP) for the mapping (x, w) -> output.
 *
 * This function computes the VJP by leveraging the Laplace distribution for
 * smooth approximation and batching the operations for efficiency.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n), sorted by value.
 * @param output: Output tensor of shape (batch_size, m).
 * @param normLog: Normalization log tensor of shape (batch_size, m).
 * @param grad: Gradient output tensor of shape (batch_size, n).
 * @param v: Vector tensor of shape (batch_size, m).
 * @param n: Number of input elements per batch.
 * @param m: Number of output elements per batch.
 * @param batch_size: Number of batches.
 * @param alpha: Scale parameter for the Laplace distribution.
 */
template <typename T>
void solveLap_vjp_wrapper(const T* sorted_input, const T* output, const T* normLog, T* grad, const T* v,
                    const int n, const int m, const int batch_size, const T alpha) {
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
 * Jacobian-vector product (JVP) function for the mapping (x, w) -> output.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n), sorted by value.
 * @param output: Output tensor of shape (batch_size, m).
 * @param sorted_v: Vector tensor of shape (batch_size, n), aligned with sorted_input.
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, m).
 */
std::vector<at::Tensor> solveLap_jvp(at::Tensor sorted_input, at::Tensor output, at::Tensor sorted_v, const float alpha) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(sorted_input);
    CHECK_INPUT(output);
    CHECK_INPUT(sorted_v);

    // Check tensor dimensions
    TORCH_CHECK(sorted_input.dim() == 2, "'sorted_input' tensor must have 2 dimensions");
    TORCH_CHECK(output.dim() == 2, "'w' tensor must have 2 dimensions");
    TORCH_CHECK(sorted_v.sizes() == sorted_input.sizes(), "Tensor 'v' must have the same shape as 'sorted_input'.");
    TORCH_CHECK(sorted_input.size(0) == output.size(0), "The first dimensions of 'input' and 'w' should be equal.");

    // Extract dimensions
    auto batch_size = sorted_input.size(0);
    auto n = sorted_input.size(1);
    auto m = output.size(1);

    // Allocate output tensors
    auto normLog = at::empty_like(output);
    auto grad = at::zeros_like(output);

    // Dispatch based on the input type (float or double)
    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "calculation_normalization_factor", ([&] {
        const auto sorted_input_ptr = sorted_input.data_ptr<scalar_t>();
        const auto w_out_ptr = output.data_ptr<scalar_t>();
        auto normLog_ptr = normLog.data_ptr<scalar_t>();

        calculation_normalization_factor(sorted_input_ptr, w_out_ptr, normLog_ptr,
                                         n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "solveLap_jvp_wrapper", ([&] {
        const auto sorted_input_ptr = sorted_input.data_ptr<scalar_t>();
        const auto w_out_ptr = output.data_ptr<scalar_t>();
        const auto normLog_ptr = normLog.data_ptr<scalar_t>();
        auto grad_ptr = grad.data_ptr<scalar_t>();
        const auto sorted_v_ptr = sorted_v.data_ptr<scalar_t>();

        solveLap_jvp_wrapper<scalar_t>(sorted_input_ptr, w_out_ptr, normLog_ptr, grad_ptr, sorted_v_ptr,
                                       n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    return {grad, normLog};
}

/**
 * Vector-Jacobian product (VJP) function for the mapping (x, w) -> output.
 *
 * @param sorted_input: Input tensor of shape (batch_size, n), sorted by value.
 * @param output: Output tensor of shape (batch_size, m).
 * @param v: Vector tensor of shape (batch_size, m).
 * @param alpha: Scale parameter for the Laplace distribution.
 * @return: Gradient tensor of shape (batch_size, n).
 */
//at::Tensor solveLap_vjp(at::Tensor sorted_input, at::Tensor output, at::Tensor v, at::Tensor indices, const float alpha) {
at::Tensor solveLap_vjp(at::Tensor sorted_input, at::Tensor output, at::Tensor v, const float alpha) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(sorted_input);
    CHECK_INPUT(output);
    CHECK_INPUT(v);

    // Check tensor dimensions
    TORCH_CHECK(sorted_input.dim() == 2, "'sorted_input' tensor must have 2 dimensions");
    TORCH_CHECK(output.dim() == 2, "'w' tensor must have 2 dimensions");
    TORCH_CHECK(v.sizes() == output.sizes(), "Tensor 'v' must have the same shape as 'output'.");
    TORCH_CHECK(sorted_input.size(0) == output.size(0), "The first dimensions of 'sorted_input' and 'w' should be equal.");

    // Extract dimensions
    auto batch_size = sorted_input.size(0);
    auto n = sorted_input.size(1);
    auto m = output.size(1);

    // Allocate output tensors
    auto normLog = at::empty_like(output);
    auto grad = at::zeros_like(sorted_input);

    // Dispatch based on the input type (float or double)
    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "calculation_normalization_factor", ([&] {
        const auto sorted_input_ptr = sorted_input.data_ptr<scalar_t>();
        const auto w_out_ptr = output.data_ptr<scalar_t>();
        auto normLog_ptr = normLog.data_ptr<scalar_t>();

        calculation_normalization_factor(sorted_input_ptr, w_out_ptr, normLog_ptr,
                                         n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    AT_DISPATCH_FLOATING_TYPES(sorted_input.scalar_type(), "solveLap_vjp_wrapper", ([&] {
        const auto sorted_input_ptr = sorted_input.data_ptr<scalar_t>();
        const auto w_out_ptr = output.data_ptr<scalar_t>();
        const auto normLog_ptr = normLog.data_ptr<scalar_t>();
        auto grad_ptr = grad.data_ptr<scalar_t>();
        const auto v_ptr = v.data_ptr<scalar_t>();

        solveLap_vjp_wrapper<scalar_t>(sorted_input_ptr, w_out_ptr, normLog_ptr, grad_ptr, v_ptr,
                                       n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    return grad;
//    return at::empty_like(grad).scatter_(1, indices, grad);
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
template <typename T>
T PDFdivCDF(T a, T b, T alpha) {
    if (a >= b) {
        throw std::invalid_argument("Invalid input: a (" + std::to_string(a) + ") must be less than b (" + std::to_string(b) + ").");
    }

    if (b <= 0) {
        return static_cast<T>(1);
    }
    if (a >= 0) {
        return static_cast<T>(-1);
    }
    // a < 0 < b
    T exp_a = std::expm1(a / alpha);
    T exp_neg_b = std::expm1(-b / alpha);
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
template <typename T>
T PDFadivCDF(T a, T b, T alpha) {
    if (a >= b) {
        throw std::invalid_argument("Invalid input: a (" + std::to_string(a) + ") must be less than b (" + std::to_string(b) + ").");
    }

    if (b <= 0) {
        return -std::exp((a - b) / alpha) / std::expm1((a - b) / alpha);
    }
    if (a >= 0) {
        return static_cast<T>(-1) / std::expm1((a - b) / alpha);
    }
    // a < 0 < b
    T exp_a = std::exp(a / alpha);
    T exp_neg_b = std::expm1(-b / alpha);
    T exp_a_over_alpha = std::expm1(a / alpha);
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
template <typename T>
T PDFbdivCDF(T a, T b, T alpha) {
    if (a >= b) {
        throw std::invalid_argument("Invalid input: a (" + std::to_string(a) + ") must be less than b (" + std::to_string(b) + ").");
    }

    if (b <= 0) {
        return static_cast<T>(-1) / std::expm1((a - b) / alpha);
    }
    if (a >= 0) {
        return -std::exp((a - b) / alpha) / std::expm1((a - b) / alpha);
    }
    // a < 0 < b
    T exp_neg_b = std::exp(-b / alpha);
    T exp_neg_b_expm1 = std::expm1(-b / alpha);
    T exp_a_over_alpha = std::expm1(a / alpha);
    return -exp_neg_b / (exp_neg_b_expm1 + exp_a_over_alpha);
}


/*
 * Auxiliary function to compute a value based on the input x.
 * @param x: Input value of type T.
 * @return: Computed value based on the input x.
 */
template <typename T>
T auxiliary_function(T x) {
    return (x > 0.5) ? (1 - x) / x : static_cast<T>(1);
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
template <typename T>
void derivative_lossCE_wrapper(const T* input, const T* output, const long* target, T* D_input, T* D_output, T* D_alpha,
                     const int n, const int m, const int batch_size, const T alpha) {
    T r, a, b, diff, pdf_a_div_cdf, pdf_b_div_cdf, pdf_div_cdf, common_term, extra_term;
    int target_index;
    // Initialize gradients to zero
    for (int bs = 0; bs < batch_size; ++bs) {
        for (int i = 0; i < n; ++i) {
            target_index = target[bs * n + i];
            r = input[bs * n + i];

            if (target_index == 0) {
                // First class: derivative of -log(CDF(output[0] - r, alpha))
                diff = output[bs * m] - r;
                a = cdfLap(diff, alpha);
                b = auxiliary_function(a);

                D_input[bs * n + i] += b / alpha;
                D_output[bs * m] -= b / alpha;
                D_alpha[bs] += diff * b / (alpha * alpha);

            } else if (target_index == n - 1) {
                // Last class: derivative of -log(1 - CDF(r - output[m-1], alpha))
                diff = r - output[bs * m + target_index - 1];
                a = cdfLap(diff, alpha);
                b = auxiliary_function(a);

                D_input[bs * n + i] -= b / alpha;
                D_output[bs * m + target_index - 1] += b / alpha;
                D_alpha[bs] += diff * b / (alpha * alpha);

            } else {
                // Intermediate classes: derivative of -log(CDF(b - r, alpha) - CDF(a - r, alpha))
                a = output[bs * m + target_index - 1];
                b = output[bs * m + target_index];

                pdf_a_div_cdf = PDFadivCDF(a - r, b - r, alpha);
                pdf_b_div_cdf = PDFbdivCDF(a - r, b - r, alpha);
                pdf_div_cdf = PDFdivCDF(a - r, b - r, alpha);

                common_term = (b - r) * pdf_div_cdf / (alpha * alpha);
                extra_term = (b - a) * pdf_a_div_cdf / (alpha * alpha);

                D_input[bs * n + i] += pdf_div_cdf / alpha;
                D_output[bs * m + target_index - 1] += pdf_a_div_cdf / alpha;
                D_output[bs * m + target_index] -= pdf_b_div_cdf / alpha;
                D_alpha[bs] += common_term + extra_term;
            }
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
std::vector<at::Tensor> derivative_lossCE(at::Tensor input, at::Tensor output, at::Tensor target, const float alpha) {
    // Ensure alpha is not zero
    assert((alpha != 0) && "Parameter 'alpha' can not be zero!");

    // Ensure input tensors are valid
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(target);

    // Check tensor dimensions
    TORCH_CHECK(input.dim() == 2, "'input' tensor must have 2 dimensions");
    TORCH_CHECK(output.dim() == 2, "'output' tensor must have 2 dimensions");
    TORCH_CHECK(target.sizes() == input.sizes(), "Tensor 'target' must have the same shape as 'input'.");
    TORCH_CHECK(input.size(0) == output.size(0), "The first dimensions of 'input' and 'w' should be equal.");

    // Extract dimensions
    auto batch_size = input.size(0);
    auto n = input.size(1);
    auto m = output.size(1);
    TORCH_CHECK(m == n - 1, "The size of the second dimension of the 'output' should be smaller by 1 than the size of the second dimension of the 'input'");

    // Allocate output tensor
    auto D_input = at::zeros_like(input);
    auto D_output = at::zeros_like(output);
    auto D_alpha = at::zeros({batch_size}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "derivative_lossCE_wrapper", ([&] {
        const auto input_ptr = input.data_ptr<scalar_t>();
        const auto output_ptr = output.data_ptr<scalar_t>();
        const auto target_ptr = target.data_ptr<long>();
        auto D_input_ptr = D_input.data_ptr<scalar_t>();
        auto D_output_ptr = D_output.data_ptr<scalar_t>();
        auto D_alpha_ptr = D_alpha.data_ptr<scalar_t>();

        derivative_lossCE_wrapper<scalar_t>(input_ptr, output_ptr, target_ptr, D_input_ptr, D_output_ptr, D_alpha_ptr,
                                            n, m, batch_size, static_cast<scalar_t>(alpha));
    }));

    return {D_input, D_output, D_alpha};
}

/**
 * Pybind11 module definition for the Soft-permute extension.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solveLap", &solveLap, "solveLap forward (CPU)");
    m.def("solveLap_jvp", &solveLap_jvp, "solveLap jvp (CPU)");
    m.def("solveLap_vjp", &solveLap_vjp, "solveLap vjp (CPU)");
    m.def("stochastic_matrix", &doubleStochasticMatrix, "Double Stochastic Matrix (CPU)");
    m.def("lossCE", &lossCE, "Cross-Entropy loss (CPU)");
    m.def("DlossCE", &derivative_lossCE, "derivative of Cross-Entropy loss (CPU)");
}
