import time
import torch
import numpy as np
from lapsum.topk import soft_topk, log_soft_topk


def numerical_vjp(function, x, k, alpha, largest, v, h=1e-5):
    grad_approx = torch.zeros_like(x)
    for i in range(x.numel()):
        e = torch.zeros_like(x).view(-1)
        e[i] = h  # Perturb one dimension at a time
        e = e.view_as(x)  # Reshape back to original shape

        grad_approx.view(-1)[i] = (
                torch.dot(
                    v.flatten(),
                    (function(x + e, k, alpha, 1, largest) - function(x - e, k, alpha, 1, largest)).flatten()
                ) / (2 * h)
        )
    return grad_approx


def numerical_vjp_k(function, r, k_weights, alpha, largest, v, h=1e-4):
    grad_numerical = torch.zeros_like(k_weights)
    for i in range(len(k_weights)):
        k_plus = k_weights.clone()
        k_plus[i] += h
        output_plus = function(r, k_plus, alpha, 1, largest)

        k_minus = k_weights.clone()
        k_minus[i] -= h
        output_minus = function(r, k_minus, alpha, 1, largest)

        grad_numerical[i] = ((output_plus - output_minus) * v).sum() / (2 * h)
    return grad_numerical


def check_value(x, v, text):
    assert x.shape == v.shape, f"Shape mismatch: {x.shape} vs {v.shape}"
    def fun():
        if isinstance(x, torch.Tensor):
            return torch.allclose, torch.linalg.norm
        else:
            return np.allclose, np.linalg.norm

    function, dist = fun()
    check = None
    for tol_exp in range(-15, 0):
        if function(x, v, rtol=1e-05, atol=10 ** tol_exp):
            check = f"Error within atol=1e{tol_exp}"
            break
    if check:
        print(f"✅ - {text} ({check})")
    else:
        print(f"❌ - {text} [dist: {dist(x - v):.4f}]")
        print(f"Expected: {v}")
        print(f"Got: {x}")


def run_test_suite(device, dtype, function, bs, n, largest, num_runs=5):
    """Run complete test suite for given configuration."""
    print(f"\n{'=' * 50}")
    print(f"Testing {function.__name__} on {device.type.upper()} and precision {dtype}")
    print(f"Batch size: {bs}, Elements: {n}")
    print(f"{'=' * 50}\n")

    factory_kwargs = {"device": device, "requires_grad": True, "dtype": dtype}
    forward_times = []
    backward_times = []

    h = 1e-4
    for i in range(num_runs):
        alpha = torch.tensor(np.clip(np.random.rand(), 0.1, 1), **factory_kwargs)
        r = torch.randn(bs, n, **factory_kwargs)
        k = torch.tensor(np.random.rand(bs, 1) * n, **{**factory_kwargs, "dtype": r.dtype})
        v = torch.randn_like(r)

        # Forward pass
        start = time.perf_counter()
        prob = function(r, k, alpha, dim=1, largest=largest)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - start
        forward_times.append(forward_time)

        # Verify forward
        if function == log_soft_topk:
            test_sum = torch.logsumexp(prob, dim=1, keepdim=True).exp()
        else:
            test_sum = prob.sum(dim=1, keepdim=True)

        check_value(test_sum, k, "Sum verification")

        # Backward pass
        start = time.perf_counter()
        r.grad = None
        k.grad = None
        alpha.grad = None
        prob.backward(v)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        backward_time = time.perf_counter() - start
        backward_times.append(backward_time)

        # Numerical gradient checks
        numerical_derivative = numerical_vjp(function, r, k, alpha, largest, v, h)
        check_value(r.grad, numerical_derivative, "Gradient w.r.t. input")

        numerical_k_grad = numerical_vjp_k(function, r, k, alpha, largest, v, h)
        check_value(k.grad, numerical_k_grad, "Gradient w.r.t. k")

        numerical_alpha_grad = torch.mul(
            v,
            function(r, k, alpha + h, 1, largest) - function(r, k, alpha - h, 1, largest)
        ) / (2 * h)
        check_value(alpha.grad, numerical_alpha_grad.sum(), "Gradient w.r.t. alpha")
        print(f"Run {i + 1}: Forward={forward_time:.4f}s, Backward={backward_time:.4f}s\n")

    # Print statistics
    print(f"\n{'=' * 50}")
    print(f"Performance summary ({function.__name__} on {device.type.upper()})")
    print(f"Forward pass:  Avg={np.mean(forward_times):.4f}s ± {np.std(forward_times):.4f}")
    print(f"Backward pass: Avg={np.mean(backward_times):.4f}s ± {np.std(backward_times):.4f}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":

    print("Running manual tests...")
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        for function in [soft_topk, log_soft_topk]:
            for largest in [True, False]:
                run_test_suite(device, torch.double, function, bs=3, n=200, largest=largest, num_runs=3)
