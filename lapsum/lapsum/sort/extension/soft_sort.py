import torch
from torch.autograd import Function
from ..extension import soft_sort_cpu as ext_cpu
from ..extension import soft_sort_cuda as ext_cuda


class SoftSort(Function):

    @staticmethod
    def forward(ctx, x, alpha, dim=-1):
        """
        Performs the forward pass for the Soft-Sort function.

        This function computes the probabilities tensor based on the input tensor, weight tensor,
        and regularization parameter. It also saves necessary information in the context object
        for use during the backward pass.

        Args:
            ctx: Context object to save intermediate results and other information required
                 for gradient computation during the backward pass.
            w: Weight tensor with the size along the specified dimension (`dim`) can differ,
               but the remaining dimensions must match those of `x`.
            alpha: Regularization parameter (must be non-zero) controlling the smoothness
                   of the Soft-Sort operation.
            dim: Dimension over which the Soft-Sort operation is performed.

        Returns:
            output: Sorted tensor.
        """
        assert x.ndim == 2, "x must be a 2-dimensional tensor"
        assert alpha != 0, "alpha must not be 0"

        sorted, indices = torch.sort(x, dim=dim, descending=bool(alpha < 0))

        if x.is_cuda:
            output = ext_cuda.forward(sorted, alpha)
        else:
            output = ext_cpu.forward(sorted, alpha)

        # Save tensors for backward
        ctx.save_for_backward(sorted, indices, output)
        ctx.alpha = alpha
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Performs the backward pass for the Soft-Sort function.

        This function computes the gradients of the loss with respect to the input tensor `x`,
        the weight tensor `w`, and the regularization parameter `alpha` using the saved tensors
        and attributes from the forward pass. It also computes the vector-Jacobian product (VJP)
        using the provided `grad_output`.

        Args:
            ctx: Context object containing saved tensors and attributes from the forward pass,
                 required for gradient computation.
            grad_output: Gradient tensor propagated from the subsequent layer, used to compute
                        the vector-Jacobian product.

        Returns:
            grad_input: Gradient of the loss with respect to the input tensor `x`.
            grad_w: Gradient of the loss with respect to the weight tensor `w`.
            grad_alpha: Gradient of the loss with respect to the regularization parameter `alpha`.
        """
        # Retrieve saved tensors and attributes
        sorted, indices, output = ctx.saved_tensors
        alpha = ctx.alpha
        dim = ctx.dim

        if sorted.is_cuda:
            grad_output = ext_cuda.vjp(sorted, output, grad_output.contiguous(), alpha)
        else:
            grad_output = ext_cpu.vjp(sorted, output, grad_output.contiguous(), alpha)

        return torch.empty_like(grad_output).scatter_(1, indices, grad_output), None, None


def soft_sort(x, alpha: float | torch.Tensor=-1, dim: int=1):
    """
    Wrapper function for the Soft-Sort operation.

    This function serves as a convenient interface to perform the Soft-Top-k operation
    on the input tensor `x` using the weight tensor `w` and the regularization parameter `alpha`.
    The operation is performed over the specified dimension.

    Args:
        x: Input tensor.
        alpha: Regularization parameter controlling the smoothness of the Soft-Top-k operation
               (default: -1, which uses a predefined value or behavior).
        dim: Dimension over which the Soft-Sort operation is performed.

    Returns:
        The result of the Soft-Sort operation, typically a tensor of probabilities.
    """
    return SoftSort.apply(x, alpha, dim)
