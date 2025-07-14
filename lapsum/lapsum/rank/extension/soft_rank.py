import torch
from torch.autograd import Function
from ..extension import soft_rank_cpu as ext_cpu
from ..extension import soft_rank_cuda as ext_cuda


class SoftRank(Function):

    @staticmethod
    def forward(ctx, x, alpha, dim):
        """
        Performs the forward pass for the Soft-Top-k function.

        This function computes the probabilities tensor based on the input tensor, weight tensor,
        and regularization parameter. It also saves necessary information in the context object
        for use during the backward pass.

        Args:
            ctx: Context object to save intermediate results and other information required
                 for gradient computation during the backward pass.
            x: Input tensor with at least 2 dimensions.
            alpha: Regularization parameter (must be non-zero) controlling the smoothness
                   of the Soft-Top-k operation.
            dim: Dimension over which the Soft-Top-k operation is performed.

        Returns:
            prob: Probabilities tensor.
        """
        assert x.ndim == 2, "x must be a 2-dimensional tensor"
        assert alpha != 0, "alpha must not be 0"

        if x.is_cuda:
            output = ext_cuda.forward(x, alpha)
        else:
            output = ext_cpu.forward(x, alpha)

        # Save tensors for backward
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Performs the backward pass for the Soft-Rank function.

        This function computes the gradients of the loss with respect to the input tensor `x`,
        and the regularization parameter `alpha` using the saved tensors and attributes from the forward pass.
        It also computes the vector-Jacobian product (VJP) using the provided `grad_output`.

        Args:
            ctx: Context object containing saved tensors and attributes from the forward pass,
                 required for gradient computation.
            grad_output: Gradient tensor propagated from the subsequent layer, used to compute
                        the vector-Jacobian product.

        Returns:
            grad_input: Gradient of the loss with respect to the input tensor `x`.
            grad_alpha: Gradient of the loss with respect to the regularization parameter `alpha`.
        """
        # Retrieve saved tensors and attributes
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        dim = ctx.dim

        if x.is_cuda:
            grad_output, grad_alpha = ext_cuda.vjp(x, grad_output.contiguous(), alpha)
        else:
            grad_output, grad_alpha = ext_cpu.vjp(x, grad_output.contiguous(), alpha)
        
        return grad_output, grad_alpha, None


def soft_rank(x, alpha: float | torch.Tensor=-1, dim: int=1):
    """
    Wrapper function for the Soft-Rank operation.

    This function serves as a convenient interface to perform the Soft-Top-k operation
    on the input tensor `x` using the weight tensor `w` and the regularization parameter `alpha`.
    The operation is performed over the specified dimension.

    Args:
        x: Input tensor.
        alpha: Regularization parameter controlling the smoothness of the Soft-Top-k operation
               (default: -1, which uses a predefined value or behavior).
        dim: Dimension over which the Soft-Top-k operation is performed.

    Returns:
        The result of the Soft-Top-k operation, typically a tensor of probabilities.
    """
    return SoftRank.apply(x, alpha, dim)
