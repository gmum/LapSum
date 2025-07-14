from typing import Union

import torch
from torch.autograd import Function
from ..extension import soft_topk_cpu as ext_cpu
from ..extension import soft_topk_cuda as ext_cuda


class SoftTopK(Function):
    @staticmethod
    def forward(ctx, x, k, alpha, dim, largest):
        """
        Performs the forward pass for the Soft-Top-k function.

        Args:
            ctx: Context object for saving information for backward pass.
            x: Input tensor (at least 2D).
            k: Weight tensor (must match x dimensions except on 'dim').
            alpha: Regularization parameter (non-zero).
            dim: Dimension over which Soft-Top-k is performed.
            largest: Whether important are the largest or smallest elements.

        Returns:
            prob: Probabilities tensor with the same shape as x.
        """
        x_dims = list(x.shape)
        original_shape = x_dims.copy()

        assert len(x_dims) >= 2, f"x must be at least 2-dimensional, got {x.dim()}"
        x_dims.pop(dim)
        k_dims = list(k.shape)
        assert len(k_dims) >= 2, f"k must be at least 2-dimensional, got {k.dim()}"
        k_dims.pop(dim)
        assert (
            k_dims == x_dims
        ), f"k dimensions mismatch: k: {list(k.shape)}, x: {original_shape}"
        assert alpha != 0, "alpha must not be 0"

        # Prepare permutation and save original shape
        dims = list(range(x.dim()))
        perm_dims = dims.copy()
        perm_dims.pop(dim)
        perm_dims.append(dim)  # Move target dim to end

        # Prepare reshaped tensors
        if len(x_dims) > 1:
            # Save original shape and permutation for backward
            ctx.original_shape = original_shape
            ctx.perm_dims = perm_dims
            ctx.k_dim = k.dim()

            # Permute and reshape
            r = x.permute(*perm_dims).flatten(0, -2)  # Flatten all but last dim
            w = k.unsqueeze(dim) if k.dim() == x.dim() - 1 else k
            w = w.permute(*perm_dims).flatten(0, -2)
        else:
            r = x
            w = k
            ctx.original_shape = None  # No reshaping needed

        # Process alpha
        if isinstance(alpha, torch.Tensor):
            alpha.data = alpha.abs().neg() if largest else alpha.abs()
        else:
            alpha = -abs(alpha) if largest else abs(alpha)

        # Compute output
        if x.is_cuda:
            output, prob = ext_cuda.forward(r, w, alpha)
        else:
            output, prob = ext_cpu.forward(r, w, alpha)

        # Reshape prob back to original shape
        prob = prob.squeeze(1)
        if ctx.original_shape is not None:
            # Unflatten and inverse permute
            prob = prob.view([*[original_shape[i] for i in perm_dims[:-1]], -1])
            inv_perm = [perm_dims.index(i) for i in range(x.dim())]
            prob = prob.permute(*inv_perm)

        # Save for backward
        ctx.save_for_backward(r, output)
        ctx.alpha = alpha
        ctx.dim = dim

        return prob

    @staticmethod
    def backward(ctx, grad_output):
        """
        Performs the backward pass for the Soft-Top-k function.

        Args:
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient tensor from subsequent layer.

        Returns:
            grad_input: Gradient w.r.t. input x (same shape as x)
            grad_w: Gradient w.r.t. weights k (same shape as k)
            grad_alpha: Gradient w.r.t. alpha
            None: Placeholder for dim
            None: Placeholder for largest
        """
        # Retrieve saved tensors and attributes
        r, output = ctx.saved_tensors
        alpha = ctx.alpha
        dim = ctx.dim

        # Get original shape and permutation if they exist
        original_shape = getattr(ctx, "original_shape", None)
        perm_dims = getattr(ctx, "perm_dims", None)
        k_dim = getattr(ctx, "k_dim", None)

        # Prepare grad_output for VJP
        if grad_output.ndim == 2:
            grad_output = grad_output.unsqueeze(1)

        # Reshape grad_output if needed to match flattened forward pass
        if original_shape is not None:
            # Permute and flatten grad_output to match forward pass shape
            grad_output_flat = grad_output.permute(*perm_dims).flatten(0, -2)
        else:
            grad_output_flat = grad_output

        # Compute VJP
        if r.is_cuda:
            grad_r, grad_w_flat, grad_alpha = ext_cuda.vjp(
                r, output, grad_output_flat.contiguous(), alpha
            )
        else:
            grad_r, grad_w_flat, grad_alpha = ext_cpu.vjp(
                r, output, grad_output_flat.contiguous(), alpha
            )

        # Reshape gradients back to original shapes
        if original_shape is not None:
            # Reshape grad_r back to original x shape
            grad_input = grad_r.view([original_shape[i] for i in perm_dims])
            inv_perm = [perm_dims.index(i) for i in range(len(perm_dims))]
            grad_input = grad_input.permute(*inv_perm)

            # Reshape grad_w back to k's original shape
            grad_w = grad_w_flat.view([original_shape[i] for i in perm_dims[:-1]])
            inv_perm_w = [perm_dims.index(i) for i in range(len(perm_dims))]
            grad_w = grad_w.permute(*inv_perm_w)
            if k_dim is not None and k_dim == len(perm_dims) - 1:
                grad_w = grad_w.squeeze(dim)
        else:
            grad_input = grad_r
            grad_w = grad_w_flat

        return grad_input, grad_w, grad_alpha, None, None


def soft_topk(
    x: torch.Tensor,
    k: torch.Tensor,
    alpha: Union[float, torch.Tensor],
    dim: int = 1,
    largest: bool = True,
) -> torch.Tensor:
    """
    Computes the Soft-Top-k operation with automatic dimension handling and input validation.

    This function provides a smoothed, differentiable version of top-k selection,
    where the hardness of the selection is controlled by the alpha parameter.

    Args:
        x: Input tensor of shape (..., N, ...) where N is the size along dimension `dim`
        k: Weight tensor of shape:
           - Either (..., M, ...) where other dimensions match x (M can differ from N)
           - Or (..., 1, ...) for broadcasting
        alpha: Smoothing parameter (scalar or tensor):
               - Larger values make the operation closer to hard top-k
               - Smaller values make it more uniform
               - Sign controls selection direction (handled automatically based on `largest`)
        dim: Dimension along which to compute the operation (default: 1)
        largest: If True, larger input values have a higher probability; if false, the opposite

    Returns:
        torch.Tensor: Soft top-k probabilities with the same shape as input x

    Examples:
        >>> x = torch.randn(2, 5)  # batch_size=2, features=5
        >>> k = torch.randn(2, 3)  # different size along dim 1
        >>> probs = soft_topk(x, k, alpha=0.1, dim=1)
    """
    # Automatically handle dimension wrapping for negative dim values
    dim = dim if dim >= 0 else x.dim() + dim

    return SoftTopK.apply(x, k, alpha, dim, largest)
