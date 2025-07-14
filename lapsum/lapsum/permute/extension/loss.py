import torch
from torch.autograd import Function
from ..extension import soft_permute_cpu as ext_cpu
from ..extension import soft_permute_cuda as ext_cuda


class CrossEntropyLoss(Function):

    @staticmethod
    def forward(ctx, x, target, alpha):

        sorted_x, indices = torch.sort(x, dim=1)
        w = torch.arange(1, x.shape[1], dtype=x.dtype, device=x.device)
        w = torch.tile(w.unsqueeze(0), (x.shape[0], 1))

        if x.is_cuda:
            b = ext_cuda.solveLap(sorted_x, w, alpha)
            output = ext_cuda.lossCE(x, b, target, alpha)
        else:
            b = ext_cpu.solveLap(sorted_x, w, alpha)
            output = ext_cpu.lossCE(x, b, target, alpha)

        # Save tensors for backward
        ctx.save_for_backward(x, sorted_x, indices, b, target)
        ctx.alpha = alpha

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.contiguous()

        # Retrieve saved tensors and ctx attributes
        x, sorted_x, indices, b, target = ctx.saved_tensors
        alpha = ctx.alpha

        if x.is_cuda:
            d_x, d_b, d_alpha = ext_cuda.DlossCE(x, b, target, alpha)
            vj = ext_cuda.solveLap_vjp(sorted_x, b, d_b, alpha)
            jv, _ = ext_cuda.solveLap_jvp(sorted_x, b, sorted_x, alpha)
        else:
            d_x, d_b, d_alpha = ext_cpu.DlossCE(x, b, target, alpha)
            vj = ext_cpu.solveLap_vjp(sorted_x, b, d_b, alpha)
            jv, _ = ext_cpu.solveLap_jvp(sorted_x, b, sorted_x, alpha)

        grad = torch.empty_like(vj).scatter_(1, indices, vj)
        return d_x + grad, None, torch.einsum("di,di->d", d_b, (b - jv) / alpha) + d_alpha



# wrap in a function, to support default args and keyword args.
def cross_entropy(x, target, alpha: float|torch.Tensor=1, reduction='mean'):
    loss = CrossEntropyLoss.apply(x, target, alpha)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
