import torch
from ..extension import soft_permute_cpu as ext_cpu
from ..extension import soft_permute_cuda as ext_cuda


def solveLap(sorted_x, sorted_w, alpha):
    if sorted_x.is_cuda:
        return ext_cuda.solveLap(sorted_x, sorted_w, alpha)
    return ext_cpu.solveLap(sorted_x, sorted_w, alpha)

def jvp(sorted_x, y, sorted_v, alpha):
    if sorted_x.is_cuda:
        return ext_cuda.solveLap_jvp(sorted_x, y, sorted_v, alpha)
    return ext_cpu.solveLap_jvp(sorted_x, y, sorted_v, alpha)

def vjp(sorted_x, y, v, indices, alpha):
    if sorted_x.is_cuda:
        grad = ext_cuda.solveLap_vjp(sorted_x, y, v, alpha)
    else:
        grad = ext_cpu.solveLap_vjp(sorted_x, y, v, alpha)
    return torch.empty_like(grad).scatter_(1, indices, grad)

def stochM(x, b, alpha):
    if x.is_cuda:
        return ext_cuda.stochastic_matrix(x, b, alpha)
    return ext_cpu.stochastic_matrix(x, b, alpha)

def lossCE(x, b, target, alpha):
    if x.is_cuda:
        return ext_cuda.lossCE(x, b, target, alpha)
    return ext_cpu.lossCE(x, b, target, alpha)

def D_lossCE(x, b, target, alpha):
    if x.is_cuda:
        return ext_cuda.DlossCE(x, b, target, alpha)
    return ext_cpu.DlossCE(x, b, target, alpha)
