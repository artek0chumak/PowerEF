import torch
from torch.optim import Optimizer
import wandb

from .utils import orthogonalize, squarize


class PowerSGD_EF21(Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, rank=8):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        with torch.no_grad():
            for group in self.param_groups:
                group["G"] = [
                    squarize(torch.zeros_like(p)) if len(p.size()) > 1 else None
                    for p in group["params"]
                ]
                group["Q"] = [
                    torch.randn(squarize(p).size(1), rank) if len(p.size()) > 1 else None
                    for p in group["params"]
                ]
                group["momentum_buffer"] = [
                    torch.zeros_like(p) if momentum > 0 else None
                    for p in group["params"]
                ]

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum_buffer_list = group["momentum_buffer"]
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            before_grad_diff = 0
            group_approx_error = 0
            grad_norms = dict()

            for idx, param in enumerate(group['params']):
                if len(param.size()) == 1:
                    grad = param.grad
                else:
                    grad = squarize(param.grad)
                    c = grad - group["G"][idx]
                    before_grad_diff += c.norm().item()
                    r = c @ group["Q"][idx]
                    orthogonalize(r)
                    group["Q"][idx] = c.T @ r
                    c = r @ group["Q"][idx].T
                    group["G"][idx] += c
                    grad = group["G"][idx].reshape(param.size())

                group_approx_error += torch.norm(param.grad - grad).item() ** 2

                if momentum_buffer_list[idx] is not None:
                    change = lr * grad + momentum * momentum_buffer_list[idx]
                else:
                    change = lr * grad
                param -= change
                momentum_buffer_list[idx] = lr * grad + momentum * momentum_buffer_list[idx]
                grad_norms[f"grad_{idx}"] = param.grad.norm()
                grad_norms[f"approx_grad_{idx}"] = grad.norm()
            wandb.log(
                {
                    "grad_apprx_diff": group_approx_error,
                    "before_grad_diff": before_grad_diff,
                    **grad_norms
                },
                commit=False
            )
        return loss
