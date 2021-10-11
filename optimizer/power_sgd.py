import torch
from torch.optim import Optimizer

from .utils import orthogonalize, squarize


class PowerSGD(Optimizer):
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
                group["M"] = [
                    squarize(torch.zeros_like(p)) if len(p.size()) > 1 else None
                    for p in group["params"]
                ]
                group["Q"] = [
                    torch.randn(squarize(p).size(1), rank) if len(p.size()) > 1 else None
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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for idx, param in enumerate(group['params']):
                if len(param.size()) == 1:
                    grad = param.grad
                else:
                    grad = squarize(param.grad)
                    group["M"][idx] += grad
                    r = group["M"][idx] @ group["Q"][idx]
                    orthogonalize(r)
                    group["Q"][idx] = group["M"][idx].T @ r
                    group["M"][idx] = group["M"][idx] - r @ group["Q"][idx].T
                    grad = (r @ group["Q"][idx].T).reshape(param.size())
                param.grad = grad
                param -= lr * grad
        return loss
