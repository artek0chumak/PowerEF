import torch
import math
import wandb

from torch.optim import Optimizer


class ApproxSGD(Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
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

        for group in self.param_groups:
            group["momentum_buffer"] = [
                torch.zeros_like(p).to(p.device) if momentum > 0 else None
                for p in group["params"]
            ]

        self.approx = None

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
            group_approx_error = 0
            grad_norms = dict()

            for idx, param in enumerate(group['params']):
                if len(param.size()) == 1:
                    grad = param.grad
                else:
                    if self.approx:
                        grad_norms[f"grad_{idx}"] = param.grad.norm().item()
                        grad = self.approx(group, idx)
                        grad_norms[f"approx_grad_{idx}"] = grad.norm().item()
                    else:
                        grad = param.grad

                group_approx_error += torch.norm(param.grad - grad).item() ** 2

                if momentum_buffer_list[idx] is not None:
                    change = (1 - momentum) * grad + momentum * momentum_buffer_list[idx]
                    momentum_buffer_list[idx] = (1 - momentum) * grad + momentum * momentum_buffer_list[idx]
                else:
                    change = grad
                param -= lr * change

            wandb.log(
                {
                    "grad_apprx_diff": math.sqrt(group_approx_error),
                    **grad_norms
                },
                commit=False
            )
        return loss
