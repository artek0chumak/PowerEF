import torch
import math
import wandb

from torch.optim import Optimizer


class ApproxAdam(Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.99):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2)
        super().__init__(params, defaults)

        for group in self.param_groups:
            group["first_momentum_buffer"] = [
                torch.zeros_like(p).to(p.device) for p in group["params"]
            ]
            group["second_momentum_buffer"] = [
                torch.zeros_like(p).to(p.device) for p in group["params"]
            ]

        self.num_iters = 0
        self.eps = 1e-8

        self.approx = None

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        self.num_iters += 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            first_momentum_buffer_list = group["first_momentum_buffer"]
            second_momentum_buffer_list = group["second_momentum_buffer"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
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

                first_momentum_buffer_list[idx] = beta1 * first_momentum_buffer_list[idx] + (1 - beta1) * grad
                second_momentum_buffer_list[idx] = beta2 * second_momentum_buffer_list[idx] + (1 - beta2) * grad ** 2
                unbiased_first_momentum = first_momentum_buffer_list[idx] / (1 - beta1 ** self.num_iters)
                unbiased_second_momentum = second_momentum_buffer_list[idx] / (1 - beta2 ** self.num_iters)
                change = unbiased_first_momentum / torch.sqrt(unbiased_second_momentum + self.eps)
                param -= lr * change

            wandb.log(
                {
                    "grad_apprx_diff": math.sqrt(group_approx_error),
                    **grad_norms
                },
                commit=False
            )
        return loss
