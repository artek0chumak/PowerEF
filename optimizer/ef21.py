import torch

from .utils import orthogonalize, rectanglize


class EF21:
    def __init__(self, rank):
        self.rank = rank

    def add_groups(self, optimizer):
        with torch.no_grad():
            for group in optimizer.param_groups:
                group["G"] = [
                    rectanglize(torch.zeros_like(p)).to(p.device) if len(p.size()) > 1 else None
                    for p in group["params"]
                ]
                group["Q"] = [
                    torch.randn(rectanglize(p).size(1), self.rank).to(p.device) if len(p.size()) > 1 else None
                    for p in group["params"]
                ]

        optimizer.approx = EF21.approx

    @staticmethod
    def approx(group, idx):
        param = group["params"][idx]
        grad = rectanglize(param.grad)
        c = grad - group["G"][idx]
        p = c @ group["Q"][idx]
        orthogonalize(p)
        group["Q"][idx] = c.T @ p
        c = p @ group["Q"][idx].T
        group["G"][idx] += c
        return group["G"][idx].reshape(param.size())


class EF21Plus(EF21):
    def add_groups(self, optimizer):
        super().add_groups(optimizer)

        with torch.no_grad():
            for group in optimizer.param_groups:
                group["Q_bias"] = [
                    torch.randn(rectanglize(p).size(1), self.rank).to(p.device) if len(p.size()) > 1 else None
                    for p in group["params"]
                ]

        optimizer.approx = EF21Plus.approx

    @staticmethod
    def approx(group, idx):
        param = group["params"][idx]
        grad = rectanglize(param.grad)

        p_bias = grad @ group["Q_bias"][idx]
        orthogonalize(p_bias)
        group["Q_bias"][idx] = grad.T @ p_bias
        bias_grad = p_bias @ group["Q_bias"][idx].T
        bias_grad = bias_grad.reshape(param.size())

        markov_grad = EF21.approx(group, idx)

        is_bias_distortion_lower = (
                torch.norm(param.grad - bias_grad).item() < torch.norm(param.grad - markov_grad).item()
        )
        if is_bias_distortion_lower:
            return bias_grad
        return markov_grad
