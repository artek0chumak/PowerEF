import torch

from .utils import orthogonalize, rectanglize


class EF21:
    def __init__(self, rank):
        self.rank = rank

    def add_groups(self, optimizer):
        with torch.no_grad():
            for group in optimizer.param_groups:
                group["G"] = [
                    rectanglize(torch.zeros_like(p)) if len(p.size()) > 1 else None
                    for p in group["params"]
                ]
                group["Q"] = [
                    torch.randn(rectanglize(p).size(1), self.rank) if len(p.size()) > 1 else None
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
