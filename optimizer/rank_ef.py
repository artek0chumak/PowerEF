import torch

from .utils import orthogonalize, rectanglize


class RankEF:
    def __init__(self, rank):
        self.rank = rank

    def add_groups(self, optimizer):
        with torch.no_grad():
            for group in optimizer.param_groups:
                group["M"] = [
                    rectanglize(torch.zeros_like(p)) if len(p.size()) > 1 else None
                    for p in group["params"]
                ]
                group["Q"] = [
                    torch.randn(rectanglize(p).size(1), self.rank) if len(p.size()) > 1 else None
                    for p in group["params"]
                ]

        optimizer.approx = RankEF.approx

    @staticmethod
    def approx(group, idx):
        param = group["params"][idx]
        grad = rectanglize(param.grad)
        group["M"][idx] += grad
        r = group["M"][idx] @ group["Q"][idx]
        orthogonalize(r)
        group["Q"][idx] = group["M"][idx].T @ r
        group["M"][idx] = group["M"][idx] - r @ group["Q"][idx].T
        return (r @ group["Q"][idx].T).reshape(param.size())
