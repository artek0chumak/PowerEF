import torch


@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0) * col


def rectanglize(tensor):
    return tensor.view(tensor.size(-1), -1)
