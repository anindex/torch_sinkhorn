import functools
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import torch
import numpy as np


def safe_log(
    x: torch.Tensor,
    *,
    eps: Optional[float] = 1e-12
) -> torch.Tensor:
    return torch.where(x > 0.0, torch.log(x), np.log(eps))


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.flatten()
    q = q.flatten()
    return p @ (safe_log(p) - safe_log(q))


def gen_kl(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.flatten()
    q = q.flatten()
    return p @ (safe_log(p) - safe_log(q)) + torch.sum(q) - torch.sum(p)


def gen_js(p: torch.Tensor, q: torch.Tensor, c: float = 0.5) -> float:
    return c * (gen_kl(p, q) + gen_kl(q, p))


def softmin(
    x: torch.Tensor, gamma: float, dim: Optional[int] = None
) -> torch.Tensor:
    return -gamma * torch.logsumexp(x / -gamma, dim=dim)



def sort_and_argsort(
    x: torch.Tensor,
    *,
    argsort: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if argsort:
        i_x = torch.argsort(x)
        return x[i_x], i_x
    return torch.sort(x), None



def lambertw(
    z: torch.Tensor, tol: float = 1e-8, max_iter: int = 100
) -> torch.Tensor:

    def initial_iacono(x: torch.Tensor) -> torch.Tensor:
        y = torch.sqrt(1.0 + torch.e * x)
        num = 1.0 + 1.14956131 * y
        denom = 1.0 + 0.45495740 * torch.log1p(y)
        return -1.0 + 2.036 * torch.log(num / denom)

    def cond_fun(container):
        it, converged, _ = container
        return torch.any(~converged) and it < max_iter

    def halley_iteration(container):
        it, _, w = container

        f = w - z * torch.exp(-w)
        delta = f / (w + 1.0 - 0.5 * (w + 2.0) * f / (w + 1.0))

        w_next = w - delta

        not_converged = torch.abs(delta) <= tol * torch.abs(w_next)
        return it + 1, not_converged, w_next

    w = initial_iacono(z)
    converged = torch.zeros_like(w, dtype=bool)

    while cond_fun((0, converged, w)):
        _, converged, w = halley_iteration((0, converged, w))

    return w