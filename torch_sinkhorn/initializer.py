import abc
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from torch_sinkhorn.problem import LinearProblem


class SinkhornInitializer(abc.ABC):

    @abc.abstractmethod
    def init_dual_a(
        self,
        ot_prob: LinearProblem,
    ) -> torch.Tensor:
        """Initialize Sinkhorn potential f_u.

        Returns:
        potential size ``[..., n,]``.
        """

    @abc.abstractmethod
    def init_dual_b(
        self,
        ot_prob: LinearProblem,
    ) -> torch.Tensor:
        """Initialize Sinkhorn potential g_v.

        Returns:
        potential size ``[..., m,]``.
        """

    def __call__(
        self,
        ot_prob: LinearProblem,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        n, m = ot_prob.C.shape[-2:]
        if a is None:
            a = self.init_dual_a(ot_prob)
        if b is None:
            b = self.init_dual_b(ot_prob)

        assert a.shape[-1] == n, f"Expected `f_u` to have shape `{..., n}`, found `{a.shape}`."
        assert b.shape[-1] == m, f"Expected `g_v` to have shape `{..., m}`, found `{b.shape}`."

        a = torch.where(ot_prob.a > 0., a, -torch.inf)
        b = torch.where(ot_prob.b > 0., b, -torch.inf)

        return a, b


class DefaultInitializer(SinkhornInitializer):

    def init_dual_a(
        self,
        ot_prob: LinearProblem,
    ) -> torch.Tensor:
        return torch.zeros_like(ot_prob.a)

    def init_dual_b(
        self,
        ot_prob: LinearProblem,
    ) -> torch.Tensor:
        return torch.zeros_like(ot_prob.b)


class RandomInitializer(SinkhornInitializer):

    def __init__(
        self,
        seed: Optional[int] = None,
    ):
        self.seed = seed

    def init_dual_a(
        self,
        ot_prob: LinearProblem,
    ) -> torch.Tensor:
        return torch.randn_like(ot_prob.a)

    def init_dual_b(
        self,
        ot_prob: LinearProblem,
    ) -> torch.Tensor:
        return torch.randn_like(ot_prob.b)
