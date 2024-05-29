import abc
from typing import Any, Dict, Optional, Sequence, Tuple, Literal

import torch

from torch_sinkhorn.problem import LinearProblem


class LRInitializer(abc.ABC):

    def __init__(self, rank: int, **kwargs: Any):
        self.rank = rank
        self._kwargs = kwargs

    @abc.abstractmethod
    def init_q(
        self,
        ot_prob: LinearProblem,
        *,
        init_g: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Initialize low-rank factor q.

        Returns:
            Array of shape ``[n, rank]``.
        """

    @abc.abstractmethod
    def init_r(
        self,
        ot_prob: LinearProblem,
        *,
        init_g: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Initialize low-rank factor r.

        Returns:
            Array of shape ``[m, rank]``.
        """
    
    @abc.abstractmethod
    def init_g(
        self,
        ot_prob: LinearProblem,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Initialize the low-rank factor :math:`g`.

        Returns:
        Array of shape ``[rank,]``.
        """

    def __call__(
      self,
      ot_prob: LinearProblem,
      q: Optional[torch.Tensor] = None,
      r: Optional[torch.Tensor] = None,
      g: Optional[torch.Tensor] = None,
      **kwargs: Any
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize the factors :math:`Q`, :math:`R` and :math:`g`.
        """

        if g is None:
            g = self.init_g(ot_prob, **kwargs)
        if q is None:
            q = self.init_q(ot_prob, init_g=g, **kwargs)
        if r is None:
            r = self.init_r(ot_prob, init_g=g, **kwargs)

        assert g.shape == (self.rank,)
        assert q.shape == (ot_prob.a.shape[0], self.rank)
        assert r.shape == (ot_prob.b.shape[0], self.rank)

        return q, r, g



class RandomInitializer(LRInitializer):

    def init_q(
        self,
        ot_prob: LinearProblem,
        *,
        init_g: torch.Tensor = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        a = ot_prob.a
        init_q = torch.abs(torch.rand((a.shape[0], self.rank))).type_as(a)
        return a[:, None] * (init_q / torch.sum(init_q, dim=1, keepdims=True))

    def init_r(
        self,
        ot_prob: LinearProblem,
        *,
        init_g: torch.Tensor = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs, init_g
        b = ot_prob.b
        init_r = torch.abs(torch.rand((b.shape[0], self.rank))).type_as(b)
        return b[:, None] * (init_r / torch.sum(init_r, dim=1, keepdims=True))

    def init_g(
        self,
        ot_prob: LinearProblem,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        init_g = torch.abs(torch.rand((self.rank,))).type_as(ot_prob.C) + 1.0
        return init_g / torch.sum(init_g)


class Rank2Initializer(LRInitializer):

    def _compute_factor(
        self,
        ot_prob: LinearProblem,
        init_g: torch.Tensor,
        *,
        which: Literal["q", "r"],
    ) -> torch.Tensor:
        a, b = ot_prob.a, ot_prob.b
        marginal = a if which == "q" else b
        n, r = marginal.shape[0], self.rank

        lambda_1 = torch.min(
            torch.tensor([torch.min(a), torch.min(init_g), torch.min(b)])
        ) * 0.5

        g1 = torch.arange(1, r + 1).to(torch.float32)
        g1 /= g1.sum()
        g2 = (init_g - lambda_1 * g1) / (1.0 - lambda_1)

        x = torch.arange(1, n + 1).to(torch.float32)
        x /= x.sum()
        y = (marginal - lambda_1 * x) / (1.0 - lambda_1)

        f = ((lambda_1 * x[:, None] @ g1.reshape(1, -1)) +
                ((1.0 - lambda_1) * y[:, None] @ g2.reshape(1, -1)))
        return f.type_as(a)

    def init_q(
        self,
        ot_prob: LinearProblem,
        *,
        init_g: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._compute_factor(ot_prob, init_g, which="q")

    def init_r(
        self,
        ot_prob: LinearProblem,
        *,
        init_g: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._compute_factor(ot_prob, init_g, which="r")

    def init_g(
        self,
        ot_prob: LinearProblem,
        **kwargs: Any,
    ) -> torch.Tensor:
        return torch.ones((self.rank,)).type_as(ot_prob.C) / self.rank
