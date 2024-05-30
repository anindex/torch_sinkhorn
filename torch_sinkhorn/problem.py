from typing import Union, Optional, Literal
import torch
import numpy as np
from torch_sinkhorn.utils import safe_log


def nanmax(M: torch.Tensor):
    min_value = torch.finfo(M.dtype).min
    output = torch.max(torch.nan_to_num(M, nan=min_value))
    return output


class Epsilon:
    """Epsilon scheduler for Sinkhorn and Sinkhorn Step."""

    def __init__(
        self,
        target: float = None,
        scale_epsilon: float = None,
        init: float = 1.0,
        decay: float = 1.0
    ):
        self._target_init = target
        self._scale_epsilon = scale_epsilon
        self._init = init
        self._decay = decay

    @property
    def target(self) -> float:
        """Return the final regularizer value of scheduler."""
        target = 5e-2 if self._target_init is None else self._target_init
        scale = 1.0 if self._scale_epsilon is None else self._scale_epsilon
        return scale * target

    def at(self, iteration: int = 1) -> float:
        """Return (intermediate) regularizer value at a given iteration."""
        if iteration is None:
            return self.target
        # check the decay is smaller than 1.0.
        decay = min(self._decay, 1.0)
        # the multiple is either 1.0 or a larger init value that is decayed.
        multiple = max(self._init * (decay ** iteration), 1.0)
        return multiple * self.target

    def done(self, eps: float) -> bool:
        """Return whether the scheduler is done at a given value."""
        return eps == self.target

    def done_at(self, iteration: int) -> bool:
        """Return whether the scheduler is done at a given iteration."""
        return self.done(self.at(iteration))
    

class LinearEpsilon(Epsilon):

    def __init__(self, target: float = 0.1, 
                 scale_epsilon: float = 1, 
                 init: float = 1, 
                 decay: float = 1):
        super().__init__(target, scale_epsilon, init, decay)
    
    def at(self, iteration: int = 1) -> float:
        if iteration is None:
            return self.target
        
        eps = max(self._init - self._decay * iteration, self.target)
        return eps * self._scale_epsilon


class LinearProblem():

    def __init__(
        self,
        C: torch.Tensor,
        epsilon: Union[Epsilon, float] = 0.01,
        a: torch.Tensor = None,
        b: torch.Tensor = None,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        scale_cost: Union[int, float, Literal["mean", "max_cost", "median"]] = 'max_cost',
    ) -> None:
        self._epsilon = epsilon
        batch_dim = C.shape[:-2]
        self.a = a if a is not None else (torch.ones(batch_dim + (C.shape[-2],)).type_as(C) / C.shape[-2])
        self.b = b if b is not None else (torch.ones(batch_dim + (C.shape[-1],)).type_as(C) / C.shape[-1])
        self.tau_a = tau_a
        self.tau_b = tau_b
        self._kernel_matrix = None
        self._cost_matrix = C
        self._scale_cost = scale_cost

    def potential_from_scaling(self, scaling: torch.Tensor) -> torch.Tensor:
        return self.epsilon * safe_log(scaling)

    def scaling_from_potential(self, potential: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(potential)
        return torch.where(
            finite, torch.exp(torch.where(finite, potential / self.epsilon, 0.0)), 0.0
        )

    def marginal_from_scalings(
            self,
            u: torch.Tensor,
            v: torch.Tensor,
            dim: int = -2,
        ) -> torch.Tensor:
            """Output marginal of transportation matrix from scalings."""
            u, v = (v, u) if dim == -2 else (u, v)
            return u * self.apply_kernel(v, dim=dim)

    def marginal_from_potentials(
        self, f: torch.Tensor, g: torch.Tensor, dim: int = -2
    ) -> torch.Tensor:
        h = (f if dim == -1 else g)
        z = self.apply_lse_kernel(f, g, self.epsilon, dim=dim)
        return torch.exp((z + h) / self.epsilon)

    def update_potential(
        self, f: torch.Tensor, g: torch.Tensor, log_marginal: torch.Tensor,
        iteration: int = None, dim: int = -2,
    ) -> torch.Tensor:
        app_lse = self.apply_lse_kernel(f, g, self.epsilon, dim=dim)
        return self.epsilon * log_marginal - torch.where(torch.isfinite(app_lse), app_lse, 0)
    
    def update_scaling(
        self,
        scaling: torch.Tensor,
        marginal: torch.Tensor,
        iteration: Optional[int] = None,
        dim: int = -2,
    ) -> torch.Tensor:
        eps = self._epsilon.at(iteration) if isinstance(self._epsilon, Epsilon) else self._epsilon
        app_kernel = self.apply_kernel(scaling, eps, dim=dim)
        return marginal / torch.where(app_kernel > 0, app_kernel, 1.0)

    def transport_from_potentials(
        self, f: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """Output transport matrix from potentials."""
        return torch.exp(self._center(f, g) / self.epsilon)

    def transport_from_scalings(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Output transport matrix from pair of scalings."""
        return self.K * u[..., :, None] * v[..., None, :]

    def apply_kernel(
            self,
            scaling: torch.Tensor,
            eps: Optional[float] = None,
            dim: int = -2,
        ) -> torch.Tensor:
            if eps is None:
                kernel = self.K
            else:
                kernel = self.K ** (self.epsilon / eps)
            kernel = kernel if dim == -1 else kernel.mT
            return torch.einsum("...ij,...j->...i", kernel, scaling)

    def apply_lse_kernel(
        self, f: torch.Tensor, g: torch.Tensor, eps: float, dim: int
    ) -> torch.Tensor:
        w_res = self._softmax(f, g, eps, dim=dim)
        remove = f if dim == -1 else g
        return w_res - torch.where(torch.isfinite(remove), remove, 0)

    def _center(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return f[..., :, None] + g[..., None, :] - self.C

    def _softmax(
        self, f: torch.Tensor, g: torch.Tensor, eps: float, dim: int
    ) -> torch.Tensor:
        """Apply softmax row or column wise"""

        lse_output = torch.logsumexp(
            self._center(f, g) / eps, dim=dim
        )
        return eps * lse_output

    @property
    def C(self) -> torch.Tensor:
        if self._cost_matrix is None:
            eps = torch.finfo(self._kernel_matrix.dtype).tiny
            cost = -safe_log(self._kernel_matrix + eps)
            cost *= self.inv_scale_cost
            return self.epsilon * cost
        return self._cost_matrix * self.inv_scale_cost

    @property
    def K(self) -> torch.Tensor:
        if self._kernel_matrix is None:
            return torch.exp(-(self._cost_matrix * self.inv_scale_cost / self.epsilon))
        return self._kernel_matrix ** self.inv_scale_cost

    @property
    def inv_scale_cost(self) -> torch.Tensor:
        if isinstance(self._scale_cost, (int, float, np.number)):
            return 1.0 / self._scale_cost
        if self._scale_cost == "max_cost":
            return 1.0 / nanmax(self._cost_matrix)
        if self._scale_cost == "mean":
            return 1.0 / torch.nanmean(self._cost_matrix, dim=(-2, -1), keepdim=True)
        if self._scale_cost == "median":
            return 1.0 / torch.nanmedian(self._cost_matrix, dim=(-2, -1), keepdim=True)
        raise ValueError(f"Scaling {self._scale_cost} not implemented.")

    @property
    def epsilon(self) -> float:
        return self._epsilon.target if isinstance(self._epsilon, Epsilon) else self._epsilon

    @property
    def is_balanced(self) -> bool:
        return self.tau_a == 1.0 and self.tau_b == 1.0

    @property
    def is_uniform(self) -> bool:
        return self.a is None and self.b is None

    @property
    def is_equal_size(self) -> bool:
        return self.C.shape[-1] == self.C.shape[-2]
