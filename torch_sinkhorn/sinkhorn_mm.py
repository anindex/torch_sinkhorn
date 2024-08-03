from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    List,
    Union,
)
import numpy as np
import torch
from torch_sinkhorn.problem import LinearProblem
from torch_sinkhorn.initializer import DefaultInitializer, RandomInitializer, SinkhornInitializer
from torch_sinkhorn.utils import safe_log, logsumexp, softmin


def cost_tensor(
    x_s: Tuple[torch.Tensor, ...]
) -> torch.Tensor: # NOTE: Square Euclidean cost function for now

    k = len(x_s)
    ns = [x.shape[0] for x in x_s]
    cost_t = torch.zeros(ns).type_as(x_s[0])

    for i in range(k):
        for j in range(i + 1, k):
            cost_m = torch.sum(torch.square(x_s[i][:, None] - x_s[j][None, :]), dim=-1)
            dim = list(range(i)) + list(range(i + 1, j)) + list(range(j + 1, k))
            for d in dim:  # TODO: improve this
                cost_m = torch.unsqueeze(cost_m, dim=d)
            cost_t += cost_m
    return cost_t


def remove_tensor_sum(
    c: torch.Tensor, u: Tuple[torch.Tensor, ...]
) -> torch.Tensor:

    k = len(u)
    for i in range(k):
        dim = list(range(i)) + list(range(i + 1, k))
        u_i = u[i]
        for d in dim:  # TODO: improve this
            u_i = torch.unsqueeze(u_i, dim=d)
        c -= u_i
    return c


def tensor_marginals(coupling: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return tuple(tensor_marginal(coupling, ix) for ix in range(coupling.ndim))


def tensor_marginal(coupling: torch.Tensor, slice_index: int) -> torch.Tensor:
    k = coupling.ndim
    dim = list(range(slice_index)) + list(range(slice_index + 1, k))
    return coupling.sum(dim=dim)


def coupling_tensor(
    potentials: Tuple[torch.Tensor], cost_t: torch.Tensor, epsilon: float
) -> torch.Tensor:
  return torch.exp(-remove_tensor_sum(cost_t, potentials) / epsilon)


def compute_ent_reg_cost(
    potentials: Tuple[torch.Tensor], cost_t: torch.Tensor, a_s: Tuple[torch.Tensor, ...], epsilon: float
) -> torch.Tensor:
    ent_reg_cost = 0.0
    for potential, a in zip(potentials, a_s):
            pot = torch.where(torch.isfinite(potential), potential, 0)
            ent_reg_cost += torch.sum(pot * a)

    ent_reg_cost += epsilon * (1 - torch.sum(coupling_tensor(potentials, cost_t, epsilon)))
    return ent_reg_cost


class MMSinkhornState():

    def __init__(
        self,
        potentials: Tuple[torch.Tensor, ...],
        errors: torch.Tensor = None,
        costs: torch.Tensor = None,
    ):
        self.errors = errors
        self.costs = costs
        self.potentials = potentials
        self.converged_at = -1

    def solution_error(
        self,
        cost_t: torch.Tensor,
        a_s: Tuple[torch.Tensor, ...],
        epsilon: float,
        norm: float = 2.0
    ) -> float:
        coupl_tensor = coupling_tensor(self.potentials, cost_t, epsilon)
        marginals = tensor_marginals(coupl_tensor)
        errors = torch.tensor([
            torch.norm(marginal - a, p=norm, dim=-1)
            for a, marginal in zip(a_s, marginals)
        ]).type_as(cost_t)
        return torch.sum(errors)

    def ent_reg_cost(
        self, 
        cost_t: torch.Tensor,
        a_s: Tuple[torch.Tensor, ...],
        epsilon: float,
    ) -> float:
        return compute_ent_reg_cost(self.potentials, cost_t, a_s, epsilon)


class MMSinkhornOutput():
    
    def __init__(
        self,
        potentials: Tuple[torch.Tensor, ...],
        costs: torch.Tensor,
        errors: torch.Tensor,
        cost_t: torch.Tensor,
        a_s: Tuple[torch.Tensor, ...],
        epsilon: Optional[float] = None,
        inner_iterations: Optional[int] = None,
        use_danskin: bool = False
    ) -> None:
        self.use_danskin = use_danskin
        if self.use_danskin:
            potentials = [p.detach() for p in potentials]
        self.potentials = potentials
        self.costs = costs
        self.errors = errors
        self.cost_t = cost_t
        self.epsilon = epsilon
        self.ent_reg_cost = compute_ent_reg_cost(potentials, cost_t, a_s, epsilon)
        self.inner_iterations = inner_iterations

    @property
    def n_iters(self) -> int:  # noqa: D102
        """Total number of iterations that were needed to terminate."""
        return torch.sum(self.errors != -1) * self.inner_iterations

    @property
    def tensor(self) -> torch.Tensor:
        """Transport tensor."""
        return torch.exp(
            -remove_tensor_sum(self.cost_t, self.potentials) / self.epsilon
        )

    @property
    def marginals(self) -> Tuple[torch.Tensor, ...]:
        """:math:`k` marginal probability weight vectors."""
        return tensor_marginals(self.tensor)

    def marginal(self, k: int) -> torch.Tensor:
        """Return the marginal probability weight vector at slice :math:`k`."""
        return tensor_marginal(self.tensor, k)

    @property
    def transport_mass(self) -> float:
        """Sum of transport tensor."""
        return torch.sum(self.tensor)


class MMSinkhorn:

    def __init__(
        self,
        threshold: float = 1e-3,
        norm: float = 2.0,
        inner_iterations: int = 1,
        min_iterations: int = 1,
        max_iterations: int = 100,
        use_danskin: bool = True,
    ):
        self.threshold = threshold
        self.inner_iterations = inner_iterations
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.norm = norm
        self.use_danskin = use_danskin

    def __call__(
        self,
        x_s: Tuple[torch.Tensor, ...],
        a_s: Optional[Tuple[torch.Tensor, ...]] = None,
        epsilon: Optional[float] = None
    ) -> MMSinkhornOutput:
        n_s = [x.shape[0] for x in x_s]

        if a_s is None:
            a_s = [torch.ones(n).type_as(x_s[0]) / n for n in n_s]
        else:
            a_s = [(torch.ones(n).type_as(x_s[0]) / n if a is None else a) for a, n in zip(a_s, n_s)]

        assert len(n_s) == len(a_s), (len(n_s), len(a_s))
        for n, a in zip(n_s, a_s):
            assert n == a.shape[0], (n, a.shape[0])

        cost_t = cost_tensor(x_s)
        errors = -torch.ones((self.max_iterations,)).type_as(cost_t)
        costs = -torch.ones((self.max_iterations,)).type_as(cost_t)
        potentials = tuple(torch.zeros(n).type_as(cost_t) for n in n_s)
        state = MMSinkhornState(potentials=potentials, errors=errors, costs=costs)
        self.epsilon = 0.05 * torch.mean(cost_t) if epsilon is None else epsilon

        final_state = self.iterations(cost_t, a_s, state)
        return self.output_from_state(cost_t, a_s, final_state), final_state

    def output_from_state(
        self, cost_t: torch.Tensor, a_s: Tuple[torch.Tensor, ...], state: MMSinkhornState,
    ) -> torch.Tensor:
        return MMSinkhornOutput(
            state.potentials,
            state.costs,
            state.errors,
            cost_t, a_s,
            self.epsilon,
            self.inner_iterations,
            self.use_danskin
        )

    def one_iteration(
        self, cost_t: torch.Tensor, a_s: Tuple[torch.Tensor, ...], state: MMSinkhornState,
        iteration: int, compute_error: bool = True
    ) -> MMSinkhornState:
        k = len(a_s)

        def one_slice(potentials: Tuple[torch.Tensor, ...], l: int, a: torch.Tensor):
            pot = potentials[l]
            dim = list(range(l)) + list(range(l + 1, k))
            app_lse = softmin(
                remove_tensor_sum(cost_t, potentials), self.epsilon, dim=dim
            )
            pot += self.epsilon * torch.log(a) + torch.where(torch.isfinite(app_lse), app_lse, 0)
            return potentials[:l] + (pot,) + potentials[l + 1:]

        for l in range(k):
            state.potentials = one_slice(state.potentials, l, a_s[l])

        if iteration % self.inner_iterations == 0:
            it = iteration // self.inner_iterations
            if compute_error:
                err = state.solution_error(
                    cost_t, a_s, self.epsilon, norm=self.norm
                )
                cost = state.ent_reg_cost(cost_t, a_s, self.epsilon)
            else:
                err = -1
                cost = -1
            state.errors[..., it] = err
            state.costs[..., it] = cost
        return state

    def _converged(self, state: MMSinkhornState, iteration: int) -> bool:
        if iteration < self.min_iterations:
            return False
        it = iteration // self.inner_iterations
        err = state.errors[..., it - 1]
        return (err < self.threshold).all()

    def _diverged(self, state: MMSinkhornState, iteration: int) -> bool:
        it = iteration // self.inner_iterations
        err = torch.isinf(state.errors[..., it - 1]).any() or torch.isnan(state.errors[..., it - 1]).any()
        cost = torch.isinf(state.costs[..., it - 1]).any() or torch.isnan(state.costs[..., it - 1]).any()
        return err or cost

    def _continue(self, state: MMSinkhornState, iteration: int) -> bool:
        return iteration < self.max_iterations and not self._converged(state, iteration) and not self._diverged(state, iteration)

    def iterations(
        self, cost_t: torch.Tensor, a_s: Tuple[torch.Tensor, ...], state: MMSinkhornState, compute_error: bool = True
    ) -> MMSinkhornState:
        iteration = 0
        while self._continue(state, iteration):
            state = self.one_iteration(cost_t, a_s, state, iteration, compute_error=compute_error)
            iteration += 1
        if self._converged(state, iteration):
            state.converged_at = iteration 
        return state


if __name__ == "__main__":
    from torch_sinkhorn.problem import Epsilon
    from torch_sinkhorn.utils import plot_coupling
    from torch_robotics.torch_utils.torch_timer import TimerCUDA
    
    n_s, d = [6] * 4, 2
    x_s = [torch.rand(n, d) for n in n_s]
    a_s = None

    sinkhorn = MMSinkhorn()
    with TimerCUDA() as t:
        W, state = sinkhorn(x_s, a_s)
    print(t.elapsed)
    print(f"Converged at {state.converged_at}")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(state.errors[state.errors != -1])
    plt.figure()
    plt.plot(state.costs[state.costs != -1])
    # plt.figure()
    # plot_coupling(W.tensor[0])
    # plt.show()