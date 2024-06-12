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
    Union,
)
import numpy as np
import torch
from torch_sinkhorn.problem import LinearProblem
from torch_sinkhorn.initializer import DefaultInitializer, RandomInitializer, SinkhornInitializer
from torch_sinkhorn.utils import safe_log, logsumexp


def phi_star(h: torch.Tensor, rho: float) -> torch.Tensor:
  return rho * (torch.exp(h / rho) - 1)


def rho(epsilon: float, tau: float) -> float:
  return (epsilon * tau) / (1. - tau)


def derivative_phi_star(f: torch.Tensor, rho: float) -> torch.Tensor:
  return torch.exp(f / rho)


def grad_of_marginal_fit(
    c: torch.Tensor, h: torch.Tensor, tau: float, epsilon: float
) -> torch.Tensor:
  if tau == 1.0:
    return c
  r = rho(epsilon, tau)
  return torch.where(c > 0, c * derivative_phi_star(-h, r), 0.0)


def solution_error(
    fu: torch.Tensor,
    gv: torch.Tensor,
    ot_prob: LinearProblem,
    parallel_dual_updates: bool,
    lse_mode: bool = True
) -> torch.Tensor:
    if ot_prob.is_balanced and not parallel_dual_updates:
        return marginal_error(
            fu, gv, ot_prob.b, ot_prob, dim=-2, lse_mode=lse_mode
        )

    grad_a = grad_of_marginal_fit(
        ot_prob.a, fu, ot_prob.tau_a, ot_prob.epsilon
    )
    grad_b = grad_of_marginal_fit(
        ot_prob.b, gv, ot_prob.tau_b, ot_prob.epsilon
    )

    err = marginal_error(fu, gv, grad_a, ot_prob, dim=-1, lse_mode=lse_mode)
    err += marginal_error(fu, gv, grad_b, ot_prob, dim=-2, lse_mode=lse_mode)
    return err


def marginal_error(
    fu: torch.Tensor,
    gv: torch.Tensor,
    target: torch.Tensor,
    ot_prob: LinearProblem,
    dim: int = -2,
    norm: int = 2,
    lse_mode: bool = True
) -> torch.Tensor:
    if lse_mode:
        marginal = ot_prob.marginal_from_potentials(fu, gv, dim=dim)
    else:
        marginal = ot_prob.marginal_from_scalings(fu, gv, dim=dim)
    # distance between target and marginal
    return torch.norm(marginal - target, p=norm, dim=-1)


def compute_kl_reg_ot_cost(
    f: torch.Tensor, 
    g: torch.Tensor, 
    ot_prob: LinearProblem, 
    lse_mode: bool = True,
    use_danskin: bool = False
) -> torch.Tensor:
    f = f.detach() if use_danskin else f
    g = g.detach() if use_danskin else g

    supp_a = ot_prob.a > 0
    supp_b = ot_prob.b > 0
    fa = ot_prob.potential_from_scaling(ot_prob.a)
    if ot_prob.tau_a == 1.0:
        div_a = torch.sum(torch.where(supp_a, ot_prob.a * (f - fa), 0.0), dim=-1)
    else:
        rho_a = rho(ot_prob.epsilon, ot_prob.tau_a)
        div_a = -torch.sum(
            torch.where(supp_a, ot_prob.a * phi_star(-(f - fa), rho_a), 0.0),
            dim=-1,
        )

    gb = ot_prob.potential_from_scaling(ot_prob.b)
    if ot_prob.tau_b == 1.0:
        div_b = torch.sum(torch.where(supp_b, ot_prob.b * (g - gb), 0.0), dim=-1)
    else:
        rho_b = rho(ot_prob.epsilon, ot_prob.tau_b)
        div_b = -torch.sum(
            torch.where(supp_b, ot_prob.b * phi_star(-(g - gb), rho_b), 0.0),
            dim=-1,
        )

    # Using https://arxiv.org/pdf/1910.12958.pdf (24)
    if lse_mode:
        total_sum = torch.sum(ot_prob.marginal_from_potentials(f, g), dim=(-2, -1))
    else:
        total_sum = torch.sum(ot_prob.marginal_from_scalings(f, g), dim=(-2, -1))
    return div_a + div_b + ot_prob.epsilon * (
        torch.sum(ot_prob.a, dim=-1) * torch.sum(ot_prob.b, dim=-1) - total_sum
    )


def recenter(
        f: torch.Tensor,
        g: torch.Tensor,
        ot_prob: LinearProblem,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ot_prob.is_balanced:
            # center the potentials for numerical stability
            is_finite = torch.isfinite(f)
            shift = torch.sum(torch.where(is_finite, f, 0.0), dim=-1, keepdim=True) / torch.sum(is_finite, dim=-1, keepdim=True)
            return f - shift, g + shift

        if ot_prob.tau_a == 1.0 or ot_prob.tau_b == 1.0:
            # re-centering wasn't done during the lse-step, ignore
            return f, g

        rho_a = rho(ot_prob.epsilon, ot_prob.tau_a)
        rho_b = rho(ot_prob.epsilon, ot_prob.tau_b)
        tau = rho_a * rho_b / (rho_a + rho_b)

        shift = tau * (
            logsumexp(-f / rho_a, b=ot_prob.a, dim=-1) -
            logsumexp(-g / rho_b, b=ot_prob.b, dim=-1)
        )[..., None]
        return f + shift, g - shift


class SinkhornState():

    def __init__(
        self,
        errors: torch.Tensor = None,
        costs: torch.Tensor = None,
        fu: torch.Tensor = None,
        gv: torch.Tensor = None,
    ):
        self.errors = errors
        self.costs = costs
        self.fu = fu
        self.gv = gv
        self.converged_at = -1

    def solution_error(
        self,
        ot_prob: LinearProblem,
        parallel_dual_updates: bool,
        recenter_potentials: bool = True,
        lse_mode: bool = True,
    ) -> torch.Tensor:
        fu, gv = self.fu, self.gv
        if recenter_potentials and lse_mode:
            fu, gv = recenter(fu, gv, ot_prob)

        return solution_error(
            fu,
            gv,
            ot_prob,
            parallel_dual_updates=parallel_dual_updates,
            lse_mode=lse_mode
        )

    def kl_reg_ot_cost(
        self, ot_prob: LinearProblem, lse_mode: bool = True, use_danskin: bool = False
    ) -> float:
        return compute_kl_reg_ot_cost(self.fu, self.gv, ot_prob, lse_mode=lse_mode, use_danskin=use_danskin)


class SinkhornOutput():

    def __init__(
        self,
        fu: torch.Tensor,
        gv: torch.Tensor,
        costs: torch.Tensor,
        errors: torch.Tensor,
        ot_prob: LinearProblem,
        epsilon: float,
        inner_iterations: int,
        lse_mode: bool = True,
        use_danskin: bool = False
    ) -> None:
        self.f = fu
        self.g = gv
        self.costs = costs
        self.errors = errors
        self.ot_prob = ot_prob
        self.epsilon = epsilon
        self.inner_iterations = inner_iterations
        self.use_danskin = use_danskin
        self.kl_reg_ot_cost = compute_kl_reg_ot_cost(fu, gv, ot_prob, lse_mode=lse_mode, use_danskin=use_danskin)

    @property
    def dual_cost(self) -> torch.Tensor:
        a, b = self.ot_prob.a, self.ot_prob.b
        dual_cost = torch.sum(torch.where(a > 0.0, a * self.f, 0), dim=-1)
        dual_cost += torch.sum(torch.where(b > 0.0, b * self.g, 0), dim=-1)
        return dual_cost

    @property
    def kl_reg_cost(self) -> float:
        return self.kl_reg_ot_cost

    @property
    def ent_reg_cost(self) -> float:
        ent_a = torch.sum(torch.special.entr(self.ot_prob.a), dim=-1)
        ent_b = torch.sum(torch.special.entr(self.ot_prob.b), dim=-1)
        return self.kl_reg_ot_cost - self.epsilon * (ent_a + ent_b)

    @property
    def kl_reg_cost(self) -> float:
        return self.kl_reg_ot_cost

    @property
    def a(self) -> torch.Tensor:
        return self.ot_prob.a

    @property
    def b(self) -> torch.Tensor:
        return self.ot_prob.b

    @property
    def n_iters(self) -> torch.Tensor:
        return torch.sum(self.errors != -1, dim=-1) * self.inner_iterations

    @property
    def converged(self) -> torch.Tensor:
        return torch.any(self.costs == -1, dim=-1) and torch.all(torch.isfinite(self.costs), dim=-1)

    @property
    def scalings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.ot_prob.scaling_from_potential(self.f)
        v = self.ot_prob.scaling_from_potential(self.g)
        return u, v
    
    def marginal(self, dim: int) -> torch.Tensor:
        return self.ot_prob.marginal_from_potentials(self.f, self.g, dim=dim)

    @property
    def matrix(self) -> torch.Tensor:
        try:
            return self.ot_prob.transport_from_potentials(self.f, self.g)
        except ValueError:
            return self.ot_prob.transport_from_scalings(*self.scalings)

    @property
    def transport_mass(self) -> torch.Tensor:
        return self.marginal(-2).sum(dim=-1)


class Momentum:

    def __init__(
        self,
        start: int = 0,
        error_threshold: float = torch.inf,
        value: float = 1.0,
        inner_iterations: int = 1,
    ) -> None:
        self.start = start
        self.error_threshold = error_threshold
        self.value = value
        self.inner_iterations = inner_iterations

    def weight(self, state: SinkhornState, iteration: int) -> float:
        if self.start == 0:
            return self.value
        idx = self.start // self.inner_iterations

        return self.lehmann(state) if iteration >= self.start and state.errors[..., idx - 1] < self.error_threshold \
            else self.value

    def lehmann(self, state: SinkhornState) -> float:
        """See Lehmann, T., Von Renesse, M.-K., Sambale, A., and
            Uschmajew, A. (2021). A note on overrelaxation in the
            sinkhorn algorithm. Optimization Letters, pages 1–12. eq. 5."""
        idx = self.start // self.inner_iterations
        error_ratio = torch.minimum(
            state.errors[..., idx - 1] / state.errors[..., idx - 2], 0.99
        )
        power = 1.0 / self.inner_iterations
        return 2.0 / (1.0 + torch.sqrt(1.0 - error_ratio ** power))

    def __call__(  
        self,
        weight: float,
        value: torch.Tensor,
        new_value: torch.Tensor,
        lse_mode: bool = True,
    ) -> torch.Tensor:
        if lse_mode:
            value = torch.where(torch.isfinite(value), value, 0.0)
            return (1.0 - weight) * value + weight * new_value
        value = torch.where(value > 0.0, value, 1.0)
        return value ** (1.0 - weight) * new_value ** weight


class Sinkhorn:

    def __init__(
        self,
        lse_mode: bool = True,
        threshold: float = 1e-3,
        inner_iterations: int = 1,
        min_iterations: int = 1,
        max_iterations: int = 100,
        parallel_dual_updates: bool = False,
        recenter_potentials: bool = True,
        initializer: Literal["default", "random"] = "default",
        use_danskin: bool = True,
        **kwargs: Any,
    ):
        self.lse_mode = lse_mode
        self.threshold = threshold
        self.inner_iterations = inner_iterations
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.recenter_potentials = recenter_potentials
        self.momentum = Momentum(inner_iterations=inner_iterations)

        self.parallel_dual_updates = parallel_dual_updates
        self.use_danskin = use_danskin
        self.initializer = initializer

    def __call__(
        self,
        ot_prob: LinearProblem,
        init: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        compute_error: bool = True,
    ) -> torch.Tensor:

        initializer = self.create_initializer()
        init_dual_a, init_dual_b = initializer(
            ot_prob, *init
        )
        final_state = self.iterations(ot_prob, (init_dual_a, init_dual_b), compute_error=compute_error)
        return self.output_from_state(ot_prob, final_state), final_state

    def create_initializer(self) -> SinkhornInitializer:  
        if isinstance(self.initializer, SinkhornInitializer):
            return self.initializer
        if self.initializer == "default":
            return DefaultInitializer()
        if self.initializer == "random":
            return RandomInitializer()
        raise NotImplementedError(
            f"Initializer `{self.initializer}` is not yet implemented."
        )

    def lse_step(
        self, ot_prob: LinearProblem, state: SinkhornState,
        iteration: int
    ) -> SinkhornState:

        def k(tau_i: float, tau_j: float) -> float:
            num = -tau_j * (tau_a - 1) * (tau_b - 1) * (tau_i - 1)
            denom = (tau_j - 1) * (tau_a * (tau_b - 1) + tau_b * (tau_a - 1))
            return num / denom

        def xi(tau_i: float, tau_j: float) -> float:
            k_ij = k(tau_i, tau_j)
            return k_ij / (1.0 - k_ij)

        def smin(
            potential: torch.Tensor, marginal: torch.Tensor, tau: float
        ) -> torch.Tensor:
            r = rho(ot_prob.epsilon, tau)
            return -r * logsumexp(-potential / r, b=marginal, dim=-1)

        # only for an unbalanced problems with `tau_{a,b} < 1`
        recenter_potentials = (
            self.recenter_potentials and ot_prob.tau_a < 1.0 and ot_prob.tau_b < 1.0
        )
        w = self.momentum.weight(state, iteration)
        tau_a, tau_b = ot_prob.tau_a, ot_prob.tau_b
        old_fu, old_gv = state.fu, state.gv

        if recenter_potentials:
            k11, k22 = k(tau_a, tau_a), k(tau_b, tau_b)
            xi12, xi21 = xi(tau_a, tau_b), xi(tau_b, tau_a)

        # update g potential
        new_gv = tau_b * ot_prob.update_potential(
            old_fu, old_gv, safe_log(ot_prob.b), iteration, dim=-2
        )
        if recenter_potentials:
            new_gv -= k22 * smin(old_fu, ot_prob.a, tau_a)[..., None]
            new_gv += xi21 * smin(new_gv, ot_prob.b, tau_b)[..., None]
        gv = self.momentum(w, old_gv, new_gv, self.lse_mode)

        if not self.parallel_dual_updates:
            old_gv = gv

        # update f potential
        new_fu = tau_a * ot_prob.update_potential(
            old_fu, old_gv, safe_log(ot_prob.a), iteration, dim=-1
        )
        if recenter_potentials:
            new_fu -= k11 * smin(old_gv, ot_prob.b, tau_b)[..., None]
            new_fu += xi12 * smin(new_fu, ot_prob.a, tau_a)[..., None]
        fu = self.momentum(w, old_fu, new_fu, self.lse_mode)

        state.fu = fu
        state.gv = gv
        return state

    def kernel_step(
        self, ot_prob: LinearProblem, state: SinkhornState,
        iteration: int
    ) -> SinkhornState:
        w = self.momentum.weight(state, iteration)
        old_gv = state.gv
        new_gv = ot_prob.update_scaling(
            state.fu, ot_prob.b, iteration, dim=-1
        ) ** ot_prob.tau_b
        gv = self.momentum(w, state.gv, new_gv, self.lse_mode)
        new_fu = ot_prob.update_scaling(
            old_gv if self.parallel_dual_updates else gv,
            ot_prob.a,
            iteration,
            dim=-1
        ) ** ot_prob.tau_a
        fu = self.momentum(w, state.fu, new_fu, self.lse_mode)

        state.fu = fu
        state.gv = gv
        return state

    def one_iteration(
        self, ot_prob: LinearProblem, state: SinkhornState,
        iteration: int, compute_error: bool = True
    ) -> SinkhornState:

        if self.lse_mode:
            state = self.lse_step(ot_prob, state, iteration)
        else:
            state = self.kernel_step(ot_prob, state, iteration)

        # re-computes error if compute_error is True, else set it to -1.
        if iteration % self.inner_iterations == 0:
            it = iteration // self.inner_iterations
            if compute_error:
                err = state.solution_error(
                    ot_prob,
                    parallel_dual_updates=self.parallel_dual_updates,
                    recenter_potentials=self.recenter_potentials,
                    lse_mode=self.lse_mode,
                )
                cost = state.kl_reg_ot_cost(ot_prob, lse_mode=self.lse_mode, use_danskin=self.use_danskin)
            else:
                err = -1
                cost = -1
            state.errors[..., it] = err
            state.costs[..., it] = cost
        return state

    def _converged(self, state: SinkhornState, iteration: int) -> bool:
        if iteration < self.min_iterations:
            return False
        it = iteration // self.inner_iterations
        err = state.errors[..., it - 1]
        return (err < self.threshold).all()

    def _diverged(self, state: SinkhornState, iteration: int) -> bool:
        it = iteration // self.inner_iterations
        err = torch.isinf(state.errors[..., it - 1]).any() or torch.isnan(state.errors[..., it - 1]).any()
        cost = torch.isinf(state.costs[..., it - 1]).any() or torch.isnan(state.costs[..., it - 1]).any()
        return err or cost

    def _continue(self, state: SinkhornState, iteration: int) -> bool:
        return iteration < self.max_iterations and not self._converged(state, iteration) and not self._diverged(state, iteration)

    def init_state(
        self, init: Tuple[torch.Tensor, torch.Tensor]
    ) -> SinkhornState:
        fu, gv = init
        batch_dim = fu.shape[:-1]
        total_size = np.ceil(self.max_iterations / self.inner_iterations).astype(int)
        errors = -torch.ones(batch_dim + (total_size, )).type_as(fu)
        costs = -torch.ones(batch_dim + (total_size, )).type_as(fu)
        state = SinkhornState(errors=errors, costs=costs, fu=fu, gv=gv)
        return state

    def output_from_state(
        self, ot_prob: LinearProblem, state: SinkhornState
    ) -> torch.Tensor:
        f = state.fu if self.lse_mode else ot_prob.potential_from_scaling(state.fu)
        g = state.gv if self.lse_mode else ot_prob.potential_from_scaling(state.gv)
        if self.recenter_potentials:
            f, g = recenter(f, g, ot_prob)
        return SinkhornOutput(
            f,
            g,
            state.costs,
            state.errors,
            ot_prob,
            ot_prob.epsilon,
            self.inner_iterations,
            self.lse_mode,
            self.use_danskin
        )

    def iterations(
        self, ot_prob: LinearProblem, init: Tuple[torch.Tensor, torch.Tensor], compute_error: bool = True
    ) -> SinkhornState:
        state = self.init_state(init)
        iteration = 0
        while self._continue(state, iteration):
            state = self.one_iteration(ot_prob, state, iteration, compute_error=compute_error)
            iteration += 1
        if self._converged(state, iteration):
            state.converged_at = iteration 
        return state


if __name__ == "__main__":
    from torch_sinkhorn.problem import Epsilon
    from torch_sinkhorn.utils import plot_coupling
    from torch_robotics.torch_utils.torch_timer import TimerCUDA
    # epsilon = Epsilon(target=0.05, init=1., decay=0.8)
    # x = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.3, 0.2]])
    # y = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.3, 0.2]])
    # C = torch.cdist(x, y, p=2)
    batch = 32
    C = torch.rand(batch, 100, 100)
    ot_prob = LinearProblem(
        C, epsilon=0.05,
        # tau_a=0.5, tau_b=0.5
    )
    sinkhorn = Sinkhorn(lse_mode=True, min_iterations=50, max_iterations=50, parallel_dual_updates=False, recenter_potentials=True)
    with TimerCUDA() as t:
        W, state = sinkhorn(ot_prob)
    print(t.elapsed)
    print(f"Converged at {state.converged_at}")
    import matplotlib.pyplot as plt
    plt.figure()
    mean_errors = torch.mean(state.errors, dim=0)
    var_errors = torch.var(state.errors, dim=0)
    plt.errorbar(
        torch.arange(mean_errors.shape[-1]),
        mean_errors,
        yerr=var_errors,
        label="error"
    )
    plt.figure()
    mean_costs = torch.mean(state.costs, dim=0)
    var_costs = torch.var(state.costs, dim=0)
    plt.errorbar(
        torch.arange(mean_costs.shape[-1]),
        mean_costs,
        yerr=var_costs,
        label="cost"
    )
    plt.figure()
    plot_coupling(W.matrix[0])
    plt.show()
