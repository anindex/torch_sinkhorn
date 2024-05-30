from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch

from torch_sinkhorn.initializer_lr import LRInitializer, RandomInitializer, Rank2Initializer
from torch_sinkhorn.problem import LinearProblem
from torch_sinkhorn.sinkhorn import Sinkhorn
from torch_sinkhorn.utils import safe_log, gen_js, gen_kl


class LRSinkhornState():

    def __init__(
            self,
            errors: torch.Tensor = None,
            costs: torch.Tensor = None,
            q: torch.Tensor = None,
            r: torch.Tensor = None,
            g: torch.Tensor = None,
            gamma: float = 10.0,
        ):
            self.errors = errors
            self.costs = costs
            self.q = q
            self.r = r
            self.g = g
            self.gamma = gamma
            self.crossed_threshold = False
            self.converged_at = -1

    def compute_error(
        self, previous_state: "LRSinkhornState"
    ) -> torch.Tensor:
        err_q = gen_js(self.q, previous_state.q, c=1.0)
        err_r = gen_js(self.r, previous_state.r, c=1.0)
        err_g = gen_js(self.g, previous_state.g, c=1.0)

        return ((1.0 / self.gamma) ** 2) * (err_q + err_r + err_g)

    def ent_reg_cost(
        self,
        ot_prob: LinearProblem,
        *,
        epsilon: float,
        use_danskin: bool = False
    ) -> torch.Tensor:
        return compute_ent_reg_ot_cost(
            self.q,
            self.r,
            self.g,
            ot_prob,
            epsilon=epsilon,
            use_danskin=use_danskin
        )

    def solution_error(
        self, ot_prob: LinearProblem, norm_error: Tuple[int, ...]
    ) -> torch.Tensor:
        return solution_error(self.q, self.r, ot_prob, norm_error)


def compute_ent_reg_ot_cost(
    q: torch.Tensor,
    r: torch.Tensor,
    g: torch.Tensor,
    ot_prob: LinearProblem,
    epsilon: float,
    use_danskin: bool = False
) -> torch.Tensor:

    def ent(x: torch.Tensor) -> float:
        # generalized entropy
        return torch.sum(torch.special.entr(x) + x, dim=(-2, -1))

    tau_a, tau_b = ot_prob.tau_a, ot_prob.tau_b

    q = q.detach() if use_danskin else q
    r = r.detach() if use_danskin else r
    g = g.detach() if use_danskin else g

    C_r = torch.einsum("...ij,...jk->...ik", ot_prob.C, r)
    cost = torch.sum(C_r * q * (1.0 / g)[..., None, :], dim=(-2, -1))
    cost -= epsilon * (ent(q) + ent(r) + ent(g))
    if tau_a != 1.0:
        cost += tau_a / (1.0 - tau_a) * gen_kl(torch.sum(q, dim=-1), ot_prob.a)
    if tau_b != 1.0:
        cost += tau_b / (1.0 - tau_b) * gen_kl(torch.sum(r, dim=-1), ot_prob.b)

    return cost


def solution_error(
    q: torch.Tensor, r: torch.Tensor, ot_prob: LinearProblem,
    norm_error: int = 2
) -> torch.Tensor:
    err = torch.norm(torch.sum(q, dim=-1) - ot_prob.a, p=norm_error, dim=-1)
    err += torch.norm(torch.sum(r, dim=-1) - ot_prob.b, p=norm_error, dim=-1)
    err += torch.norm(torch.sum(q, dim=-2) - torch.sum(r, dim=-2), p=norm_error, dim=-1)
    return err


class LRSinkhornOutput():

    def __init__(
            self,
            q: torch.Tensor,
            r: torch.Tensor,
            g: torch.Tensor,
            costs: torch.Tensor,
            errors: torch.Tensor,
            ot_prob: LinearProblem,
            epsilon: float,
            inner_iterations: int,
            use_danskin: bool = False
        ):
            self.q = q
            self.r = r
            self.g = g
            self.costs = costs
            self.errors = errors
            self.ot_prob = ot_prob
            self.epsilon = epsilon
            self.inner_iterations = inner_iterations
            self.ent_reg_ot_cost = compute_ent_reg_ot_cost(q, r, g, ot_prob, epsilon, use_danskin)

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
    def matrix(self) -> torch.Tensor:
        return torch.einsum("...ij,...jk->...ik", self.q * self._inv_g[..., None, :], self.r.mT)

    def apply(self, inputs: torch.Tensor, dim: int = -2) -> torch.Tensor:
        q, r = (self.q, self.r) if dim == -1 else (self.r, self.q)
        i_r = torch.einsum("...ij,...jk->...ik", inputs, r)
        return torch.einsum("...ij,...jk->...ik", i_r * self._inv_g, q.mT)

    def marginal(self, dim: int) -> torch.Tensor:
        batch_dim = self.q.shape[:-2]
        length = self.q.shape[-2] if dim == -2 else self.r.shape[-2]
        return self.apply(torch.ones(batch_dim + (length,)).type_as(self.q), dim=dim)

    @property
    def ent_reg_cost(self) -> float:
        return self.ent_reg_ot_cost

    @property
    def transport_mass(self) -> float:
        return self.marginal(-2).sum(dim=-1)

    @property
    def _inv_g(self) -> torch.Tensor:
        return 1.0 / self.g


class LRSinkhorn(Sinkhorn):

    def __init__(
        self,
        rank: int,
        lse_mode: bool = True,
        gamma: float = 10.0,
        gamma_rescale: bool = True,
        epsilon: float = 0.0,
        threshold: float = 1e-3,
        min_iterations: int = 1,
        inner_iterations: int = 1,
        max_iterations: int = 50,
        kwargs_dys: Optional[Mapping[str, Any]] = None,
        kwargs_init: Optional[Mapping[str, Any]] = None,
        initializer: Union[Literal["random", "rank2"],
                            LRInitializer] = "random",
        use_danskin: bool = False,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold,
                         inner_iterations=inner_iterations,
                         min_iterations=max(2 * inner_iterations, min_iterations),
                         max_iterations=max_iterations,
                         **kwargs)
        self.lse_mode = lse_mode
        self.rank = rank
        self.gamma = gamma
        self.gamma_rescale = gamma_rescale
        self.epsilon = epsilon
        self.initializer = initializer
        self.use_danskin = use_danskin
        self.kwargs_dys = {} if kwargs_dys is None else kwargs_dys
        self.kwargs_init = {} if kwargs_init is None else kwargs_init

    def __call__(
        self,
        ot_prob: LinearProblem,
        init: Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
                    Optional[torch.Tensor]] = (None, None, None),
        **kwargs: Any,
    ) -> LRSinkhornOutput:
        initializer = self.create_initializer()
        init = initializer(ot_prob, *init, **kwargs)
        final_state = self.iterations(ot_prob, init, **kwargs)
        return self.output_from_state(ot_prob, final_state), final_state

    def create_initializer(self) -> LRInitializer:
        if isinstance(self.initializer, LRInitializer):
            assert self.initializer.rank == self.rank, \
                f"Expected initializer's rank to be `{self.rank}`," \
                f"found `{self.initializer.rank}`."
            return self.initializer
        if self.initializer == "random":
            return RandomInitializer(rank=self.rank)
        if self.initializer == "rank2":
            return Rank2Initializer(rank=self.rank)
        raise NotImplementedError(
            f"Initializer `{self.initializer}` is not yet implemented."
        )

    def _get_costs(
        self,
        ot_prob: LinearProblem,
        state: LRSinkhornState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        log_q, log_r, log_g = (
            safe_log(state.q), safe_log(state.r), safe_log(state.g)
        )

        inv_g = 1.0 / state.g[..., None, :]
        tmp = torch.einsum("...ij,...jk->...ik", ot_prob.C, state.r)

        grad_q = tmp * inv_g
        C_q = torch.einsum("...ij,...jk->...ik", ot_prob.C.mT, state.q)
        grad_r = C_q * inv_g
        grad_g = -torch.sum(state.q * tmp, dim=-2) / (state.g ** 2)

        grad_q += self.epsilon * log_q
        grad_r += self.epsilon * log_r
        grad_g += self.epsilon * log_g

        if self.gamma_rescale:
            norm_q = torch.max(torch.abs(grad_q)) ** 2
            norm_r = torch.max(torch.abs(grad_r)) ** 2
            norm_g = torch.max(torch.abs(grad_g)) ** 2
            gamma = self.gamma / torch.max(torch.tensor([norm_q, norm_r, norm_g]))
        else:
            gamma = self.gamma

        eps_factor = 1.0 / (self.epsilon * gamma + 1.0)
        gamma *= eps_factor

        c_q = -gamma * grad_q + eps_factor * log_q
        c_r = -gamma * grad_r + eps_factor * log_r
        c_g = -gamma * grad_g + eps_factor * log_g

        return c_q, c_r, c_g, gamma

    def dykstra_update_lse(
        self,
        c_q: torch.Tensor,
        c_r: torch.Tensor,
        h: torch.Tensor,
        gamma: float,
        ot_prob: LinearProblem,
        min_entry_value: float = 1e-6,
        tolerance: float = 5e-2,
        min_iter: int = 0,
        max_iter: int = 200
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.rank
        n, m = ot_prob.C.shape[-2:]
        batch_dim = c_q.shape[:-2]
        loga, logb = torch.log(ot_prob.a), torch.log(ot_prob.b)
        err = torch.tensor(torch.inf)
        h_old = h
        g1_old, g2_old = torch.zeros(batch_dim + (r,)).type_as(loga), torch.zeros(batch_dim + (r,)).type_as(loga)
        f1, f2 = torch.zeros(batch_dim + (n,)).type_as(loga), torch.zeros(batch_dim + (m,)).type_as(loga)
        w_gi, w_gp = torch.zeros_like(g1_old), torch.zeros_like(g1_old)
        w_q, w_r = torch.zeros_like(g1_old), torch.zeros_like(g1_old)

        def _softm(f: torch.Tensor, g: torch.Tensor, c: torch.Tensor, dim: int) -> torch.Tensor:
            return torch.logsumexp(gamma * (f[..., :, None] + g[..., None, :] - c), dim=dim)

        i = 0
        while i < max_iter:
            if i >= min_iter and (err < tolerance).all():
                break
            # First Projection
            f1 = torch.where(
                torch.isfinite(loga),
                (loga - _softm(f1, g1_old, c_q, dim=-1)) / gamma + f1, loga
            )
            f2 = torch.where(
                torch.isfinite(logb),
                (logb - _softm(f2, g2_old, c_r, dim=-1)) / gamma + f2, logb
            )

            h = h_old + w_gi
            h = torch.clip(h, max=np.log(min_entry_value) / gamma)
            w_gi += h_old - h
            h_old = h
            # Update couplings
            g_q = _softm(f1, g1_old, c_q, dim=-2)
            g_r = _softm(f2, g2_old, c_r, dim=-2)
            # Second Projection
            h = (1.0 / 3.0) * (h_old + w_gp + w_q + w_r)
            h += g_q / (3.0 * gamma)
            h += g_r / (3.0 * gamma)
            g1 = h + g1_old - g_q / gamma
            g2 = h + g2_old - g_r / gamma

            w_q = w_q + g1_old - g1
            w_r = w_r + g2_old - g2
            w_gp = h_old + w_gp - h

            q = torch.exp(gamma * (f1[..., :, None] + g1[..., None, :] - c_q))
            r = torch.exp(gamma * (f2[..., :, None] + g2[..., None, :] - c_r))

            g1_old = g1
            g2_old = g2
            h_old = h

            err = solution_error(q, r, ot_prob)
            i += 1
        g = torch.exp(gamma * h)
        return q, r, g

    def dykstra_update_kernel(
        self,
        k_q: torch.Tensor,
        k_r: torch.Tensor,
        k_g: torch.Tensor,
        gamma: float,
        ot_prob: LinearProblem,
        min_entry_value: float = 1e-6,
        tolerance: float = 5e-2,
        min_iter: int = 0,
        max_iter: int = 200
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.rank
        n, m = ot_prob.C.shape[-2:]
        batch_dim = ot_prob.C.shape[:-2]
        a, b = ot_prob.a, ot_prob.b
        supp_a, supp_b = a > 0, b > 0
        err = torch.tensor(torch.inf)

        g_old = k_g
        v1_old, v2_old = torch.ones(batch_dim + (r,)).type_as(a), torch.ones(batch_dim + (r,)).type_as(a)
        u1, u2 = torch.ones(batch_dim + (n,)).type_as(a), torch.ones(batch_dim + (m,)).type_as(a)

        q_gi, q_gp = torch.ones_like(v1_old), torch.ones_like(v2_old)
        q_q, q_r = torch.ones_like(q_gi), torch.ones_like(q_gp)

        i = 0
        while i < max_iter:
            if i >= min_iter and (err < tolerance).all():
                break
            # First Projection
            kq_v = torch.einsum("...ij,...j->...i", k_q, v1_old)
            kr_v = torch.einsum("...ij,...j->...i", k_r, v2_old)
            u1 = torch.where(supp_a, a / kq_v, 0.0)
            u2 = torch.where(supp_b, b / kr_v, 0.0)
            g = torch.clip(g_old * q_gi, min=min_entry_value)
            q_gi = (g_old * q_gi) / g
            g_old = g
            # Second Projection
            v1_trans = torch.einsum("...ij,...j->...i", k_q.mT, u1)
            v2_trans = torch.einsum("...ij,...j->...i", k_r.mT, u2)
            g = (g_old * q_gp * v1_old * q_q * v1_trans * v2_old * q_r *
                v2_trans) ** (1 / 3)
            v1 = g / v1_trans
            v2 = g / v2_trans
            q_gp = (g_old * q_gp) / g
            q_q = (q_q * v1_old) / v1
            q_r = (q_r * v2_old) / v2
            v1_old = v1
            v2_old = v2
            g_old = g

            q = u1[..., :, None] * k_q * v1[..., None, :]
            r = u2[..., :, None] * k_r * v2[..., None, :]
            err = solution_error(q, r, ot_prob)
            i += 1
        return q, r, g

    def lse_step(
        self, ot_prob: LinearProblem, state: LRSinkhornState,
        iteration: int
    ) -> LRSinkhornState:
        """LR Sinkhorn LSE update."""
        # TODO: implement unbalanced case.
        c_q, c_r, c_g, gamma = self._get_costs(ot_prob, state)
        c_q, c_r, h = c_q / -gamma, c_r / -gamma, c_g / gamma
        q, r, g = self.dykstra_update_lse(
            c_q, c_r, h, gamma, ot_prob, **self.kwargs_dys
        )
        state.q = q
        state.r = r
        state.g = g
        state.gamma = gamma
        return state

    def kernel_step(
        self, ot_prob: LinearProblem, state: LRSinkhornState,
        iteration: int
    ) -> LRSinkhornState:
        """LR Sinkhorn Kernel update."""
        # TODO: implement unbalanced case.
        c_q, c_r, c_g, gamma = self._get_costs(ot_prob, state)
        c_q, c_r, c_g = torch.exp(c_q), torch.exp(c_r), torch.exp(c_g)
        q, r, g = self.dykstra_update_kernel(
            c_q, c_r, c_g, gamma, ot_prob, **self.kwargs_dys
        )
        state.q = q
        state.r = r
        state.g = g
        state.gamma = gamma
        return state

    def one_iteration(
        self, ot_prob: LinearProblem, state: LRSinkhornState,
        iteration: int, compute_error: bool = True
    ) -> LRSinkhornState:
        if self.lse_mode:  # In lse_mode, run additive updates.
            state = self.lse_step(ot_prob, state, iteration)
        else:
            state = self.kernel_step(ot_prob, state, iteration)

        # re-computes error if compute_error is True, else set it to inf.
        if iteration % self.inner_iterations == 0:
            it = iteration // self.inner_iterations
            if compute_error:
                cost = state.ent_reg_cost(ot_prob, epsilon=self.epsilon, use_danskin=self.use_danskin)
                error = state.compute_error(state)
            else:
                cost = -1
                error = -1
            state.crossed_threshold = state.crossed_threshold or ((state.errors[..., it - 1] >= self.threshold).all() and (error < self.threshold).all())
            state.costs[..., it] = cost
            state.errors[..., it] = error
        return state

    def iterations(
        self, ot_prob: LinearProblem, init: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], compute_error: bool = True
    ) -> LRSinkhornState:
        state = self.init_state(init)
        iteration = 0
        while self._continue(state, iteration):
            state = self.one_iteration(ot_prob, state, iteration, compute_error=compute_error)
            iteration += 1
        if self._converged(state, iteration):
            state.converged_at = iteration 
        return state

    def init_state(
        self, init: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> LRSinkhornState:
        q, r, g = init
        total_size = np.ceil(self.max_iterations / self.inner_iterations).astype(int)
        batch_dim = q.shape[:-2]
        return LRSinkhornState(
            q=q,
            r=r,
            g=g,
            gamma=self.gamma,
            costs=-torch.ones(batch_dim + (total_size,)).type_as(q),
            errors=-torch.ones(batch_dim + (total_size,)).type_as(q)
        )

    def output_from_state(
        self, ot_prob: LinearProblem, state: LRSinkhornState
    ) -> LRSinkhornOutput:
        return LRSinkhornOutput(
            q=state.q,
            r=state.r,
            g=state.g,
            ot_prob=ot_prob,
            costs=state.costs,
            errors=state.errors,
            epsilon=self.epsilon,
            inner_iterations=self.inner_iterations,
        )

    def _converged(self, state: LRSinkhornState, iteration: int) -> bool:
        if iteration < self.min_iterations:
            return False
        crossed = state.crossed_threshold
        it = iteration // self.inner_iterations
        prev_error, curr_error = state.errors[..., it - 2], state.errors[..., it - 1]
        if crossed:
            if (prev_error < self.threshold).all() and (curr_error < self.threshold).all():
                return True
        else:
            if (prev_error < self.threshold).all() and (curr_error < prev_error).all():
                return True
        return False

    def _diverged(self, state: LRSinkhornState, iteration: int) -> bool:
        it = iteration // self.inner_iterations
        err = torch.isinf(state.errors[..., it - 1]).any() or torch.isnan(state.errors[..., it - 1]).any()
        cost = torch.isinf(state.costs[..., it - 1]).any() or torch.isnan(state.costs[..., it - 1]).any()
        return err or cost

    def _continue(self, state: LRSinkhornState, iteration: int) -> bool:
        """Continue while not(converged) and not(diverged)."""
        return iteration < self.max_iterations and not self._converged(state, iteration) and not self._diverged(state, iteration)


if __name__ == "__main__":
    from torch_sinkhorn.problem import Epsilon
    from torch_robotics.torch_utils.torch_timer import TimerCUDA
    batch = 32
    epsilon = Epsilon(target=0.05, init=1., decay=0.8)
    ot_prob = LinearProblem(
        torch.rand((batch, 100, 100)), epsilon
    )
    sinkhorn = LRSinkhorn(rank=2, min_iterations=50, max_iterations=50, lse_mode=False, inner_iterations=1)
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
    plt.imshow(W.matrix.mean(0).cpu().numpy())
    plt.show()
