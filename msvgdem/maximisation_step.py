from jaxtyping import Array, Float
from simple_pytree import Pytree, static_field
from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Tuple
from coinem.model import AbstractModel
from coinem.dataset import Dataset
from coinem.gradient_transforms import GradientTransformation, OptimiserState

import jax.tree_util as jtu
import jax.numpy as jnp

"""State for the maximisation step. """
AbstractMaximisationState = NamedTuple


@dataclass
class AbstractMaximisationStep(Pytree):
    """The M-step of the EM algorithm."""

    model: AbstractModel = static_field()

    def init(self, theta, key) -> AbstractMaximisationState:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        state: AbstractMaximisationState,
        latent: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Tuple[Float[Array, "Q"], AbstractMaximisationState]:
        raise NotImplementedError


class GradientMaximisationState(AbstractMaximisationState):
    optimiser_state: OptimiserState
    prev_theta: Float[Array, "Q"] = None


@dataclass
class MaximisationStep(AbstractMaximisationStep):
    """The M-step of the EM algorithm."""

    optimiser: GradientTransformation = static_field()

    def init(self, params, key) -> GradientMaximisationState:
        return GradientMaximisationState(optimiser_state=self.optimiser.init(params), prev_theta=params)

    def update(
        self,
        maximisation_state: GradientMaximisationState,
        latent: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Tuple[Float[Array, "Q"], GradientMaximisationState]:
        # Unpack maximisation state
        theta_opt_state = maximisation_state.optimiser_state

        # Find negative average score of theta, since we are maximising, but optimisers minimise.
        average_score_theta = self.model.average_score_theta(latent, theta, data)

        negative_average_score_theta = jtu.tree_map(lambda x: -x, average_score_theta)

        # Find update rule for theta
        theta_updates, theta_new_opt_state = self.optimiser.update(
            negative_average_score_theta, theta_opt_state, theta
        )

        # Apply updates to theta
        theta_new = jtu.tree_map(lambda p, u: p + u, theta, theta_updates)

        # Update maximisation state
        maximisation_state_new = GradientMaximisationState(
            optimiser_state=theta_new_opt_state,
            prev_theta=theta
        )

        return theta_new, maximisation_state_new


@dataclass
class AcceleratedMaximisationStep(MaximisationStep):
    """The M-step of the EM algorithm with acceleration on theta updates."""

    acceleration: float = static_field(default=0.9)  # The c1(1-c2) term as a single parameter
    use_log_space: bool = static_field(default=False)  # Flag to enable log-space transformation for stability
    
    def update(
        self,
        maximisation_state: GradientMaximisationState,
        latent: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Tuple[Float[Array, "Q"], GradientMaximisationState]:
        # First perform the regular M-step update
        theta_new, maximisation_state_new = super().update(
            maximisation_state, latent, theta, data
        )
        
        # Get the previous theta from state
        prev_theta = maximisation_state.prev_theta
        
        # If prev_theta is None (first iteration), just use current theta
        if prev_theta is None:
            prev_theta = theta
        
        if self.use_log_space:
            # Perform acceleration in log space for numerical stability
            
            # Convert to log space (safely)
            log_theta_new = jtu.tree_map(lambda x: jnp.log(jnp.maximum(x, 1e-10)), theta_new)
            log_prev_theta = jtu.tree_map(lambda x: jnp.log(jnp.maximum(x, 1e-10)), prev_theta)
            
            # Compute difference in log space
            log_diff = jtu.tree_map(lambda new, old: new - old, log_theta_new, log_prev_theta)
            
            # Apply acceleration in log space
            log_accelerated = jtu.tree_map(
                lambda log_new, log_diff: log_new + self.acceleration * log_diff,
                log_theta_new,
                log_diff
            )
            
            # Convert back from log space
            theta_accelerated = jtu.tree_map(jnp.exp, log_accelerated)
        else:
            # Then apply acceleration: theta_tilde = theta_{t+1} + acceleration * (theta_{t+1} - theta_t)
            # We use theta_new - prev_theta instead of theta_new - theta for true Nesterov acceleration
            theta_diff = jtu.tree_map(lambda new, old: new - old, theta_new, prev_theta)

            theta_accelerated = jtu.tree_map(
                lambda new, diff: new + self.acceleration * diff, 
                theta_new, 
                theta_diff
            )
        
        return theta_accelerated, maximisation_state_new


class MarginalMaximisationState(AbstractMaximisationState):
    pass


@dataclass
class MarginalStep(AbstractMaximisationStep):
    """The M-step of the EM algorithm, when the optimal theta is a function of the latent particles only."""

    def init(self, params, key) -> MarginalMaximisationState:
        return MarginalMaximisationState()

    def update(
        self,
        maximisation_state: MarginalMaximisationState,
        latent: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Tuple[Float[Array, "Q"], MarginalMaximisationState]:
        # Find update rule for theta as a function of the particle cloud only
        theta_new = self.model.optimal_theta(latent)

        return theta_new, maximisation_state


class MomentumMaximisationState(AbstractMaximisationState):
    optimiser_state: OptimiserState
    momentum: Float[Array, "Q"]


@dataclass
class MomentumMaximisationStep(MaximisationStep):
    """The M-step of the EM algorithm - exact translation from authors' code."""

    step_size: float = static_field(default=1e-2)     # Time step size (dt)
    gamma_theta: float = static_field(default=0.9)    # Friction coefficient for parameters
    eta_theta: float = static_field(default=200.0)    # Mass/temperature ratio for parameters

    def init(self, params, key) -> MomentumMaximisationState:
        # Initialize momentum to zero (cold start for parameters)
        momentum = jtu.tree_map(lambda x: jnp.zeros_like(x), params)
        return MomentumMaximisationState(
            optimiser_state=self.optimiser.init(params), 
            momentum=momentum
        )

    def update(
        self,
        maximisation_state: MomentumMaximisationState,
        latent: Float[Array, "N D"],
        theta: Float[Array, "Q"],
        data: Dataset,
    ) -> Tuple[Float[Array, "Q"], MomentumMaximisationState]:
        # Unpack maximisation state
        theta_opt_state = maximisation_state.optimiser_state
        momentum = maximisation_state.momentum

        # Time step (use the step_size parameter)
        dt = self.step_size
        
        # Compute exponential decay factor (direct from authors' code)
        tgameta = self.gamma_theta * self.eta_theta
        tomega = jnp.exp(-tgameta * dt)
        omtomega = 1 - tomega
        
        # Compute c1 coefficient
        c1 = omtomega / self.gamma_theta
        
        # Apply Nesterov-style lookahead (direct from authors' code)
        # param.data = param.data + c1 * tmo
        theta_lookahead = jtu.tree_map(
            lambda t, m: t + c1 * m,
            theta,
            momentum
        )

        # Compute gradient at lookahead position
        average_score_theta = self.model.average_score_theta(latent, theta_lookahead, data)
        negative_average_score_theta = jtu.tree_map(lambda x: -x, average_score_theta)

        # Compute c2 coefficient
        c2 = (dt - omtomega / tgameta) / self.gamma_theta # This is the same as the c2 coefficient i tried to find for msvgdem
        
        # Update momentum (direct from authors' code)
        # tmo_next = tomega * tmo - omtomega * param.grad.data / tgameta
        momentum_new = jtu.tree_map(
            lambda m, g: tomega * m - (omtomega / tgameta) * g,
            momentum,
            negative_average_score_theta
        )

        # Update parameters (direct from authors' code)
        # theta_next = theta - c2 * param.grad.data
        theta_new = jtu.tree_map(
            lambda t, g: t - c2 * g,
            theta,
            negative_average_score_theta
        )

        # Update maximisation state
        maximisation_state_new = MomentumMaximisationState(
            optimiser_state=theta_opt_state,  # Keep the same optimiser state
            momentum=momentum_new
        )

        return theta_new, maximisation_state_new
