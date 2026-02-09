from __future__ import annotations

__all__ = ["cocob", "CocobState", "GradientTransformation", "OptimiserState"]

import jax.tree_util as jtu
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from typing import NamedTuple, Callable, Tuple

"""A stateful gradient transformation."""
OptimiserState = NamedTuple


class GradientTransformation(NamedTuple):
    """A stateful gradient transformation.

    Attributes:
      init (Callable[[PyTree], OptimiserState]): A function that initializes the state of the transformation.
      update (Callable[[PyTree, OptimiserState, PyTree], PyTree]): A function that applies the transformation to the gradient updates.
    """

    init: Callable[[PyTree], OptimiserState]
    update: Callable[[PyTree, OptimiserState, PyTree], PyTree]


class CocobState(OptimiserState):
    """State for the adaptive coin algorithm.

    Attributes:
      init_params (PyTree): The initial parameters.
      grad_sum (PyTree): The sum of the gradients.
      abs_grad_sum (PyTree): The sum of the absolute gradients.
      reward (PyTree): The reward.
      Lipschitz (PyTree): The Lipschitz constant.
    """

    init_params: PyTree
    grad_sum: PyTree
    abs_grad_sum: PyTree
    reward: PyTree
    Lipschitz: PyTree


def cocob(alpha: float = 0.0, eps: float = 1e-8) -> GradientTransformation:
    """
    The COntinuos COin Betting (COCOB) optimizer with adaptive learning of the Lipschitz constant.

    This is designed to work with any parameters/gradients comprising a PyTree of JAX arrays.

    Args:
      alpha (float): Smoothing parameter for the updates. Defaults to 0.0 <-> No smoothing.
      eps: A small constant applied to denominator outside of the square root to avoid dividing by zero when rescaling.

    Returns:
      A `GradientTransformation` object.
    """

    def init_fn(params: PyTree) -> CocobState:
        """Initializes the state for AdaCoin.

        Args:
          params(PyTree): a tree of parameters.

        Returns:
          CocobState: An initial state.
        """

        zeros = jtu.tree_map(jnp.zeros_like, params)
        jitter = jtu.tree_map(lambda x: eps * jnp.ones_like(x), params)
        return CocobState(params, zeros, zeros, zeros, jitter)

    def update_fn(
        gradient: PyTree, state: CocobState, params: PyTree
    ) -> Tuple[PyTree, CocobState]:
        """Applies AdaCoin to the gradient updates.

        Args:
          gradient (PyTree): a tree of gradient updates.
          state (AdaCoin): the current state.
          params (PyTree): a tree of parameters.

        Returns:
          Tuple[PyTree, CocobState]: A tuple of the updated gradient and the updated state.
        """

        params0 = state.init_params
        grad_sum = state.grad_sum
        abs_grad_sum = state.abs_grad_sum
        rt_minus_1 = state.reward
        lt_minus_1 = state.Lipschitz

        # Gradient
        ct = jtu.tree_map(lambda x: -x, gradient)  #  Minimisation convention

        # Absolute |ct|
        abs_ct = jtu.tree_map(jnp.abs, ct)

        # sum of gradients
        grad_sum = jtu.tree_map(jnp.add, grad_sum, ct)
        abs_grad_sum = jtu.tree_map(jnp.add, abs_grad_sum, abs_ct)

        # Maximum observed scalem Lt
        lt = jtu.tree_map(
            lambda ct_i, lt_m1_i: jnp.maximum(ct_i, lt_m1_i), abs_ct, lt_minus_1
        )

        # Reward
        rt = jtu.tree_map(
            lambda rt_mt_i, params_i, ct_i, params0_i: jnp.maximum(
                rt_mt_i + jnp.multiply(params_i - params0_i, ct_i), 0
            ),
            rt_minus_1,
            params,
            ct,
            params0,
        )

        # Param updates
        updates = jtu.tree_map(
            lambda params0_i, grad_sum_i, abs_grad_sum_i, rt_i, lt_i: params0_i
            + grad_sum_i
            / (lt_i * (jnp.maximum(abs_grad_sum_i + lt_i, alpha * lt_i)))
            * (lt_i + rt_i),
            params0,
            grad_sum,
            abs_grad_sum,
            rt,
            lt,
        )

        # Update state
        new_state = CocobState(params0, grad_sum, abs_grad_sum, rt, lt)

        # Difference between old and new params
        updates_differenced = jtu.tree_map(lambda p, u: u - p, params, updates)

        return updates_differenced, new_state

    return GradientTransformation(init_fn, update_fn)
