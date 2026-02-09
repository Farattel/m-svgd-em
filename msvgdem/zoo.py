import jax.random as jr

# Compatibility for older JAX versions: define KeyArray if it doesn't exist
try:
    from jax.random import KeyArray
except ImportError:
    from typing import Any
    KeyArray = Any  # Use Any as a fallback type

from jaxtyping import Array, Float
from typing import Tuple, Callable, Dict

from coinem.expectation_maximisation import expectation_maximisation
from coinem.model import AbstractModel
from coinem.maximisation_step import MaximisationStep, AcceleratedMaximisationStep, MomentumMaximisationStep
from coinem.expectation_step import (
    SteinExpectationStep,
    SoulExpectationStep,
    ParticleGradientExpectationStep,
    AcceleratedSteinExpectationStep,
    MomentumParticleGradientExpectationStep,
)
from coinem.dataset import Dataset
from coinem.gradient_transforms import cocob, GradientTransformation

import optax as ox


def svgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    latent_optimiser: GradientTransformation,
    theta_optimiser: GradientTransformation,
    num_steps: int,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """
    Perform the Stein variational gradient descent EM algorithm.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        latent_optimiser (GradientTransformation): The latent optimiser.
        theta_optimiser (GradientTransformation): The parameter optimiser.
        num_steps (int): The number of steps to perform, K.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to {}.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    return expectation_maximisation(
        expectation_step=SteinExpectationStep(model=model, optimiser=latent_optimiser),
        maximisation_step=MaximisationStep(model=model, optimiser=theta_optimiser),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )


def coin_svgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    alpha: float = 0.0,
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """Perform the CoinEM algorithm.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    optimiser = cocob(alpha=alpha)

    return svgd(
        model=model,
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        latent_optimiser=optimiser,
        theta_optimiser=optimiser,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )


def adam_svgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """Perform the Adam SVGD algorithm.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to None.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    latent_optimiser = ox.adam(latent_step_size)
    theta_optimiser = ox.adam(theta_step_size)

    return svgd(
        model=model,
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        latent_optimiser=latent_optimiser,
        theta_optimiser=theta_optimiser,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )


def ada_svgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """Perform the Adam SVGD algorithm.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to None.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    latent_optimiser = ox.adagrad(latent_step_size)
    theta_optimiser = ox.adagrad(theta_step_size)

    return svgd(
        model=model,
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        latent_optimiser=latent_optimiser,
        theta_optimiser=theta_optimiser,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )


def rms_svgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """Perform the RMSprop SVGD algorithm.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to None.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    latent_optimiser = ox.rmsprop(latent_step_size)
    theta_optimiser = ox.rmsprop(theta_step_size)

    return svgd(
        model=model,
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        latent_optimiser=latent_optimiser,
        theta_optimiser=theta_optimiser,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )


def soul(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
):
    """Perform the SoulEM algorithm.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to None.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    return expectation_maximisation(
        expectation_step=SoulExpectationStep(model=model, step_size=latent_step_size),
        maximisation_step=MaximisationStep(
            model=model, optimiser=ox.sgd(theta_step_size)
        ),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )


def pgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """Perform the Particle Gradient Descent algorithm.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to None.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """
    return expectation_maximisation(
        expectation_step=ParticleGradientExpectationStep(
            model=model, step_size=latent_step_size
        ),
        maximisation_step=MaximisationStep(
            model=model, optimiser=ox.sgd(theta_step_size)
        ),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )


def rms_soul(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
):
    """Perform the SoulEM algorithm with adagrad on the theta updates.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to None.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    return expectation_maximisation(
        expectation_step=SoulExpectationStep(model=model, step_size=latent_step_size),
        maximisation_step=MaximisationStep(
            model=model, optimiser=ox.rmsprop(theta_step_size)
        ),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )


def ada_soul(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
):
    """Perform the SoulEM algorithm with adagrad on the theta updates.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to None.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    return expectation_maximisation(
        expectation_step=SoulExpectationStep(model=model, step_size=latent_step_size),
        maximisation_step=MaximisationStep(
            model=model, optimiser=ox.adagrad(theta_step_size)
        ),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )


def adam_soul(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
):
    """Perform the SoulEM algorithm with adam on the theta updates.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to None.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    return expectation_maximisation(
        expectation_step=SoulExpectationStep(model=model, step_size=latent_step_size),
        maximisation_step=MaximisationStep(
            model=model, optimiser=ox.adam(theta_step_size)
        ),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )


def ada_pgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """Perform the Particle Gradient Descent algorithm with adagrad on the theta updates.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to None.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """
    return expectation_maximisation(
        expectation_step=ParticleGradientExpectationStep(
            model=model, step_size=latent_step_size
        ),
        maximisation_step=MaximisationStep(
            model=model, optimiser=ox.adagrad(theta_step_size)
        ),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )


def adam_pgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """Perform the Particle Gradient Descent algorithm with adam on the theta updates.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to None.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """
    return expectation_maximisation(
        expectation_step=ParticleGradientExpectationStep(
            model=model, step_size=latent_step_size
        ),
        maximisation_step=MaximisationStep(
            model=model, optimiser=ox.adam(theta_step_size)
        ),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )


def rms_pgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-2,
    theta_step_size: float = 1e-2,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """Perform the Particle Gradient Descent algorithm with RMSprop on the theta updates.

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-2.
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-2.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to None.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """
    return expectation_maximisation(
        expectation_step=ParticleGradientExpectationStep(
            model=model, step_size=latent_step_size
        ),
        maximisation_step=MaximisationStep(
            model=model, optimiser=ox.rmsprop(theta_step_size)
        ),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )

def m_svgd_em(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    latent_optimiser: GradientTransformation,
    theta_optimiser: GradientTransformation,
    num_steps: int,
    e_acceleration: float = 0.1,
    m_acceleration: float = 0.1,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """
    Perform EM algorithm with accelerated SVGD E-step and accelerated M-step.
    
    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        latent_optimiser (GradientTransformation): The latent optimiser.
        theta_optimiser (GradientTransformation): The parameter optimiser.
        num_steps (int): The number of steps to perform, K.
        e_acceleration (float): The acceleration parameter for E-step. Defaults to 0.1.
        m_acceleration (float): The acceleration parameter for M-step. Defaults to 0.1.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to {}.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    return expectation_maximisation(
        expectation_step=AcceleratedSteinExpectationStep(
            model=model, 
            optimiser=latent_optimiser,
            acceleration=e_acceleration
        ),
        maximisation_step=AcceleratedMaximisationStep(
            model=model, 
            optimiser=theta_optimiser,
            acceleration=m_acceleration
        ),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )

def mpgd(
    model: AbstractModel,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    latent_step_size: float = 1e-4,
    theta_step_size: float = 1e-4,
    gamma_x: float = 0.9,
    eta_x: float = 500.0,
    gamma_theta: float = 0.9,
    eta_theta: float = 500.0,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """
    Perform the Momentum Particle Gradient Descent (MPGD) algorithm.

    Uses the ACTUAL parameters from the authors' paper experiments that achieve
    faster convergence than PGD. Parameters taken directly from:
    mpgd-main/scripts/configs/cifar/mpd_cnet.py and mnist/mpd_mlp.py

    Args:
        model (AbstractModel): The model.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        latent_step_size (float, optional): The latent step size. Defaults to 1e-4 (paper value).
        theta_step_size (float, optional): The parameter step size. Defaults to 1e-4 (paper value).
        gamma_x (float, optional): Friction coefficient for particles. Defaults to 0.9 (paper value).
        eta_x (float, optional): Mass/temperature ratio for particles. Defaults to 500.0 (paper value).
        gamma_theta (float, optional): Friction coefficient for parameters. Defaults to 0.9 (paper value).
        eta_theta (float, optional): Mass/temperature ratio for parameters. Defaults to 500.0 (paper value).
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to None.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
        
    Note:
        These are the ACTUAL paper parameters that achieve faster convergence than PGD:
        - Smaller step sizes (1e-4 vs 1e-3) for precise momentum control
        - Higher eta values (500 vs 200) for strong momentum preservation  
        - Optimized gamma/eta ratios for the specific momentum dynamics
        
        For comparison with PGD, use the same step sizes in both algorithms.
    """
    return expectation_maximisation(
        expectation_step=MomentumParticleGradientExpectationStep(
            model=model, 
            step_size=latent_step_size,
            gamma_x=gamma_x,
            eta_x=eta_x
        ),
        maximisation_step=MomentumMaximisationStep(
            model=model, 
            optimiser=ox.sgd(theta_step_size),
            step_size=theta_step_size,
            gamma_theta=gamma_theta,
            eta_theta=eta_theta
        ),
        data=data,
        latent_init=latent_init,
        theta_init=theta_init,
        num_steps=num_steps,
        batch_size=batch_size,
        key=key,
        metrics=metrics,
    )

