from jaxtyping import Array, Float, PyTree
from simple_pytree import Pytree
from jax import grad, vmap

from simple_pytree import Pytree
from dataclasses import dataclass
from jaxtyping import Float, Array
from jax import grad, vmap, jacobian
import jax.numpy as jnp
import jax.tree_util as jtu


from coinem.dataset import Dataset

__all__ = ["AbstractModel"]


@dataclass
class AbstractModel(Pytree):
    """Base class for p(θ, x)."""

    def log_prob(
        self,
        latent: PyTree[Float[Array, "D *"]],
        theta: PyTree[Float[Array, "Q *"]],
        data: Dataset,
    ) -> Float[Array, ""]:
        """Compute gradient of the objective function at x.

        Args:
            latent (PyTree[Float[Array, "D *"]]): Latent variables with leading axis of shape D.
            theta (PyTree[Float[Array, "Q *"]]): Parameters with leading axis shape Q.
            data (Dataset): Dataset.

        Returns:
            Float[Array, ""]: log-probability of the data.
        """

        raise NotImplementedError

    def score_latent(
        self,
        latent: PyTree[Float[Array, "D *"]],
        theta: PyTree[Float[Array, "Q *"]],
        data: Dataset,
    ) -> PyTree[Float[Array, "D *"]]:
        """
        Compute gradient of the objective function at latent variables. FOR A SINGLE PARTICLE.

        Args:
            latent (PyTree[Float[Array, "D *"]]): Latent variables with leading axis of shape D.
            theta (PyTree[Float[Array, "Q *"]]): Parameters with leading axis shape Q.
            data (Dataset): Dataset.

        Returns:
            PyTree[Float[Array, "D *"]]: The gradient of the log-density for the latent variables with leading axis of shape D.
        """
        return grad(self.log_prob, argnums=0)(latent, theta, data)

    def score_theta(
        self,
        latent: PyTree[Float[Array, "D *"]],
        theta: PyTree[Float[Array, "Q *"]],
        data: Dataset,
    ) -> PyTree[Float[Array, "Q *"]]:
        """
        Compute gradient of the objective function at theta.

        Args:
            latent (PyTree[Float[Array, "D *"]]): Latent variables with leading axis of shape D.
            theta (PyTree[Float[Array, "Q *"]]): Parameters with leading axis shape Q.
            data (Dataset): Dataset.

        Returns:
            PyTree[Float[Array, "Q *"]]: The gradient of the log-density for the latent variables with leading axis of shape Q.
        """

        return grad(self.log_prob, argnums=1)(latent, theta, data)

    def score_latent_particles(
        self,
        latent_particles: PyTree[Float[Array, "N D *"]],
        theta: PyTree[Float[Array, "Q *"]],
        data: Dataset,
    ) -> PyTree[Float[Array, "N D *"]]:
        """
        Compute gradient of the objective function at latent variables. MULTIPLE PARTICLES.

        Args:
            latent_particles (PyTree[Float[Array, "N D *"]]): latent_particles variables with first two axes of shape (N, D).
            theta (PyTree[Float[Array, "Q *"]]): Parameters with leading axis of shape Q.
            data (Dataset): Dataset.

        Returns:
            PyTree[Float[Array, "N D *"]]: The gradient of the log-density for the latent latent_particles of shape (N, D).
        """

        return vmap(lambda particle: self.score_latent(particle, theta, data))(
            latent_particles
        )

    def average_score_theta(
        self,
        latent_particles: PyTree[Float[Array, "N D *"]],
        theta: PyTree[Float[Array, "Q *"]],
        data: Dataset,
    ) -> PyTree[Float[Array, "Q *"]]:
        """
        Compute gradient of the objective function at theta. MULTIPLE PARTICLES.

        Args:
            latent_particles (PyTree[Float[Array, "N D *"]]): latent_particles variables with first two axes of shape (N, D).
            theta (PyTree[Float[Array, "Q *"]]): Parameters with leading axis of shape Q.
            data (Dataset): Dataset.

        Returns:
            PyTree[Float[Array, "Q *"]]: The gradient of the log-density for theta averaged over the particle cloud with leading axis of shape Q.
        """

        scores = vmap(lambda particle: self.score_theta(particle, theta, data))(
            latent_particles
        )

        return jtu.tree_map(jtu.Partial(jnp.mean, axis=0), scores)

    def optimal_theta(
        self, latent_particles: PyTree[Float[Array, "N D *"]]
    ) -> PyTree[Float[Array, "Q *"]]:
        """
        In certain models, the Mstep can be computed analytically. This function returns the optimal theta given the latent particles.

        Args:
            latent_particles (PyTree[Float[Array, "N D *"]]): latent_particles variables with first two axes of shape (N, D).

        Returns:
            PyTree[Float[Array, "Q *"]]: The optimal theta.
        """
        raise NotImplementedError

    # def average_hessian_theta(
    #     self,
    #     latent_particles: Float[Array, "N D"],
    #     theta: Float[Array, "Q"],
    #     data: Dataset,
    # ) -> Float[Array, "Q Q"]:
    #     """
    #     Compute gradient of the objective function at theta. MULTIPLE PARTICLES.

    #     Args:
    #         latent_particles (Float[Array, "N D"]): latent_particles variables of shape (N, D).
    #         theta (Float[Array, "Q"]): Parameters of shape (Q,).

    #     Returns:
    #         Float[Array, "D"]: The gradient of the log-density for theta averaged over the particle cloud of shape (Q,Q).
    #     """

    #     return jnp.mean(
    #         vmap(
    #             lambda particle: jacobian(self.score_theta, argnums=1)(
    #                 particle, theta, data
    #             )
    #         )(latent_particles),
    #         axis=0,
    #     )  # <- Not checked.
