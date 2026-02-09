from dataclasses import dataclass, field
from jaxtyping import Array, Float
from typing import Tuple
from simple_pytree import Pytree
from abc import abstractmethod
from jax import vmap, grad
import jax.numpy as jnp

from coinem.utils import ComputeDistances


@dataclass
class AbstractKernel(Pytree):
    """Base class for kernels."""

    @abstractmethod
    def K_dK(
        self, x: Float[Array, "N D"]
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N D"]]:
        """Compute kernel Gram matrix and gradient.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Tuple[Float[Array, "N N"], Float[Array, "N N D"]]: Kernel Gram matrix.
        """
        raise NotImplementedError


@dataclass
class RBF(AbstractKernel):
    """RBF kernel with static bandwidth selection.

    Args:
        h (Float[Array, "1"]): Bandwidth parameter if median_trick is False. Defaults to 1.0.
    """

    h: Float[Array, "1"] = field(default_factory=lambda: jnp.array([1.0]))

    def K_dK(
        self, x: Float[Array, "N D"]
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N D"]]:
        """Compute RBF kernel Gram matrix and gradient.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Tuple[Float[Array, "N N"], Float[Array, "N N D"]]: Kernel Gram matrix.
        """
        distances = ComputeDistances(x)

        K = jnp.exp(-0.5 * distances.square_dists / self.h**2)  # [N N]

        # ∇x Kxx = [∑_i exp(-0.5 * (x - y)^2 / h^2) * (x - y) / h^2]
        dK = jnp.sum(K[:, :, None] * distances.dists, axis=1) / self.h**2  # [N N D]

        return K, dK  # Kxx, ∇x Kxx


@dataclass
class MeanRBF(AbstractKernel):
    """RBF kernel with automatic bandwidth selection as the mean of the pairwise distances."""

    def K_dK(
        self, x: Float[Array, "N D"]
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N D"]]:
        """Compute RBF kernel Gram matrix and gradient.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Tuple[Float[Array, "N N"], Float[Array, "N N D"]]: Kernel Gram matrix.
        """
        jitter = 1e-18

        distances = ComputeDistances(x)
        h = (
            jnp.sqrt(
                0.5
                * jnp.mean(distances.square_dists)
                / jnp.log(distances.square_dists.shape[0] + 1.0)
            )
            + jitter
        )

        K = jnp.exp(-0.5 * distances.square_dists / h**2)  # [N N]

        # ∇x Kxx = [∑_i exp(-0.5 * (x - y)^2 / h^2) * (x - y) / h^2]
        dK = jnp.sum(K[:, :, None] * distances.dists, axis=1) / h**2  # [N N D]

        return K, dK  # Kxx, ∇x Kxx


@dataclass
class MedianRBF(AbstractKernel):
    """RBF kernel with automatic bandwidth selection as the median of the pairwise distances."""

    def K_dK(
        self, x: Float[Array, "N D"]
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N D"]]:
        """Compute RBF kernel Gram matrix and gradient.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Tuple[Float[Array, "N N"], Float[Array, "N N D"]]: Kernel Gram matrix.
        """
        jitter = 1e-18

        distances = ComputeDistances(x)
        h = (
            jnp.sqrt(
                0.5
                * jnp.median(distances.square_dists)
                / jnp.log(distances.square_dists.shape[0] + 1.0)
            )
            + jitter
        )

        K = jnp.exp(-0.5 * distances.square_dists / h**2)  # [N N]

        # ∇x Kxx = [∑_i exp(-0.5 * (x - y)^2 / h^2) * (x - y) / h^2]
        dK = jnp.sum(K[:, :, None] * distances.dists, axis=1) / h**2  # [N N D]

        return K, dK  # Kxx, ∇x Kxx


@dataclass
class AutoKernel(AbstractKernel):
    """Auto diff / vmap kernel."""

    @abstractmethod
    def __call__(
        self, xi: Float[Array, "D"], yi: Float[Array, "D"]
    ) -> Float[Array, "1"]:
        """Compute kernel between two points.

        Args:
            xi (Float[Array, "D"]): First point.
            yi (Float[Array, "D"]): Second point.

        Returns:
            Float[Array, "1"]: Kernel value.
        """
        raise NotImplementedError

    def K(self, x: Float[Array, "N D"]) -> Float[Array, "N N"]:
        """Compute kernel Gram matrix Kxx.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Float[Array, "N N"]: Kernel Gram matrix.
        """
        return vmap(lambda xi: vmap(lambda yi: self(xi, yi))(x))(x)

    def dK(self, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """Compute kernel gradient ∇x Kxx.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Float[Array, "N N D"]: Kernel gradient.
        """
        return jnp.sum(
            vmap(lambda yi: vmap(lambda xi: grad(self)(xi, yi))(x))(x), axis=0
        )

    def K_dK(
        self, x: Float[Array, "N D"]
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N D"]]:
        """Compute kernel Gram matrix Kxx and gradient ∇x Kxx.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Tuple[Float[Array, "N N"], Float[Array, "N N D"]]: Kernel Gram matrix and gradient.
        """
        return self.K(x), self.dK(x)


@dataclass
class AutoRBF(AutoKernel):
    """Auto diff / vmap RBF kernel.

    Args:
        h (Float[Array, "1"]): Bandwidth parameter if median_trick is False. Defaults to 1.0.
    """

    h: Float[Array, "1"] = field(default_factory=lambda: jnp.array([1.0]))

    def __call__(
        self, xi: Float[Array, "D"], yi: Float[Array, "D"]
    ) -> Float[Array, "1"]:
        """Compute kernel between two points.

        Args:
            xi (Float[Array, "D"]): First point.
            yi (Float[Array, "D"]): Second point.

        Returns:
            Float[Array, "1"]: Kernel value.
        """
        return jnp.exp(-0.5 * jnp.sum((xi - yi) ** 2) / self.h**2).squeeze()


@dataclass
class AutoMedianRBF(AutoKernel):
    """Auto diff / vmap RBF kernel with automatic bandwidth selection as the median of the pairwise distances."""

    def __call__(
        self, xi: Float[Array, "D"], yi: Float[Array, "D"], h: Array
    ) -> Float[Array, "1"]:
        """Compute kernel between two points.

        Args:
            xi (Float[Array, "D"]): First point.
            yi (Float[Array, "D"]): Second point.

        Returns:
            Float[Array, "1"]: Kernel value.
        """
        return jnp.exp(-0.5 * jnp.sum((xi - yi) ** 2) / h**2).squeeze()

    def K(self, x: Float[Array, "N D"]) -> Float[Array, "N N"]:
        """Compute kernel Gram matrix Kxx.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Float[Array, "N N"]: Kernel Gram matrix.
        """

        distances = ComputeDistances(x)
        h = jnp.sqrt(
            jnp.array([0.5])
            * jnp.median(distances.square_dists)
            / jnp.log(distances.square_dists.shape[0] + 1.0)
        )

        return vmap(lambda xi: vmap(lambda yi: self(xi, yi, h))(x))(x)

    def dK(self, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """Compute kernel gradient ∇x Kxx.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Float[Array, "N N D"]: Kernel gradient.
        """

        distances = ComputeDistances(x)
        h = jnp.sqrt(
            jnp.array([0.5])
            * jnp.median(distances.square_dists)
            / jnp.log(distances.square_dists.shape[0] + 1.0)
        )

        return jnp.sum(
            vmap(lambda yi: vmap(lambda xi: grad(self)(xi, yi, h))(x))(x), axis=0
        )

    def K_dK(
        self, x: Float[Array, "N D"]
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N D"]]:
        """Compute kernel Gram matrix Kxx and gradient ∇x Kxx.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Tuple[Float[Array, "N N"], Float[Array, "N N D"]]: Kernel Gram matrix and gradient.
        """
        distances = ComputeDistances(x)
        h = jnp.sqrt(
            jnp.array(0.5)
            * jnp.median(distances.square_dists)
            / jnp.log(distances.square_dists.shape[0] + 1.0)
        )

        K = vmap(lambda xi: vmap(lambda yi: self(xi, yi, h))(x))(x)
        dK = jnp.sum(
            vmap(lambda yi: vmap(lambda xi: grad(self)(xi, yi, h))(x))(x), axis=0
        )

        return K, dK
