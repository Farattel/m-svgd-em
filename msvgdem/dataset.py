from __future__ import annotations

from jaxtyping import Array, Float
from typing import Optional, Tuple
from simple_pytree import Pytree
from dataclasses import dataclass

import jax.random as jr

# New Jax Version does not have KeyArray(use the below depending on the version of jax you have otherwise use direct from python)
try:
    from jax.random import KeyArray
except ImportError:
    from typing import Any
    KeyArray = Any 


@dataclass
class Dataset(Pytree):
    """Base class for datasets.

    Attributes:
        X (Optional[Float[Array, "N D"]]): Input data.
        y (Optional[Float[Array, "N Q"]]): Output data.
    """

    X: Optional[Float[Array, "N D"]] = None
    y: Optional[Float[Array, "N Q"]] = None

    def __repr__(self) -> str:
        """Returns a string representation of the dataset."""
        repr = (
            f"- Number of observations: {self.n}\n- Input dimension:"
            f" {self.in_dim}\n- Output dimension: {self.out_dim}"
        )
        return repr

    def is_supervised(self) -> bool:
        """Returns `True` if the dataset is supervised."""
        return self.X is not None and self.y is not None

    def is_unsupervised(self) -> bool:
        """Returns `True` if the dataset is unsupervised."""
        return self.X is None and self.y is not None

    @property
    def n(self) -> int:
        """Number of observations."""
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        """Dimension of the inputs, X."""
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        """Dimension of the outputs, y."""
        return self.y.shape[1]


def get_batch(train_data: Dataset, batch_size: int, key: KeyArray) -> Dataset:
    """Batch the data into mini-batches. Sampling is done with replacement.

    Args:
        train_data (Dataset): The training dataset.
        batch_size (int): The batch size.
        key (KeyArray): The random key to use for the batch selection.

    Returns
    -------
        Dataset: The batched dataset.
    """
    x, y, n = train_data.X, train_data.y, train_data.n

    # Subsample mini-batch indices with replacement.
    indices = jr.choice(key, n, (batch_size,), replace=True)

    return Dataset(X=x[indices], y=y[indices])


__all__ = [
    "Dataset",
    "get_batch",
]
