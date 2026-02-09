from dataclasses import dataclass
from jaxtyping import Array, Float
from simple_pytree import Pytree, static_field
from functools import partial
import jax.numpy as jnp
import jax.scipy as jsp

__all__ = ["ComputeDistances", "cum_mean"]

"""A field that is not initialized and not printed in repr and not a Pytree leaf."""
static_hidden_field = partial(static_field, init=False, repr=False)


@dataclass
class ComputeDistances(Pytree):
    """Compute distances between points in a batch.

    Args:
        x (Float[Array, "N D"]): Input data.
    """

    x: Float[Array, "N D"]

    # dists attribute for caching distances!
    dists: Float[Array, "N N D"] = static_hidden_field()

    # square_dists attribute for caching sqaured-distances!
    square_dists: Float[Array, "N N"] = static_hidden_field()

    def __post_init__(self):
        # Matrix of entries [(x - y)].
        self.dists = self.x[:, None, :] - self.x[None, :, :]  # [N N D]

        # Matrix of entries [(x - y)^2].
        self.square_dists = jnp.sum(self.dists**2, axis=-1)  # [N N]


def cum_mean(x: Float[Array, "*"], axis: int = 0) -> Float[Array, "*"]:
    """
    Computes the cumulative mean of a JAX array along the specified axis.

    Args:
      x (Float[Array, "*"]): A JAX array.
      axis (int): An integer specifying the axis along which to compute the
        cumulative mean. Default is 0.

    Returns:
      A JAX array containing the cumulative mean along the specified axis.
    """
    n = jnp.arange(1, x.shape[axis] + 1)
    cumsum = jnp.cumsum(x, axis=axis)
    n_shape = [1] * x.ndim
    n_shape[axis] = -1
    n = n.reshape(n_shape)
    return cumsum / n


def procrustes(data1, data2):
    r"""Procrustes analysis, a similarity test for two data sets.

    Each input matrix is a set of points or vectors (the rows of the matrix).
    The dimension of the space is the number of columns of each matrix. Given
    two identically sized matrices, procrustes standardizes both such that:

    - :math:`tr(AA^{T}) = 1`.

    - Both sets of points are centered around the origin.

    Procrustes ([1]_, [2]_) then applies the optimal transform to the second
    matrix (including scaling/dilation, rotations, and reflections) to minimize
    :math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
    pointwise differences between the two input datasets.

    This function was not designed to handle datasets with different numbers of
    datapoints (rows).  If two data sets have different dimensionality
    (different number of columns), simply add columns of zeros to the smaller
    of the two.

    Parameters
    ----------
    data1 : array_like
        Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    data2 : array_like
        n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).

    Returns
    -------
    mtx1 : array_like
        A standardized version of `data1`.
    mtx2 : array_like
        The orientation of `data2` that best fits `data1`. Centered, but not
        necessarily :math:`tr(AA^{T}) = 1`.
    disparity : float
        :math:`M^{2}` as defined above.

    Raises
    ------
    ValueError
        If the input arrays are not two-dimensional.
        If the shape of the input arrays is different.
        If the input arrays have zero columns or zero rows.

    See Also
    --------
    scipy.linalg.orthogonal_procrustes
    scipy.spatial.distance.directed_hausdorff : Another similarity test
      for two data sets

    Notes
    -----
    - The disparity should not depend on the order of the input matrices, but
      the output matrices will, as only the first output matrix is guaranteed
      to be scaled such that :math:`tr(AA^{T}) = 1`.

    - Duplicate data points are generally ok, duplicating a data point will
      increase its effect on the procrustes fit.

    - The disparity scales as the number of points per input matrix.

    References
    ----------
    .. [1] Krzanowski, W. J. (2000). "Principles of Multivariate analysis".
    .. [2] Gower, J. C. (1975). "Generalized procrustes analysis".

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial import procrustes

    The matrix ``b`` is a rotated, shifted, scaled and mirrored version of
    ``a`` here:

    >>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
    >>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
    >>> mtx1, mtx2, disparity = procrustes(a, b)
    >>> round(disparity)
    0.0

    """
    mtx1 = data1
    mtx2 = data2

    # if mtx1.ndim != 2 or mtx2.ndim != 2:
    #     raise ValueError("Input matrices must be two-dimensional")
    # if mtx1.shape != mtx2.shape:
    #     raise ValueError("Input matrices must be of same shape")
    # if mtx1.size == 0:
    #     raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= jnp.mean(mtx1, 0)
    mtx2 -= jnp.mean(mtx2, 0)

    norm1 = jnp.linalg.norm(mtx1)
    norm2 = jnp.linalg.norm(mtx2)

    # if norm1 == 0 or norm2 == 0:
    #     raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = jnp.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = jnp.sum(jnp.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity


def orthogonal_procrustes(A, B):
    """
    Compute the matrix solution of the orthogonal Procrustes problem.

    Given matrices A and B of equal shape, find an orthogonal matrix R
    that most closely maps A to B using the algorithm given in [1]_.

    Parameters
    ----------
    A : (M, N) array_like
        Matrix to be mapped.
    B : (M, N) array_like
        Target matrix.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    R : (N, N) ndarray
        The matrix solution of the orthogonal Procrustes problem.
        Minimizes the Frobenius norm of ``(A @ R) - B``, subject to
        ``R.T @ R = I``.
    scale : float
        Sum of the singular values of ``A.T @ B``.

    Raises
    ------
    ValueError
        If the input array shapes don't match or if check_finite is True and
        the arrays contain Inf or NaN.

    Notes
    -----
    Note that unlike higher level Procrustes analyses of spatial data, this
    function only uses orthogonal transformations like rotations and
    reflections, and it does not use scaling or translation.

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Peter H. Schonemann, "A generalized solution of the orthogonal
           Procrustes problem", Psychometrica -- Vol. 31, No. 1, March, 1996.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import orthogonal_procrustes
    >>> A = np.array([[ 2,  0,  1], [-2,  0,  0]])

    Flip the order of columns and check for the anti-diagonal mapping

    >>> R, sca = orthogonal_procrustes(A, np.fliplr(A))
    >>> R
    array([[-5.34384992e-17,  0.00000000e+00,  1.00000000e+00],
           [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  0.00000000e+00, -7.85941422e-17]])
    >>> sca
    9.0

    """
    # if A.ndim != 2:
    #     raise ValueError('expected ndim to be 2, but observed %s' % A.ndim)
    # if A.shape != B.shape:
    #     raise ValueError('the shapes of A and B differ (%s vs %s)' % (
    #         A.shape, B.shape))
    # Be clever with transposes, with the intention to save memory.
    u, w, vt = jsp.linalg.svd(B.T.dot(A).T)

    R = u.dot(vt)
    scale = w.sum()
    return R, scale
