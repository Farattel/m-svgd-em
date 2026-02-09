from jaxtyping import Array, Float, PyTree
from typing import Callable
from functools import reduce

import jax.numpy as jnp

from coinem.kernels import AbstractKernel


def stein_grad(
    particles: PyTree[Float[Array, "N *"]],
    score: Callable[[PyTree[Float[Array, "N *"]]], PyTree[Float[Array, "N *"]]],
    kernel: AbstractKernel,
) -> PyTree[Float[Array, "N *"]]:
    """
    Computes the kernelised Stein gradient of the particles (Liu and Wang, 2016).

    Args:
        particles (PyTree[Float[Array, "N *"]]): The current particles.
        score (Callable[[PyTree[Float[Array, "N *"]]], PyTree[Float[Array, "N *"]]]): The score function.
        kernel (AbstractKernel): The kernel.

    Returns:
        PyTree[Float[Array, "N *"]]: The updated particles.
    """

    # Compute the score function:
    s = score(particles)  # ∇x p(x)

    # Flatten the particles and the score function:
    flat_particles, unravel_func = ravel_pytree(particles)
    flat_score, _ = ravel_pytree(s)

    num_particles = flat_particles.shape[0]

    # Compute the kernel and its gradient:
    K, dK = kernel.K_dK(flat_particles)  # Kxx, ∇x Kxx

    # Compute the Stein gradient Φ(x) = (Kxx ∇x p(x) + ∇x Kxx) / N:
    flat_stein = (jnp.matmul(K, flat_score) + dK) / num_particles

    return unravel_func(flat_stein)


# Adapted from JAX source code:

import warnings

import numpy as np

from jax import lax
import jax.numpy as jnp

from jax._src import dtypes
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.util import safe_zip, unzip2, HashablePartial

zip = safe_zip
from functools import reduce
from operator import mul


def ravel_pytree(pytree):
    """Ravel (flatten) a pytree of arrays down to a 1D array.

    Args:
      pytree: a pytree of arrays and scalars to ravel.

    Returns:
      A pair where the first element is a 1D array representing the flattened and
      concatenated leaf values, with dtype determined by promoting the dtypes of
      leaf values, and the second element is a callable for unflattening a 1D
      vector of the same length back to a pytree of of the same structure as the
      input ``pytree``. If the input pytree is empty (i.e. has no leaves) then as
      a convention a 1D empty array of dtype float32 is returned in the first
      component of the output.

    For details on dtype promotion, see
    https://jax.readthedocs.io/en/latest/type_promotion.html.

    """
    leaves, treedef = tree_flatten(pytree)
    flat, unravel_list = _ravel_list(leaves)
    return flat, HashablePartial(unravel_pytree, treedef, unravel_list)


def unravel_pytree(treedef, unravel_list, flat):
    return tree_unflatten(treedef, unravel_list(flat))


def _ravel_list(lst):
    if not lst:
        return jnp.array([], jnp.float32), lambda _: []

    from_dtypes = tuple(dtypes.dtype(l) for l in lst)
    to_dtype = dtypes.result_type(*from_dtypes)
    sizes, shapes = unzip2((reduce(mul, jnp.shape(x)[1:]), jnp.shape(x)) for x in lst)
    indices = tuple(np.cumsum(sizes))

    if all(dt == to_dtype for dt in from_dtypes):
        # Skip any dtype conversion, resulting in a dtype-polymorphic `unravel`.
        # See https://github.com/google/jax/issues/7809.
        del from_dtypes, to_dtype
        raveled = jnp.hstack([arr.reshape(-1, s) for arr, s in zip(lst, sizes)])
        return raveled, HashablePartial(_unravel_list_single_dtype, indices, shapes)

    # When there is more than one distinct input dtype, we perform type
    # conversions and produce a dtype-specific unravel function.
    # ravel = lambda e: jnp.ravel(lax.convert_element_type(e, to_dtype))
    raveled = jnp.hstack(lst)
    unrav = HashablePartial(_unravel_list, indices, shapes, from_dtypes, to_dtype)
    return raveled, unrav


def _unravel_list_single_dtype(indices, shapes, arr):
    chunks = jnp.split(arr, indices, axis=1)[:-1]
    return [chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)]


def _unravel_list(indices, shapes, from_dtypes, to_dtype, arr):
    arr_dtype = dtypes.dtype(arr)
    if arr_dtype != to_dtype:
        raise TypeError(
            f"unravel function given array of dtype {arr_dtype}, "
            f"but expected dtype {to_dtype}"
        )
    chunks = jnp.split(arr, indices, axis=1)[:-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore complex-to-real cast warning
        return [
            lax.convert_element_type(chunk.reshape(shape), dtype)
            for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)
        ]


# # Adapted from JAX source code:

# import warnings

# import numpy as np

# from jax import lax
# import jax.numpy as jnp

# from jax._src import dtypes
# from jax._src.tree_util import tree_flatten, tree_unflatten
# from jax._src.util import safe_zip, unzip2, HashablePartial

# zip = safe_zip
# from functools import reduce
# from operator import add


# def ravel_pytree(pytree):
#   """Ravel (flatten) a pytree of arrays down to a 1D array.

#   Args:
#     pytree: a pytree of arrays and scalars to ravel.

#   Returns:
#     A pair where the first element is a 1D array representing the flattened and
#     concatenated leaf values, with dtype determined by promoting the dtypes of
#     leaf values, and the second element is a callable for unflattening a 1D
#     vector of the same length back to a pytree of of the same structure as the
#     input ``pytree``. If the input pytree is empty (i.e. has no leaves) then as
#     a convention a 1D empty array of dtype float32 is returned in the first
#     component of the output.

#   For details on dtype promotion, see
#   https://jax.readthedocs.io/en/latest/type_promotion.html.

#   """
#   leaves, treedef = tree_flatten(pytree)
#   flat, unravel_list = _ravel_list(leaves)
#   return flat, HashablePartial(unravel_pytree, treedef, unravel_list)

# def unravel_pytree(treedef, unravel_list, flat):
#   return tree_unflatten(treedef, unravel_list(flat))

# def _ravel_list(lst):
#   if not lst: return jnp.array([], jnp.float32), lambda _: []
#   from_dtypes = tuple(dtypes.dtype(l) for l in lst)
#   to_dtype = dtypes.result_type(*from_dtypes)
#   sizes, shapes = unzip2((reduce(add, jnp.shape(x)[1:]), jnp.shape(x)) for x in lst)
#   indices = tuple(np.cumsum(sizes))


#   if all(dt == to_dtype for dt in from_dtypes):
#     # Skip any dtype conversion, resulting in a dtype-polymorphic `unravel`.
#     # See https://github.com/google/jax/issues/7809.
#     del from_dtypes, to_dtype
#     raveled = jnp.hstack(lst)
#     return raveled, HashablePartial(_unravel_list_single_dtype, indices, shapes)

#   # When there is more than one distinct input dtype, we perform type
#   # conversions and produce a dtype-specific unravel function.
#   #ravel = lambda e: jnp.ravel(lax.convert_element_type(e, to_dtype))
#   raveled = jnp.hstack(lst)
#   unrav = HashablePartial(_unravel_list, indices, shapes, from_dtypes, to_dtype)
#   return raveled, unrav

# def _unravel_list_single_dtype(indices, shapes, arr):

#   chunks = jnp.split(arr, indices, axis=1)[:-1]
#   return [chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)]

# def _unravel_list(indices, shapes, from_dtypes, to_dtype, arr):
#   arr_dtype = dtypes.dtype(arr)
#   if arr_dtype != to_dtype:
#     raise TypeError(f"unravel function given array of dtype {arr_dtype}, "
#                     f"but expected dtype {to_dtype}")
#   chunks = jnp.split(arr, indices, axis=1)[:-1]
#   with warnings.catch_warnings():
#     warnings.simplefilter("ignore")  # ignore complex-to-real cast warning
#     return [lax.convert_element_type(chunk.reshape(shape), dtype)
#             for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)]
