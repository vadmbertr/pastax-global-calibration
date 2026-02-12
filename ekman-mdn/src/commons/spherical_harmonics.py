# copy/paste from jax.scipy.special.sph_harm, fixing a bug in `_sph_harm` (https://github.com/jax-ml/jax/issues/20769)

from functools import partial
from typing import Any

import jax
from jax.core import concrete_or_error
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.jit
def _sph_harm(
    n: jax.Array,
    m: jax.Array,
    phi: jax.Array,
    legendre: jax.Array,
) -> jax.Array:
    """Computes the spherical harmonics."""

    legendre_val = legendre.at[jnp.abs(m), n, jnp.arange(len(phi))].get(mode="clip")

    angle = jnp.abs(m) * phi
    harmonics = legendre_val * jnp.exp(1j * angle)

    # Negative order.
    harmonics = jnp.where(m < 0, (-1.0) ** jnp.abs(m) * jnp.conjugate(harmonics), harmonics)

    return harmonics

 
def _gen_recurrence_mask(
    l_max: int, is_normalized: bool, dtype: Any
) -> tuple[jax.Array, jax.Array]:
    """Generates a mask for recurrence relation on the remaining entries.

    The remaining entries are with respect to the diagonal and offdiagonal
    entries.

    Args:
    l_max: see `gen_normalized_legendre`.
    is_normalized: True if the recurrence mask is used by normalized associated
        Legendre functions.

    Returns:
    Arrays representing the mask used by the recurrence relations.
    """

    # Computes all coefficients.
    m_mat, l_mat = jnp.meshgrid(jnp.arange(l_max + 1, dtype=dtype), jnp.arange(l_max + 1, dtype=dtype), indexing="ij")
    if is_normalized:
        c0 = l_mat * l_mat
        c1 = m_mat * m_mat
        c2 = 2.0 * l_mat
        c3 = (l_mat - 1.0) * (l_mat - 1.0)
        d0 = jnp.sqrt((4.0 * c0 - 1.0) / (c0 - c1))
        d1 = jnp.sqrt(((c2 + 1.0) * (c3 - c1)) / ((c2 - 3.0) * (c0 - c1)))
    else:
        d0 = (2.0 * l_mat - 1.0) / (l_mat - m_mat)
        d1 = (l_mat + m_mat - 1.0) / (l_mat - m_mat)

    d0_mask_indices = jnp.triu_indices(l_max + 1, 1)
    d1_mask_indices = jnp.triu_indices(l_max + 1, 2)
    d_zeros = jnp.zeros((l_max + 1, l_max + 1), dtype=dtype)
    d0_mask = d_zeros.at[d0_mask_indices].set(d0[d0_mask_indices])
    d1_mask = d_zeros.at[d1_mask_indices].set(d1[d1_mask_indices])

    # Creates a 3D mask that contains 1s on the diagonal plane and 0s elsewhere.
    # i = jnp.arange(l_max + 1)[:, None, None]
    # j = jnp.arange(l_max + 1)[None, :, None]
    # k = jnp.arange(l_max + 1)[None, None, :]
    i, j, k = jnp.ogrid[:l_max + 1, :l_max + 1, :l_max + 1]
    mask = (i + j - k == 0).astype(dtype)

    d0_mask_3d = jnp.einsum("jk,ijk->ijk", d0_mask, mask)
    d1_mask_3d = jnp.einsum("jk,ijk->ijk", d1_mask, mask)

    return (d0_mask_3d, d1_mask_3d)


@partial(jax.jit, static_argnums=(0, 2))
def gen_associated_legendre(l_max: int, theta: jax.Array, is_normalized: bool) -> jax.Array:
    x = jnp.cos(theta)  

    p = jnp.zeros((l_max + 1, l_max + 1, x.shape[0]), dtype=x.dtype)

    a_idx = jnp.arange(1, l_max + 1, dtype=x.dtype)
    b_idx = jnp.arange(l_max, dtype=x.dtype)
    if is_normalized:
        initial_value: ArrayLike = 0.5 / jnp.sqrt(jnp.pi)    # The initial value p(0,0).
        f_a = jnp.cumprod(-1 * jnp.sqrt(1.0 + 0.5 / a_idx))
        f_b = jnp.sqrt(2.0 * b_idx + 3.0)
    else:
        initial_value = 1.0    # The initial value p(0,0).
        f_a = jnp.cumprod(1.0 - 2.0 * a_idx)
        f_b = 2.0 * b_idx + 1.0

    p = p.at[(0, 0)].set(initial_value)

    # Compute the diagonal entries p(l,l) with recurrence.
    y = jnp.cumprod(
            jnp.broadcast_to(jnp.sqrt(1.0 - x * x), (l_max, x.shape[0])),
            axis=0)
    p_diag = initial_value * jnp.einsum("i,ij->ij", f_a, y)
    diag_indices = jnp.diag_indices(l_max + 1)
    p = p.at[(diag_indices[0][1:], diag_indices[1][1:])].set(p_diag)

    # Compute the off-diagonal entries with recurrence.
    p_offdiag = jnp.einsum(
        "ij,ij->ij", jnp.einsum("i,j->ij", f_b, x), p[jnp.diag_indices(l_max)]
    )
    offdiag_indices = (diag_indices[0][:l_max], diag_indices[1][:l_max] + 1)
    p = p.at[offdiag_indices].set(p_offdiag)

    # Compute the remaining entries with recurrence.
    d0_mask_3d, d1_mask_3d = _gen_recurrence_mask(l_max, is_normalized=is_normalized, dtype=x.dtype)

    def body_fun(i, p_val):
        coeff_0 = d0_mask_3d[i]
        coeff_1 = d1_mask_3d[i]
        h = (
            jnp.einsum(
                "ij,ijk->ijk", 
                coeff_0, 
                jnp.einsum(
                    "ijk,k->ijk", 
                    jnp.roll(p_val, shift=1, axis=1), 
                    x
                )
            ) - jnp.einsum("ij,ijk->ijk", coeff_1, jnp.roll(p_val, shift=2, axis=1))
        )
        p_val = p_val + h
        return p_val

    # TODO(jakevdp): use some sort of fixed-point procedure here instead?
    p = p.astype(jax.dtypes.result_type(p, x, d0_mask_3d))
    if l_max > 1:
        p = jax.lax.fori_loop(lower=2, upper=l_max+1, body_fun=body_fun, init_val=p)

    return p


def sph_harm_y(
    n: jax.Array,
    m: jax.Array,
    theta: jax.Array,
    phi: jax.Array,
    diff_n: int | None = None,
    n_max: int | None = None,
    legendre: jax.Array | None = None,
) -> jax.Array:
    r"""Computes the spherical harmonics.

    The JAX version has one extra argument `n_max`, the maximum value in `n`.

    The spherical harmonic of degree `n` and order `m` can be written as
    :math:`Y_n^m(\theta, \phi) = N_n^m * P_n^m(\cos \theta) * \exp(i m \phi)`,
    where :math:`N_n^m = \sqrt{\frac{\left(2n+1\right) \left(n-m\right)!}
    {4 \pi \left(n+m\right)!}}` is the normalization factor and :math:`\theta` and
    :math:`\phi` are the colatitude and longitude, respectively. :math:`N_n^m` is
    chosen in the way that the spherical harmonics form a set of orthonormal basis
    functions of :math:`L^2(S^2)`.

    Args:
        n: The degree of the harmonic; must have `n >= 0`. The standard notation for
            degree in descriptions of spherical harmonics is `l (lower case L)`. We
            use `n` here to be consistent with `scipy.special.sph_harm_y`. Return
            values for `n < 0` are undefined.
        m: The order of the harmonic; must have `|m| <= n`. Return values for
            `|m| > n` are undefined.
        theta: The polar (colatitudinal) coordinate; must be in [0, pi].
        phi: The azimuthal (longitudinal) coordinate; must be in [0, 2*pi].
        diff_n: Unsupported by JAX.
        n_max: The maximum degree `max(n)`. If the supplied `n_max` is not the true
            maximum value of `n`, the results are clipped to `n_max`. For example,
            `sph_harm(m=jnp.array([2]), n=jnp.array([10]), theta, phi, n_max=6)`
            actually returns
            `sph_harm(m=jnp.array([2]), n=jnp.array([6]), theta, phi, n_max=6)`
        legendre: Precomputed associated Legendre functions. If not provided, they
            will be computed internally.
    Returns:
        A 1D array containing the spherical harmonics at (m, n, theta, phi).
    """
    if diff_n is not None:
        raise NotImplementedError("The 'diff_n' argument to jax.scipy.special.sph_harm_y is not supported.")

    if jnp.isscalar(theta):
        theta = jnp.array([theta])
    if jnp.isscalar(phi):
        phi = jnp.array([phi])

    if n_max is None:
        n_max = jnp.max(n)
    n_max = concrete_or_error(
        int, n_max, "The `n_max` argument of `jnp.scipy.special.sph_harm` must "
        "be statically specified to use `sph_harm` within JAX transformations."
    )

    if legendre is None:
        legendre = gen_associated_legendre(n_max, theta, is_normalized=True)

    return _sph_harm(n, m, phi, legendre)
