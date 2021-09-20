"""Susceptible-infected-recovered model with time varying contact rate parameter."""

import sde
import symnum
import symnum.numpy as snp
import jax.numpy as jnp
from jax import lax

dim_x = 3
dim_y = 1
dim_w = 3
dim_z = 4
dim_v_0 = 1
dim_v = dim_w

N = 763  # total population size S + I + R


def drift_func(x, z):
    α = snp.exp(x[2])
    β, γ, ζ, ϵ = z
    return snp.array(
        [-α * x[0] * x[1] / N, α * x[0] * x[1] / N - β * x[1], γ * (ζ - x[2])]
    )


def diff_coeff(x, z):
    α = snp.exp(x[2])
    β, γ, ζ, ϵ = z
    return snp.array(
        [
            [snp.sqrt(α * x[0] * x[1] / N), 0, 0],
            [-snp.sqrt(α * x[0] * x[1] / N), snp.sqrt(β * x[1]), 0],
            [0, 0, ϵ],
        ]
    )


_forward_func = symnum.numpify_func(
    sde.integrators.euler_maruyama_step(
        *sde.transforms.transform_sde(
            lambda x: snp.array([snp.log(x[0]), snp.log(x[1]), x[2]]),
            lambda x: snp.array([snp.exp(x[0]), snp.exp(x[1]), x[2]]),
        )(drift_func, diff_coeff)
    ),
    (dim_z,),
    (dim_x,),
    (dim_v,),
    None,
    numpy_module=jnp,
)


def forward_func(z, x, v, δ):
    # Clip first two state components below at -500, in original domain corresponding to
    # exp(-500) ≈ 7 × 10^(-218) when updating state to prevent numerical NaN issues when
    # these state components tends to negative infinity. 500 was chosen as the cutoff to
    # avoid underflow / overflow as in double precision exp(-500) is non-zero and
    # exp(500) finite while for example exp(-1000) = 0 and exp(1000) = inf
    # We clip before and after _forward_func to avoid NaN gradients
    # https://github.com/tensorflow/probability/blob/master/discussion/where-nan.pdf
    x = x.at[:2].set(jnp.clip(x[:2], -500))
    x_ = _forward_func(z, x, v, δ)
    return jnp.array(
        [
            lax.select(x[0] > -500, x_[0], x[0]),
            lax.select(x[1] > -500, x_[1], x[1]),
            x_[2],
        ]
    )


def obs_func(x_seq):
    return jnp.exp(x_seq[..., 1:2])


def generate_z(u):
    return jnp.array(
        [
            jnp.exp(u[0]),  # β
            jnp.exp(u[1]),  # γ
            u[2],  # ζ
            jnp.exp(jnp.sqrt(0.75) * u[3] + 0.5 * u[1] - 3),  # ϵ
        ]
    )


def generate_x_0(z, v_0):
    return jnp.array([jnp.log(762.0), jnp.log(1.0), v_0[0]])


def generate_σ_y(u):
    return jnp.exp(u[dim_z])
