"""Hypoelliptic FitzHugh--Nagumo model of neural spiking."""

import sde
import symnum
import symnum.numpy as snp
import numpy as onp
import jax.numpy as jnp
from jax import lax

dim_x = 2
dim_w = 1
dim_z = 4
dim_v_0 = dim_x
dim_v = 2 * dim_w


def drift_func(x, z):
    σ, ε, γ, β = z
    return snp.array([(x[0] - x[0] ** 3 - x[1]) / ε, γ * x[0] - x[1] + β])


def diff_coeff(x, z):
    σ, ε, γ, β = z
    return snp.array([[0], [σ]])


forward_func = symnum.numpify_func(
    sde.integrators.strong_order_1p5_step(drift_func, diff_coeff),
    (dim_z,),
    (dim_x,),
    (dim_v,),
    None,
    numpy_module=jnp,
)


def obs_func(x_seq):
    return x_seq[..., 0:1]


def generate_z(u):
    # [σ, ϵ, γ, β]
    return jnp.array([jnp.exp(u[0]), jnp.exp(u[1]), jnp.exp(u[2]), u[3]])


def generate_σ_y(u):
    return jnp.exp(u[dim_z])


def generate_x_0(z, v_0):
    return v_0 - jnp.array([0, z[3]])


def generate_x_seq(z, x_0, v_seq, δ):
    def step_func(x, v):
        x_n = forward_func(z, x, v, δ)
        return x_n, x_n

    _, x_seq = lax.scan(step_func, x_0, v_seq)
    return x_seq


def generate_y_seq(z, x_0, v_seq, δ, num_steps_per_obs):
    x_seq = generate_x_seq(z, x_0, v_seq, δ)
    return obs_func(x_seq[num_steps_per_obs - 1 :: num_steps_per_obs])
