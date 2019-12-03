"""Model functions for Fitzhugh-Nagumo hypoelliptic diffusion model."""

import jax.numpy as np


def generate_params(u):
    return {
        'sigma': np.exp(0.5 * u[0] - 1),
        'epsilon': np.exp(0.5 * u[1] - 2),
        'gamma': 0.5 * u[2] + 1,
        'beta': 0.5 * u[3] + 1,
    }


def generate_init_state(v_0):
    return np.array([-0.5, -0.5]) + v_0


def step_mean(x, delta, sigma, epsilon, gamma, beta):
    a = (x[0] - x[0]**3 - x[1]) / epsilon
    b = gamma * x[0] - x[1] + beta
    da_dx0 = (1 - 3 * x[0]**2) / epsilon
    da_dx1 = -1 / epsilon
    db_dx0 = gamma
    db_dx1 = -1
    half_delta = delta / 2
    return x + delta * np.array((
        a + half_delta * (da_dx0 * a + da_dx1 * b),
        b + half_delta * (db_dx0 * a + db_dx1 * b)
    ))


def step_sqrt_covar(x, delta, sigma, epsilon, gamma, beta):
    sqrt_3 = 3**0.5
    return sigma * delta**0.5 * np.array((
        (delta / (sqrt_3 * epsilon), 0),
        (delta / sqrt_3 - sqrt_3 / 2, 0.5)))


def generate_x_seq(q, delta, dim_state, num_param, dim_noise):
    u, v_0, v_r = np.split(q, (num_param, num_param + dim_state))
    params = generate_params(u)
    x_init = generate_init_state(v_0)
    v_seq = np.reshape(v_r, (-1, dim_noise))

    def step_func(x, v):
        x = (
            step_mean(x, delta, **params) +
            step_sqrt_covar(x, delta, **params) @ v)
        return (x, x)

    _, x_seq = lax.scan(step_func, x_init, v_seq)

    return x_seq, params


def obs_func(x_seq):
    return x_seq[..., 0:1]
