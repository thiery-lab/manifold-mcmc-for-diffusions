import sympy
import symnum
import symnum.numpy as snp
import symnum.diffops.symbolic as diffops


def ito_transform(forward_func, backward_func):
    """Compute modified drift and diffusion coefficients under bijective transform.

    Applies Ito's lemma to a stochastic differential equation (SDE) of the form

        dX(τ) = a(X(τ), z) * dτ + B(X(τ), z) @ dW(τ)

    where `X` is a vector valued stochastic process, `z` a vector of parameters, `a` a
    drift function, `B` a diffusion coefficient function and `W` a Wiener noise process
    to derive an equivalent SDE in a random process `Y`

        dY(τ) = a'(Y(τ), z) * dτ + B'(Y(τ), z) @ dW(τ)

    defined such that `Y(τ) = f(X(τ))` where `f` is a bijective map.

    Args:
        forward_func: Function corresponding to forward evaluation of bijective map
            `f` defining state transformation.
        backward_func: Function corresponding to backward (inverse) evaluation of
            bijective map `f` defining state transformation.

    Returns:
        Transform function accepting two function arguments corresponding to the
        (untransformed) drift and diffusion coefficient functions and returning the
        corresponding tranformed drift and diffusion coefficient functions.
    """

    def transform(drift_func, diff_coeff):
        def transformed_drift_func(y, z):
            x = symnum.named_array("x", y.shape)
            a = drift_func(x, z)
            B = diff_coeff(x, z)
            x_y = backward_func(y)
            a_y = (
                diffops.jacobian_vector_product(forward_func)(x)(a)
                + diffops.matrix_hessian_product(forward_func)(x)(B @ B.T) / 2
            ).subs([(x_i, x_y_i) for x_i, x_y_i in zip(x.flatten(), x_y.flatten())])
            return snp.array([sympy.simplify(a_y[i]) for i in range(a_y.shape[0])])

        def transformed_diff_coeff(y, z):
            x = symnum.named_array("x", y.shape)
            B = diff_coeff(x, z)
            x_y = backward_func(y)
            B_y = (diffops.jacobian(forward_func)(x) @ B).subs(
                [(x_i, x_y_i) for x_i, x_y_i in zip(x.flatten(), x_y.flatten())]
            )
            return snp.array(
                [
                    [sympy.simplify(B_y[i, j]) for j in range(B_y.shape[1])]
                    for i in range(B_y.shape[0])
                ]
            )

        return transformed_drift_func, transformed_diff_coeff

    return transform


def euler_maruyama_step(drift_func, diff_coeff):
    """Construct Euler-Maruyama integrator step function."""

    def forward_func(z, x, v, δ):
        return x + δ * drift_func(x, z) + δ ** 0.5 * diff_coeff(x, z) @ v

    return forward_func


def milstein_step(drift_func, diff_coeff, noise_type="diagonal"):
    """Construct Milstein integrator step function for diagonal noise systems."""

    if noise_type == "diagonal":

        def forward_func(z, x, v, δ):
            a = drift_func(x, z)
            B = diff_coeff(x, z)
            dim_B = B.shape[0]
            B_dB_dx = snp.array([B[i, i] * B[i, i].diff(x[i]) for i in range(dim_B)])
            return x + δ * a + δ ** 0.5 * B @ v + δ * B_dB_dx * (v ** 2 - 1) / 2

    else:
        raise NotImplementedError(f"Noise type {noise_type} not implemented.")

    return forward_func


def strong_order_1p5_step(drift_func, diff_coeff, noise_type="additive"):
    """Construct strong-order 1.5 Taylor scheme step function."""

    if noise_type == "additive":

        def forward_func(z, x, v, δ):
            a = drift_func(x, z)
            da_dx = diffops.jacobian(drift_func)(x, z)
            B = diff_coeff(x, z)
            dim_noise = B.shape[1]
            d2a_dx2_BB = diffops.matrix_hessian_product(drift_func)(x, z)(B @ B.T)
            v_1, v_2 = v[:dim_noise], v[dim_noise:]
            return (
                x
                + δ * a
                + (δ ** 2 / 2) * (da_dx @ a + d2a_dx2_BB / 2)
                + δ ** 0.5 * B @ v_1
                + (δ ** 1.5 / 2) * da_dx @ B @ (v_1 + v_2 / snp.sqrt(3))
            )

    else:
        raise NotImplementedError(f"Noise type {noise_type} not implemented.")

    return forward_func
