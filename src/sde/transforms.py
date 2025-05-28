"""Transforms for stochastic differential equation (SDE) systems."""

import sympy
import symnum
import symnum.numpy as snp
import symnum.diffops.symbolic as diffops


def transform_sde(forward_func, backward_func):
    """Compute modified drift and diffusion coefficients under bijective transform.

    Applies Ito's lemma to a Ito type stochastic differential equation (SDE) of the form

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