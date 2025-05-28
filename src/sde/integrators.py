"""Numerical timestepping methods for stochastic differential equation (SDE) systems."""

import sympy
import symnum.numpy as snp
import symnum.diffops.symbolic as diffops


def euler_maruyama_step(drift_func, diff_coeff):
    """Construct Euler-Maruyama integrator step function."""

    def forward_func(z, x, v, δ):
        return x + δ * drift_func(x, z) + δ ** 0.5 * diff_coeff(x, z) @ v

    return forward_func


def milstein_step(drift_func, diff_coeff, noise_type="diagonal"):
    """Construct Milstein scheme step function."""

    if noise_type in ("scalar", "diagonal"):

        def forward_func(z, x, v, δ):
            δω = snp.sqrt(δ) * v
            a = drift_func(x, z)
            B = diff_coeff(x, z)
            if noise_type == "diagonal":
                B_dB_dx = snp.array(
                    [B[i, i] * B[i, i].diff(x[i]) for i in range(v.shape[0])]
                )
            else:
                B_dB_dx = snp.array(
                    [(B * B[i].diff(x)).sum() for i in range(x.shape[0])]
                )
            x_ = x + δ * a + B @ δω + B_dB_dx * (δω ** 2 - δ) / 2
            return snp.array([sympy.simplify(x_[i]) for i in range(x.shape[0])])

    else:
        raise NotImplementedError(f"Noise type {noise_type} not implemented.")

    return forward_func


def strong_order_1p5_step(drift_func, diff_coeff, noise_type="additive"):
    """Construct strong-order 1.5 Taylor scheme step function."""

    if noise_type == "additive":

        def forward_func(z, x, v, δ):
            dim_noise = v.shape[0] // 2
            δω = snp.sqrt(δ) * v[:dim_noise]
            δζ = δ * snp.sqrt(δ) * (v[:dim_noise] + v[dim_noise:] / snp.sqrt(3)) / 2
            x_ = (
                x
                + δ * drift_func(x, z)
                + diff_coeff(x, z) @ δω
                + (δ ** 2 / 2)
                * diffusion_operator(drift_func, diff_coeff)(drift_func)(x, z)
                + sum(
                    Lj_operator(diff_coeff, j)(drift_func)(x, z) * δζ[j]
                    for j in range(dim_noise)
                )
            )
            return snp.array([sympy.simplify(x_[i]) for i in range(x.shape[0])])

    elif noise_type == "scalar":

        def forward_func(z, x, v, δ):
            δω = snp.sqrt(δ) * v[:1]
            δζ = δ * snp.sqrt(δ) * (v[:1] + v[1:] / snp.sqrt(3)) / 2
            x_ = (
                x
                + δ * drift_func(x, z)
                + diff_coeff(x, z) @ δω
                + Lj_operator(diff_coeff, 0)(diff_coeff)(x, z) @ (δω ** 2 - δ) / 2
                + Lj_operator(diff_coeff, 0)(drift_func)(x, z) * δζ
                + diffusion_operator(drift_func, diff_coeff)(
                    lambda x, z: diff_coeff(x, z)[:, 0]
                )(x, z)
                * (δω * δ - δζ)
                + (δ ** 2 / 2)
                * diffusion_operator(drift_func, diff_coeff)(drift_func)(x, z)
                + Lj_operator(diff_coeff, 0)(Lj_operator(diff_coeff, 0)(diff_coeff))(
                    x, z
                )
                @ (δω ** 3 / 3 - δ * δω)
            )
            return snp.array([sympy.simplify(x_[i]) for i in range(x.shape[0])])

    else:
        raise NotImplementedError(f"Noise type {noise_type} not implemented.")

    return forward_func


def diffusion_operator(drift_func, diff_coeff):
    """Construct diffusion operator for autonomous Ito stochastic differential equation.

    Diffusion operator here refers to the partial differential operator which is the
    infintesimal generator of the stochastic process.

    Args:
        drift_func (Callable[[SymbolicArray, SymbolicArray], SymbolicArray]): Function
            defining drift term of diffusion, accepting symbolic state and parameter
            vectors (1D arrays) as arguments and returning a symbolic vector (1D array)
            drift term.
        diff_coeff (Callable[[SymbolicArray, SymbolicArray], SymbolicArray]): Function
            defining diffusion coefficient term,  accepting symbolic state and parameter
            vectors (1D arrays) as arguments and returning a symbolic matrix (2D array)
            diffusion coefficient term.
    """

    def _diffusion_operator(func):
        def diffusion_operator_func(x, z):
            a = drift_func(x, z)
            B = diff_coeff(x, z)
            return (
                diffops.jacobian_vector_product(func)(x, z)(a)
                + diffops.matrix_hessian_product(func)(x, z)(B @ B.T) / 2
            )

        return diffusion_operator_func

    return _diffusion_operator


def Lj_operator(diff_coeff, j=0):
    """Construct Lj operator for autonomous Ito stochastic differential equation.

    Lj operator here refers to the Lʲ partial differential operator defined in Equation
    3.2 in Chapter 5 of Kloeden and Platen (1992)

        Lʲf(x) = ∑ₖ Bₖⱼ(x) ∂ₖf(x)

    Args:
        diff_coeff (Callable[[SymbolicArray, SymbolicArray], SymbolicArray]): Function
            defining diffusion coefficient term, accepting symbolic state and parameter
            vectors (1D arrays) as arguments and returning a symbolic matrix (2D array)
            diffusion coefficient term.
        j (int): Column index of diffusion coefficient term (zero-based).
    """

    def Lj(func):
        def Lj_func(x, z):
            B = diff_coeff(x, z)
            return diffops.jacobian_vector_product(func)(x, z)(B[:, j])

        return Lj_func

    return Lj
