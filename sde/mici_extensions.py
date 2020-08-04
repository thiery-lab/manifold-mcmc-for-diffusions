import logging
import numpy as onp
import jax.numpy as np
import jax.numpy.linalg as nla
import jax.scipy.linalg as sla
from jax import lax, api
import jax.experimental.optimizers as opt
from mici.systems import (
    System,
    cache_in_state,
    cache_in_state_with_aux,
)
from mici.matrices import (
    PositiveDefiniteBlockDiagonalMatrix,
    IdentityMatrix,
    PositiveDiagonalMatrix,
    PositiveScaledIdentityMatrix,
)
from mici.transitions import Transition
from mici.states import ChainState, _cache_key_func
from mici.solvers import maximum_norm
from mici.errors import ConvergenceError


logger = logging.getLogger(__name__)


def split(v, lengths):
    """Split an array along first dimension into slices of specified lengths."""
    i = 0
    parts = []
    for j in lengths:
        parts.append(v[i : i + j])
        i += j
    if i < len(v):
        parts.append(v[i:])
    return parts


def standard_normal_neg_log_dens(q):
    """Unnormalised negative log density of standard normal vector."""
    return 0.5 * onp.sum(q ** 2)


def standard_normal_grad_neg_log_dens(q):
    """Gradient and value of negative log density of standard normal vector."""
    return q, 0.5 * onp.sum(q ** 2)


def convert_to_numpy_pytree(jax_pytree):
    """Recursively convert 'pytree' of JAX arrays to NumPy arrays."""
    if isinstance(jax_pytree, np.DeviceArray):
        return onp.asarray(jax_pytree)
    elif isinstance(jax_pytree, (float, int, complex, bool)):
        return jax_pytree
    elif isinstance(jax_pytree, tuple):
        return tuple(convert_to_numpy_pytree(subtree) for subtree in jax_pytree)
    elif isinstance(jax_pytree, list):
        return [convert_to_numpy_pytree(subtree) for subtree in jax_pytree]
    elif isinstance(jax_pytree, dict):
        return {k: convert_to_numpy_pytree(v) for k, v in jax_pytree.items()}
    else:
        raise ValueError(f"Unknown jax_pytree node type {type(jax_pytree)}")


class ConditionedDiffusionConstrainedSystem(System):
    """Specialised mici system class for conditioned diffusion problems."""

    def __init__(
        self,
        obs_interval,
        num_steps_per_obs,
        num_obs_per_subseq,
        y_seq,
        dim_u,
        dim_x,
        dim_v,
        forward_func,
        generate_x_0,
        generate_z,
        obs_func,
        obs_noise_std=0,
        use_gaussian_splitting=False,
        metric=None,
        dim_v_0=None,
    ):
        """
        Args:
            obs_interval (float): Interobservation time interval.
            num_steps_per_obs (int): Number of discrete time steps to simulate
                between each observation time.
            num_obs_per_subseq (int): Average number of observations per
                partitioned subsequence. Must be a factor of `len(y_obs_seq)`.
            y_seq (array): Two-dimensional array containing observations at
                equally spaced time intervals, with first axis of array
                corresponding to observation time index (in order of increasing
                time) and second axis corresponding to dimension of each
                (vector-valued) observation.
            dim_u (int): Dimension of vector `u` mapping to parameter vector `z`.
            dim_x (int): Dimension of state vector `x`.
            dim_v (int): Dimension of noise vector `v` consumed by
                `forward_func` to approximate time step.
            forward_func (Callable[[array, array, array, float], array]):
                Function implementing forward step of time-discretisation of
                diffusion such that `forward_func(z, x, v, δ)` for parameter
                vector `z`, current state `x` at time `t`, standard normal
                vector `v` and  small timestep `δ` and is distributed
                approximately according to `X(t + δ) | X(t) = x, Z = z`.
            generate_x_0 (Callable[[array, array], array]): Generator function
                for the initial state such that `generator_x_0(z, v_0)` for
                parameter vector `z` and standard normal vector `v_0` is
                distributed according to prior distribution on `X(0) | Z = z`.
            generate_z (Callable[[array], array]): Generator function
                for parameter vector such that `generator_z(u)` for standard
                normal vector `u` is distributed according to prior distribution
                on parameter vector `Z`.
            obs_func (Callable[[array], array]): Function mapping from state
                vector `x` at an observation time to the corresponding observed
                vector `y = obs_func(x)`.
            obs_noise_std (float): Standard-deviation of independent and zero-mean
                Gaussian noise added to observations. If equal to zero (the default)
                noiseless observations will be assumed (with the resulting latent
                state not including a component for the observation noise).
            use_gaussian_splitting (bool): Whether to use Gaussian specific splitting
                    h₁(q) = ½log(∂c(q)ᵀ∂c(q)), h₂(q, p) = ½qᵀq + ½pᵀp
                Or the more standard splitting
                    h₁(q) = ½qᵀq  + ½log(∂c(q)ᵀM⁻¹∂c(q)), h₂(q, p) =  ½pᵀM⁻¹p
                In the former case the metric matrix representation is required to be
                the identity matrix. As the unconstrained integrator steps exactly
                the Gaussian prior (i.e. without the Gram log determinant term) in the
                Gaussian splitting case, using it can give improved performance in
                high dimensional systems where the step size is limited by the
                Hamiltonian error.
            metric (Matrix): Metric matrix representation. Should be either an
                `mici.matrices.IdentityMatrix` (compulsory if `use_gaussian_splitting`
                is `True`) or an `mici.matrices.SymmetricBlockDiagonalMatrix` instance,
                with in the latter case the matrix having two blocks on the diagonal,
                the left most of size `dim_z x dim_z`, and the rightmost being positive
                diagonal. Defaults to `mici.matrices.IdentityMatrix`.
            dim_v_0 (int): Dimension of vector used to generate initial state.
                Defaults to `dim_x` if `None`.
        """

        super().__init__(
            neg_log_dens=standard_normal_neg_log_dens,
            grad_neg_log_dens=standard_normal_grad_neg_log_dens,
        )
        if (
            use_gaussian_splitting
            and metric is not None
            and not isinstance(metric, IdentityMatrix)
        ):
            raise ValueError(
                "Only identity matrix metric can be used with Gaussian splitting"
            )
        elif metric is None:
            metric = IdentityMatrix()
        self.use_gaussian_splitting = use_gaussian_splitting
        self.metric = metric
        if isinstance(metric, IdentityMatrix):
            log_det_sqrt_metric_0 = 0
        elif isinstance(metric, PositiveDefiniteBlockDiagonalMatrix) and isinstance(
            metric.blocks[1],
            (PositiveDiagonalMatrix, IdentityMatrix, PositiveScaledIdentityMatrix),
        ):
            log_det_sqrt_metric_0 = metric.blocks[0].log_abs_det / 2
        else:
            raise NotImplementedError(
                "Only identity and block diagonal metrics with diagonal lower "
                "right block currently supported."
            )

        num_obs, dim_y = y_seq.shape
        δ = obs_interval / num_steps_per_obs
        if num_obs % num_obs_per_subseq != 0:
            raise NotImplementedError(
                "Only cases where num_obs_per_subseq is a factor of num_obs "
                "supported."
            )
        num_subseq = num_obs // num_obs_per_subseq
        obs_indices = slice(num_steps_per_obs - 1, None, num_steps_per_obs)
        num_step_per_subseq = num_obs_per_subseq * num_steps_per_obs
        y_subseqs_p0 = np.reshape(y_seq, (num_subseq, num_obs_per_subseq, -1))
        y_subseqs_p1 = split(
            y_seq, (num_obs_per_subseq // 2, num_obs - num_obs_per_subseq)
        )
        y_subseqs_p1[1] = np.reshape(
            y_subseqs_p1[1], (num_subseq - 1, num_obs_per_subseq, dim_y)
        )
        noisy_observations = obs_noise_std > 0
        dim_y = y_seq.shape[-1]
        dim_v_0 = dim_x if dim_v_0 is None else dim_v_0
        self.dim_u = dim_u
        self.dim_v = dim_v
        self.dim_v_0 = dim_v_0
        self.dim_q = (
            dim_u
            + dim_v_0
            + dim_v * num_obs * num_steps_per_obs
            + (num_obs * dim_y if noisy_observations else 0)
        )
        self.num_steps_per_obs = num_steps_per_obs
        self.generate_x_0 = generate_x_0
        self.generate_z = generate_z
        self.forward_func = forward_func
        self.δ = δ

        @api.jit
        def step_func(z, x, v):
            x_n = forward_func(z, x, v, δ)
            return (x_n, x_n)

        @api.jit
        def generate_x_obs_seq(q):
            """Generate state sequence at observation time indices."""
            u, v_0, v_seq_flat = split(q, (dim_u, dim_v_0))
            z = generate_z(u)
            x_0 = generate_x_0(z, v_0)
            v_seq = np.reshape(v_seq_flat, (-1, dim_v))
            _, x_seq = lax.scan(lambda x, v: step_func(z, x, v), x_0, v_seq)
            return x_seq[obs_indices]

        def generate_y_bar(z, w_0, v_seq, n, b):
            """Generate partial observation subsequence."""
            x_0 = generate_x_0(z, w_0) if b == 0 else w_0
            _, x_seq = lax.scan(lambda x, v: step_func(z, x, v), x_0, v_seq)
            y_seq = obs_func(x_seq[obs_indices])
            if noisy_observations:
                y_seq = y_seq + obs_noise_std * n
            if b == 2:
                return y_seq.flatten()
            elif noisy_observations:
                return np.concatenate((y_seq.flatten(), x_seq[-1]))
            else:
                return np.concatenate((y_seq[:-1].flatten(), x_seq[-1]))

        @api.partial(api.jit, static_argnums=(4,))
        def partition_into_subseqs(v_seq, v_0, n, x_obs_seq, partition=0):
            """Partition noise increment and observation sequences.

            Partitition sequences in to either `num_subseq` equally sized
            subsequences (`partition == 0`)  or `num_subseq - 1` equally sized
            subsequences plus initial and final 'half' subsequences.
            """
            slice_end = None if noisy_observations else -1
            if partition == 0:
                v_subseqs = v_seq.reshape((num_subseq, num_step_per_subseq, dim_v))
                v_subseqs = (v_subseqs[0], v_subseqs[1:-1], v_subseqs[-1])
                if noisy_observations:
                    n_subseqs = n.reshape((num_subseq, num_obs_per_subseq, dim_y))
                    n_subseqs = (n_subseqs[0], n_subseqs[1:-1], n_subseqs[-1])
                else:
                    n_subseqs = (None,) * 3
                x_obs_subseqs = x_obs_seq.reshape(
                    (num_subseq, num_obs_per_subseq, dim_x)
                )
                w_inits = (v_0, x_obs_subseqs[:-2, -1], x_obs_subseqs[-2, -1])
                y_bars = (
                    np.concatenate(
                        (y_subseqs_p0[0, :slice_end].flatten(), x_obs_subseqs[0, -1])
                    ),
                    np.concatenate(
                        (
                            y_subseqs_p0[1:-1, :slice_end].reshape(
                                (num_subseq - 2, -1)
                            ),
                            x_obs_subseqs[1:-1, -1],
                        ),
                        -1,
                    ),
                    y_subseqs_p0[-1].flatten(),
                )
            else:
                v_subseqs = split(
                    v_seq,
                    (
                        (num_obs_per_subseq // 2) * num_steps_per_obs,
                        num_step_per_subseq * (num_subseq - 1),
                    ),
                )
                v_subseqs[1] = v_subseqs[1].reshape(
                    (num_subseq - 1, num_step_per_subseq, dim_v)
                )
                if noisy_observations:
                    n_subseqs = split(
                        n,
                        (
                            (num_obs_per_subseq // 2),
                            num_obs_per_subseq * (num_subseq - 1),
                        ),
                    )
                    n_subseqs[1] = n_subseqs[1].reshape(
                        (num_subseq - 1, num_obs_per_subseq, dim_y)
                    )
                else:
                    n_subseqs = (None,) * 3
                x_obs_subseqs = split(
                    x_obs_seq, (num_obs_per_subseq // 2, num_obs - num_obs_per_subseq)
                )
                x_obs_subseqs[1] = x_obs_subseqs[1].reshape(
                    (num_subseq - 1, num_obs_per_subseq, dim_x)
                )
                w_inits = (
                    v_0,
                    np.concatenate(
                        (x_obs_subseqs[0][-1:], x_obs_subseqs[1][:-1, -1]), 0
                    ),
                    x_obs_subseqs[1][-1, -1],
                )
                y_bars = (
                    np.concatenate(
                        (y_subseqs_p1[0][:slice_end].flatten(), x_obs_subseqs[0][-1])
                    ),
                    np.concatenate(
                        (
                            y_subseqs_p1[1][:, :slice_end].reshape(
                                (num_subseq - 1, -1)
                            ),
                            x_obs_subseqs[1][:, -1],
                        ),
                        -1,
                    ),
                    y_subseqs_p1[2].flatten(),
                )
            return v_subseqs, n_subseqs, w_inits, y_bars

        @api.partial(api.jit, static_argnums=(2,))
        def constr(q, x_obs_seq, partition=0):
            """Calculate constraint function for current partition."""
            if noisy_observations:
                u, v_0, v_seq_flat, n_flat = split(
                    q, (dim_u, dim_v_0, num_obs * dim_v * num_steps_per_obs, num_obs)
                )
                n = n_flat.reshape((-1, dim_y))
            else:
                u, v_0, v_seq_flat = split(q, (dim_u, dim_v_0,))
                n = None
            v_seq = v_seq_flat.reshape((-1, dim_v))
            z = generate_z(u)
            (v_subseqs, n_subseqs, w_inits, y_bars) = partition_into_subseqs(
                v_seq, v_0, n, x_obs_seq, partition
            )
            gen_funcs = (
                generate_y_bar,
                api.vmap(
                    generate_y_bar,
                    (None, 0, 0, 0 if noisy_observations else None, None),
                ),
                generate_y_bar,
            )
            return np.concatenate(
                [
                    (
                        gen_funcs[b](z, w_inits[b], v_subseqs[b], n_subseqs[b], b)
                        - y_bars[b]
                    ).flatten()
                    for b in range(3)
                ]
            )

        @api.jit
        def init_objective(q, x_obs_seq, reg_coeff):
            """Optimisation objective to find initial state on manifold."""
            u, v_0, v_seq_flat = split(q, (dim_u, dim_v_0,))
            v_subseqs = v_seq_flat.reshape((num_obs, num_steps_per_obs, dim_v))
            z = generate_z(u)
            x_0 = generate_x_0(z, v_0)
            x_inits = np.concatenate((x_0[None], x_obs_seq[:-1]), 0)

            def generate_final_state(z, v_seq, x_0):
                _, x_seq = lax.scan(lambda x, v: step_func(z, x, v), x_0, v_seq)
                return x_seq[-1]

            c = (
                api.vmap(generate_final_state, in_axes=(None, 0, 0))(
                    z, v_subseqs, x_inits
                )
                - x_obs_seq
            )
            return 0.5 * np.mean(c ** 2) + 0.5 * reg_coeff * np.mean(q ** 2), c

        @api.partial(api.jit, static_argnums=(2,))
        def jacob_constr_blocks(q, x_obs_seq, partition=0):
            """Return non-zero blocks of constraint function Jacobian.

            Input state `q` can be decomposed into `q = [u, v₀, v₁, v₂]` where global
            latent variables (parameters) are determined by `u`, initial subsequence by
            `v₀`, middle subsequences by `v₁` and final subsequence by `v₂`.

            Constraint function can then be decomposed as

                c(q) = [c₀(u, v₀), c₁(u, v₁), c₂(u, v₂)]

            abd the constraint Jacobian has the block structure

                ∂c(q) = [[∂₀c₀(u, v₀), ∂₁c₀(u, v₀),     0,     ,     0      ]
                         [∂₀c₁(u, v₁),     0      , ∂₁c₁(u, v₁),     0      ]
                         [∂₀c₂(u, v₀),     0      ,     0      , ∂₁c₂(u, v₂)]]

            where `∂₁c₁(u, v₁)` is itself block diagonal.
            """

            def g_y_bar(u, v, w_0, n, b):
                z = generate_z(u)
                if b == 0:
                    w_0, v = split(v, (dim_x,))
                v_seq = v.reshape((-1, dim_v))
                return generate_y_bar(z, w_0, v_seq, n, b)

            if noisy_observations:
                u, v_0, v_seq_flat, n_flat = split(
                    q, (dim_u, dim_v_0, num_obs * dim_v * num_steps_per_obs, num_obs)
                )
                n = n_flat.reshape((-1, dim_y))
            else:
                u, v_0, v_seq_flat = split(q, (dim_u, dim_v_0,))
                n = None
            v_seq = v_seq_flat.reshape((-1, dim_v))
            (v_subseqs, n_subseqs, w_inits, _) = partition_into_subseqs(
                v_seq, v_0, n, x_obs_seq, partition
            )
            v_bars = (
                np.concatenate([v_0, v_subseqs[0].flatten()]),
                v_subseqs[1].reshape((v_subseqs[1].shape[0], -1)),
                v_subseqs[2].flatten(),
            )
            jac_g_y_bar = api.jacrev(g_y_bar, (0, 1))
            jacob_funcs = (
                jac_g_y_bar,
                api.vmap(
                    jac_g_y_bar, (None, 0, 0, 0 if noisy_observations else None, None)
                ),
                jac_g_y_bar,
            )
            return tuple(
                zip(
                    *(
                        jacob_funcs[b](u, v_bars[b], w_inits[b], n_subseqs[b], b)
                        for b in range(3)
                    )
                )
            )

        @api.jit
        def chol_gram_blocks(dc_du_blocks, dc_dv_blocks):
            """Calculate Cholesky factors of decomposition of `∂c(q) M⁻¹ ∂c(q)ᵀ`.

            The constraint Jacobian decomposes as

                ∂c([u, v]) = [∂₀c̅(u, v), ∂₁c̅(u, v)]

            where `c̅(u, v) = c([u, v])` and `∂₀c̅(u, v)` is a dense tall rectangular
            matrix and `∂₁c̅(u, v)` is a rectangular block diagonal matrix. Similarly
            the metric matrix represention `M` is assumed to have the block structure
            `M = diag(M₀, M₁)` where `M₀` is a `dim_u × dim_u` square matrix and `M₁` is
            a `dim_v × dim_v` diagonal matrix.

            The Gram matrix can therefore be decomposed as

                ∂c([u, v]) M⁻¹ ∂c([u, v])ᵀ =
                ∂₀c̅(u, v) M₀⁻¹ ∂₀c̅(u, v)ᵀ + ∂₁c̅(u, v) M₁⁻¹ ∂₁c̅(u, v)ᵀ

            denoting

                D(u, v) = ∂₁c̅(u, v) M₁⁻¹ ∂₁c̅(u, v)ᵀ

            with `D` square block diagonal and positive definite and

                C(u, v) = M₀ + ∂₀c̅(u, v)ᵀ D(u, v)⁻¹ ∂₀c̅(u, v)

            with `C` positive definite, by the Woodbury matrix identity we have

                (∂c([u, v]) M⁻¹ ∂c([u, v])ᵀ)⁻¹ =
                D(u, v)⁻¹ - D(u, v)⁻¹ ∂₀c̅(u, v) C(u, v)⁻¹ ∂₀c̅(u, v)ᵀ D(u, v)⁻¹

            Therefore by computing the Cholesky decomposition of `C(u, v)` and the
            blocks of `D(u, v)` we can solve for systems in the Gram matrix.
            """
            M_0 = get_M_0_matrix()
            D_blocks = compute_D_blocks(dc_dv_blocks, dc_dv_blocks)
            chol_D_blocks = tuple(nla.cholesky(D_blocks[i]) for i in range(3))
            D_inv_dc_du_blocks = tuple(
                sla.cho_solve((chol_D_blocks[i], True), dc_du_blocks[i])
                for i in range(3)
            )
            chol_C = nla.cholesky(
                M_0
                + (
                    dc_du_blocks[0].T @ D_inv_dc_du_blocks[0]
                    + np.einsum("ijk,ijl->kl", dc_du_blocks[1], D_inv_dc_du_blocks[1])
                    + dc_du_blocks[2].T @ D_inv_dc_du_blocks[2]
                )
            )
            return chol_C, chol_D_blocks

        @api.jit
        def lu_jacob_product_blocks(
            dc_du_l_blocks, dc_dv_l_blocks, dc_du_r_blocks, dc_dv_r_blocks
        ):
            """Calculate LU factors of decomposition of ∂c(q) M⁻¹ ∂c(q')ᵀ

            The constraint Jacobian decomposes as

                ∂c([u, v]) = [∂₀c̅(u, v), ∂₁c̅(u, v)]

            where c̅(u, v) = c([u, v]) and ∂₀c̅(u, v) is a dense tall
            rectangular matrix and ∂₁c̅(u, v) is a rectangular block diagonal
            matrix. Similarly the metric matrix represention M is assumed to
            have the block structure M = diag(M₀, M₁) where M₀ is a square
            matrix of the same size as u and M₁ is a square matrix of the same
            size as v.

            The Jacobian product ∂c([u, v]) M⁻¹ ∂c([u', v'])ᵀ can therefore be
            decomposed as

                ∂c([u, v]) M⁻¹ ∂c([u', v'])ᵀ =
                ∂₀c̅(u, v) M₀⁻¹ ∂₀c̅(u', v')ᵀ + ∂₁c̅(u', v') M₁⁻¹ ∂₁c̅(u', v')ᵀ

            denoting

                D(u, v, u', v') = ∂₁c̅(u, v) M₁⁻¹ ∂₁c̅(u', v')ᵀ

            with D square block diagonal and

                C(u, v, u', v') = M₀ + ∂₀c̅(u', v')ᵀ D(u, v)⁻¹ ∂₀c̅(u, v)

            we then have by the Woodbury matrix identity

                (∂c([u, v]) M⁻¹ ∂c([u', v'])ᵀ)⁻¹ =
                D(u, v, u', v')⁻¹ -
                D(u, v, u', v')⁻¹ ∂₀c̅(u, v) C(u, v, u', v')⁻¹ ∂₀c̅(u', v')ᵀ
                D(u, v, u', v')⁻¹

            Therefore by computing the Cholesky decomposition of C and
            the blocks of D we can solve for systems in the Jacobian product.
            """
            M_0 = get_M_0_matrix()
            D_blocks = compute_D_blocks(dc_dv_l_blocks, dc_dv_r_blocks)
            lu_and_piv_D_blocks = tuple(sla.lu_factor(D_blocks[i]) for i in range(3))
            D_inv_dc_du_l_blocks = tuple(
                sla.lu_solve(lu_and_piv_D_blocks[i], dc_du_l_blocks[i])
                for i in range(3)
            )
            lu_and_piv_C = sla.lu_factor(
                M_0
                + (
                    dc_du_r_blocks[0].T @ D_inv_dc_du_l_blocks[0]
                    + np.einsum(
                        "ijk,ijl->kl", dc_du_r_blocks[1], D_inv_dc_du_l_blocks[1]
                    )
                    + dc_du_r_blocks[2].T @ D_inv_dc_du_l_blocks[2]
                )
            )
            return lu_and_piv_C, lu_and_piv_D_blocks

        def compute_D_blocks(dc_dv_l_blocks, dc_dv_r_blocks):
            if isinstance(metric, IdentityMatrix) or isinstance(
                metric.blocks[1], IdentityMatrix
            ):
                dc_dv_l_inv_M_1_blocks = dc_dv_l_blocks
            elif isinstance(metric.blocks[1], PositiveScaledIdentityMatrix):
                scalar = metric.blocks[1].scalar
                dc_dv_l_inv_M_1_blocks = tuple(
                    dc_dv_l_blocks[i] / scalar for i in range(3)
                )
            else:
                M_1_diag = metric.blocks[1].diagonal
                M_1_diag_blocks = split(
                    M_1_diag,
                    (
                        dc_dv_l_blocks[0].shape[1],
                        dc_dv_l_blocks[1].shape[0] * dc_dv_l_blocks[1].shape[2],
                    ),
                )
                M_1_diag_blocks[1] = M_1_diag_blocks[1].reshape(
                    (dc_dv_l_blocks[1].shape[0], dc_dv_l_blocks[1].shape[2])
                )
                dc_dv_l_inv_M_1_blocks = (
                    dc_dv_l_blocks[i] / M_1_diag_blocks[i][..., None, :]
                    for i in range(3)
                )
            D_blocks = [
                np.einsum("...ij,...kj", dc_dv_l_inv_M_1_blocks[i], dc_dv_r_blocks[i])
                for i in range(3)
            ]
            if noisy_observations:
                for i in range(3):
                    diag_indices = np.diag_indices(D_blocks[i].shape[-2])
                    if i == 2:
                        D_blocks[i] = (
                            D_blocks[i]
                            .at[(...,) + diag_indices]
                            .add(obs_noise_std ** 2)
                        )
                    else:
                        D_blocks[i] = (
                            D_blocks[i]
                            .at[(...,) + diag_indices]
                            .add(
                                obs_noise_std ** 2
                                * np.concatenate(
                                    [
                                        np.ones(D_blocks[i].shape[-1] - dim_x),
                                        np.zeros(dim_x),
                                    ]
                                )
                            )
                        )
            return D_blocks

        def get_M_0_matrix():
            if isinstance(metric, IdentityMatrix):
                return np.identity(dim_u)
            else:
                return metric.blocks[0].array

        @api.jit
        def log_det_sqrt_gram_from_chol(chol_C, chol_D_blocks):
            """Calculate log-determinant of Gram matrix from block Cholesky factors."""
            return (
                sum(
                    np.log(np.abs(chol_D_blocks[i].diagonal(0, -2, -1))).sum()
                    for i in range(3)
                )
                + np.log(np.abs(chol_C.diagonal())).sum()
                - log_det_sqrt_metric_0
            )

        @api.partial(api.jit, static_argnums=(2,))
        def log_det_sqrt_gram(q, x_obs_seq, partition=0):
            """Calculate log-determinant of constraint Jacobian Gram matrix."""
            dc_du_blocks, dc_dv_blocks = jacob_constr_blocks(q, x_obs_seq, partition)
            chol_C, chol_D_blocks = chol_gram_blocks(dc_du_blocks, dc_dv_blocks)
            return (
                log_det_sqrt_gram_from_chol(chol_C, chol_D_blocks),
                ((dc_du_blocks, dc_dv_blocks), (chol_C, chol_D_blocks)),
            )

        @api.jit
        def lmult_by_jacob_constr(dc_du_blocks, dc_dv_blocks, vct):
            """Left-multiply vector by constraint Jacobian matrix."""
            if noisy_observations:
                vct_u, vct_v, vct_n = split(
                    vct, (dim_u, dim_v_0 + num_obs * num_steps_per_obs * dim_v,)
                )
            else:
                vct_u, vct_v = split(vct, (dim_u,))
            n_block_1 = dc_dv_blocks[1].shape[0]
            j0, j2 = (
                dc_dv_blocks[0].shape[1],
                dc_dv_blocks[2].shape[1],
            )
            k0, k2 = (
                dc_dv_blocks[0].shape[0] - dim_x,
                dc_dv_blocks[2].shape[0],
            )
            jacob_vct = np.vstack(
                (dc_du_blocks[0], dc_du_blocks[1].reshape((-1, dim_u)), dc_du_blocks[2])
            ) @ vct_u + np.concatenate(
                (
                    dc_dv_blocks[0] @ vct_v[:j0],
                    np.einsum(
                        "ijk,ik->ij",
                        dc_dv_blocks[1],
                        vct_v[j0:-j2].reshape((n_block_1, -1)),
                    ).flatten(),
                    dc_dv_blocks[2] @ vct_v[-j2:],
                )
            )
            if noisy_observations:
                jacob_vct += obs_noise_std * np.concatenate(
                    [
                        np.concatenate([vct_n[:k0], np.zeros(dim_x)]),
                        np.concatenate(
                            [
                                vct_n[k0:-k2].reshape((n_block_1, -1)),
                                np.zeros((n_block_1, dim_x)),
                            ],
                            1,
                        ).flatten(),
                        vct_n[-k2:],
                    ]
                )
            return jacob_vct

        @api.jit
        def rmult_by_jacob_constr(dc_du_blocks, dc_dv_blocks, vct):
            """Right-multiply vector by constraint Jacobian matrix."""
            vct_parts = split(
                vct,
                (
                    dc_du_blocks[0].shape[0],
                    dc_du_blocks[1].shape[0] * dc_du_blocks[1].shape[1],
                ),
            )
            vct_parts[1] = vct_parts[1].reshape(dc_du_blocks[1].shape[:2])
            return np.concatenate(
                [
                    vct_parts[0] @ dc_du_blocks[0]
                    + np.einsum("ij,ijk->k", vct_parts[1], dc_du_blocks[1])
                    + vct_parts[2] @ dc_du_blocks[2],
                    vct_parts[0] @ dc_dv_blocks[0],
                    np.einsum("ij,ijk->ik", vct_parts[1], dc_dv_blocks[1]).flatten(),
                    vct_parts[2] @ dc_dv_blocks[2],
                ]
                + (
                    [
                        obs_noise_std * vct_parts[0][:-dim_x],
                        obs_noise_std * vct_parts[1][:, :-dim_x].flatten(),
                        obs_noise_std * vct_parts[2],
                    ]
                    if noisy_observations
                    else []
                )
            )

        @api.jit
        def lmult_by_inv_gram(dc_du_blocks, dc_dv_blocks, chol_C, chol_D_blocks, vct):
            """Left-multiply vector by inverse Gram matrix."""
            vct_parts = split(
                vct,
                (
                    dc_du_blocks[0].shape[0],
                    dc_du_blocks[1].shape[0] * dc_du_blocks[1].shape[1],
                ),
            )
            vct_parts[1] = np.reshape(vct_parts[1], dc_du_blocks[1].shape[:2])
            D_inv_vct_blocks = tuple(
                sla.cho_solve((chol_D_blocks[i], True), vct_parts[i]) for i in range(3)
            )
            dc_du_T_D_inv_vct = sum(
                np.einsum("...jk,...j->k", dc_du_blocks[i], D_inv_vct_blocks[i])
                for i in range(3)
            )
            C_inv_dc_du_T_D_inv_vct = sla.cho_solve((chol_C, True), dc_du_T_D_inv_vct)
            return np.concatenate(
                [
                    sla.cho_solve(
                        (chol_D_blocks[i], True),
                        vct_parts[i] - dc_du_blocks[i] @ C_inv_dc_du_T_D_inv_vct,
                    ).flatten()
                    for i in range(3)
                ]
            )

        @api.jit
        def lmult_by_inv_jacob_product(
            dc_du_l_blocks,
            dc_dv_l_blocks,
            dc_du_r_blocks,
            dc_dv_r_blocks,
            lu_and_piv_C,
            lu_and_piv_D_blocks,
            vct,
        ):
            """Left-multiply vector by inverse of Jacobian product matrix."""
            vct_parts = split(
                vct,
                (
                    dc_du_l_blocks[0].shape[0],
                    dc_du_l_blocks[1].shape[0] * dc_du_l_blocks[1].shape[1],
                ),
            )
            vct_parts[1] = np.reshape(vct_parts[1], dc_du_l_blocks[1].shape[:2])
            D_inv_vct_blocks = tuple(
                sla.lu_solve(lu_and_piv_D_blocks[i], vct_parts[i]) for i in range(3)
            )
            dc_du_r_T_D_inv_vct = sum(
                np.einsum("...jk,...j->k", dc_du_r_blocks[i], D_inv_vct_blocks[i])
                for i in range(3)
            )
            C_inv_dc_du_r_T_D_inv_vct = sla.lu_solve(lu_and_piv_C, dc_du_r_T_D_inv_vct)
            return np.concatenate(
                [
                    sla.lu_solve(
                        lu_and_piv_D_blocks[i],
                        vct_parts[i] - dc_du_l_blocks[i] @ C_inv_dc_du_r_T_D_inv_vct,
                    ).flatten()
                    for i in range(3)
                ]
            )

        @api.jit
        def normal_space_component(
            vct, dc_du_blocks, dc_dv_blocks, chol_C, chol_D_blocks
        ):
            """Compute component of vector in normal space to manifold at a point."""
            return rmult_by_jacob_constr(
                dc_du_blocks,
                dc_dv_blocks,
                lmult_by_inv_gram(
                    dc_du_blocks,
                    dc_dv_blocks,
                    chol_C,
                    chol_D_blocks,
                    lmult_by_jacob_constr(dc_du_blocks, dc_dv_blocks, vct),
                ),
            )

        def norm(x):
            """Infinity norm of a vector."""
            return np.max(np.abs(x))

        @api.partial(api.jit, static_argnums=(2, 8, 9, 10, 11))
        def quasi_newton_projection(
            q,
            x_obs_seq,
            partition,
            dc_du_prev_blocks,
            dc_dv_prev_blocks,
            chol_C_prev,
            chol_D_prev_blocks,
            dt,
            convergence_tol,
            position_tol,
            divergence_tol,
            max_iters,
        ):
            """Symmetric quasi-Newton method to solve projection onto manifold."""

            def body_func(val):
                q, mu, i, _, _ = val
                c = constr(q, x_obs_seq, partition)
                error = norm(c)
                delta_mu = rmult_by_jacob_constr(
                    dc_du_prev_blocks,
                    dc_dv_prev_blocks,
                    lmult_by_inv_gram(
                        dc_du_prev_blocks,
                        dc_dv_prev_blocks,
                        chol_C_prev,
                        chol_D_prev_blocks,
                        c,
                    ),
                )
                if not isinstance(metric, IdentityMatrix):
                    delta_q = np.concatenate(
                        [
                            metric.blocks[0].inv @ delta_mu[:dim_u],
                            metric.blocks[1].inv @ delta_mu[dim_u:],
                        ]
                    )
                else:
                    delta_q = delta_mu
                mu += delta_mu
                q -= delta_q
                i += 1
                return q, mu, i, norm(delta_q), error

            def cond_func(val):
                _, _, i, norm_delta_q, error, = val
                diverged = np.logical_or(error > divergence_tol, np.isnan(error))
                converged = np.logical_and(
                    error < convergence_tol, norm_delta_q < position_tol
                )
                return np.logical_not(
                    np.logical_or((i >= max_iters), np.logical_or(diverged, converged))
                )

            q, mu, i, norm_delta_q, error = lax.while_loop(
                cond_func, body_func, (q, np.zeros_like(q), 0, np.inf, -1.0)
            )
            if use_gaussian_splitting:
                return q, mu / np.sin(dt), i, norm_delta_q, error
            else:
                return q, mu / dt, i, norm_delta_q, error

        @api.partial(api.jit, static_argnums=(2, 6, 7, 8, 9))
        def newton_projection(
            q,
            x_obs_seq,
            partition,
            dc_du_prev_blocks,
            dc_dv_prev_blocks,
            dt,
            convergence_tol,
            position_tol,
            divergence_tol,
            max_iters,
        ):
            """Newton method to solve projection onto manifold."""

            def body_func(val):
                q, mu, i, _, _ = val
                c = constr(q, x_obs_seq, partition)
                dc_du_blocks, dc_dv_blocks = jacob_constr_blocks(
                    q, x_obs_seq, partition
                )
                lu_and_piv_C, lu_and_piv_D_blocks = lu_jacob_product_blocks(
                    dc_du_blocks, dc_dv_blocks, dc_du_prev_blocks, dc_dv_prev_blocks
                )
                error = norm(c)
                delta_mu = rmult_by_jacob_constr(
                    dc_du_prev_blocks,
                    dc_dv_prev_blocks,
                    lmult_by_inv_jacob_product(
                        dc_du_blocks,
                        dc_dv_blocks,
                        dc_du_prev_blocks,
                        dc_dv_prev_blocks,
                        lu_and_piv_C,
                        lu_and_piv_D_blocks,
                        c,
                    ),
                )
                if not isinstance(metric, IdentityMatrix):
                    delta_q = np.concatenate(
                        [
                            metric.blocks[0].inv @ delta_mu[:dim_u],
                            metric.blocks[1].inv @ delta_mu[dim_u:],
                        ]
                    )
                else:
                    delta_q = delta_mu
                mu += delta_mu
                q -= delta_q
                i += 1
                return q, mu, i, norm(delta_q), error

            def cond_func(val):
                _, _, i, norm_delta_q, error = val
                diverged = np.logical_or(error > divergence_tol, np.isnan(error))
                converged = np.logical_and(
                    error < convergence_tol, norm_delta_q < position_tol
                )
                return np.logical_not(
                    np.logical_or((i >= max_iters), np.logical_or(diverged, converged))
                )

            q, mu, i, norm_delta_q, error = lax.while_loop(
                cond_func, body_func, (q, np.zeros_like(q), 0, np.inf, -1.0)
            )
            if use_gaussian_splitting:
                return q, mu / np.sin(dt), i, norm_delta_q, error
            else:
                return q, mu / dt, i, norm_delta_q, error

        self._generate_x_obs_seq = generate_x_obs_seq
        self._constr = constr
        self._jacob_constr_blocks = jacob_constr_blocks
        self._chol_gram_blocks = chol_gram_blocks
        self._lu_jacob_product_blocks = lu_jacob_product_blocks
        self._log_det_sqrt_gram_from_chol = log_det_sqrt_gram_from_chol
        self._grad_log_det_sqrt_gram = api.jit(
            api.value_and_grad(log_det_sqrt_gram, has_aux=True), (2,)
        )
        self.value_and_grad_init_objective = api.jit(
            api.value_and_grad(init_objective, (0,), has_aux=True)
        )
        self._normal_space_component = normal_space_component
        self.quasi_newton_projection = quasi_newton_projection
        self.newton_projection = newton_projection

    @cache_in_state("pos", "x_obs_seq", "partition")
    def constr(self, state):
        return convert_to_numpy_pytree(
            self._constr(state.pos, state.x_obs_seq, state.partition)
        )

    @cache_in_state("pos", "x_obs_seq", "partition")
    def jacob_constr_blocks(self, state):
        return convert_to_numpy_pytree(
            self._jacob_constr_blocks(state.pos, state.x_obs_seq, state.partition)
        )

    @cache_in_state("pos", "x_obs_seq", "partition")
    def chol_gram_blocks(self, state):
        return convert_to_numpy_pytree(
            self._chol_gram_blocks(*self.jacob_constr_blocks(state))
        )

    @cache_in_state("pos", "x_obs_seq", "partition")
    def log_det_sqrt_gram(self, state):
        return float(self._log_det_sqrt_gram_from_chol(*self.chol_gram_blocks(state)))

    @cache_in_state_with_aux(
        ("pos", "x_obs_seq", "partition"),
        ("log_det_sqrt_gram", "jacob_constr_blocks", "chol_gram_blocks"),
    )
    def grad_log_det_sqrt_gram(self, state):
        (
            (val, (jacob_constr_blocks, chol_gram_blocks)),
            grad,
        ) = self._grad_log_det_sqrt_gram(state.pos, state.x_obs_seq, state.partition)
        return convert_to_numpy_pytree(
            (grad, float(val), jacob_constr_blocks, chol_gram_blocks)
        )

    def h1(self, state):
        if self.use_gaussian_splitting:
            return self.log_det_sqrt_gram(state)
        else:
            return self.neg_log_dens(state) + self.log_det_sqrt_gram(state)

    def dh1_dpos(self, state):
        if self.use_gaussian_splitting:
            return self.grad_log_det_sqrt_gram(state)
        else:
            return self.grad_neg_log_dens(state) + self.grad_log_det_sqrt_gram(state)

    def h2(self, state):
        if self.use_gaussian_splitting:
            return 0.5 * state.pos @ state.pos + 0.5 * state.mom @ state.mom
        else:
            return 0.5 * state.mom @ self.metric.inv @ state.mom

    def dh2_dmom(self, state):
        if self.use_gaussian_splitting:
            return state.mom
        else:
            return self.metric.inv @ state.mom

    def dh2_dpos(self, state):
        if self.use_gaussian_splitting:
            return state.pos
        else:
            return 0 * state.pos

    def dh_dpos(self, state):
        if self.use_gaussian_splitting:
            return self.dh1_dpos(state) + self.dh2_dpos(state)
        else:
            return self.dh1_dpos(state)

    def h2_flow(self, state, dt):
        if self.use_gaussian_splitting:
            sin_dt, cos_dt = np.sin(dt), np.cos(dt)
            pos = state.pos.copy()
            state.pos *= cos_dt
            state.pos += sin_dt * state.mom
            state.mom *= cos_dt
            state.mom -= sin_dt * pos
        else:
            state.pos += dt * self.dh2_dmom(state)

    def dh2_flow_dmom(self, dt):
        if self.use_gaussian_splitting:
            sin_dt, cos_dt = np.sin(dt), np.cos(dt)
            return sin_dt * IdentityMatrix(), cos_dt * IdentityMatrix()
        else:
            return (dt * self.metric.inv, IdentityMatrix())

    def update_x_obs_seq(self, state):
        state.x_obs_seq = convert_to_numpy_pytree(self._generate_x_obs_seq(state.pos))

    def normal_space_component(self, state, vct):
        return convert_to_numpy_pytree(
            self._normal_space_component(
                self.metric.inv @ vct,
                *self.jacob_constr_blocks(state),
                *self.chol_gram_blocks(state),
            )
        )

    def project_onto_cotangent_space(self, mom, state):
        mom -= self.normal_space_component(state, mom)
        return mom

    def sample_momentum(self, state, rng):
        mom = self.metric.sqrt @ rng.standard_normal(state.pos.shape)
        mom = self.project_onto_cotangent_space(mom, state)
        return mom


class SwitchPartitionTransition(Transition):
    """Markov transition that deterministically switches conditioned partition.

    The `partition` binary variable in the chain state, which sets the current
    partition used when conditioning on values of the diffusion process at
    intermediate time, is deterministically switched on each transition as well
    as updating the cached observed state sequence based on the current position
    state component.
    """

    def __init__(self, system):
        self.system = system

    state_variables = {"partition", "x_obs_seq"}
    statistic_types = None

    def sample(self, state, rng):
        state.partition = 0 if state.partition == 1 else 1
        self.system.update_x_obs_seq(state)
        return state, None


class ConditionedDiffusionHamiltonianState(ChainState):
    """Markov chain state for conditioned diffusion Hamiltonian system.

    In addition to the usual position, momentum and integration direction
    variables, the chain state is augmented with a partition indicator variable
    which defines the current partitioning of the observation sequence used
    for the partial conditioning, along with a record of the sequence of state
    values at the observation times `x_obs_seq` currently from which the
    current partial conditioning is defined.
    """

    def __init__(
        self,
        pos,
        x_obs_seq,
        partition=0,
        mom=None,
        dir=1,
        _call_counts=None,
        _dependencies=None,
        _cache=None,
        _read_only=False,
    ):
        if _call_counts is None:
            _call_counts = {}
        super().__init__(
            pos=pos,
            x_obs_seq=x_obs_seq,
            partition=partition,
            mom=mom,
            dir=dir,
            _call_counts=_call_counts,
            _dependencies=_dependencies,
            _cache=_cache,
            _read_only=_read_only,
        )


def jitted_solve_projection_onto_manifold_quasi_newton(
    state,
    state_prev,
    dt,
    system,
    convergence_tol=1e-8,
    position_tol=1e-8,
    divergence_tol=1e10,
    max_iters=50,
):
    """Symmetric quasi-Newton solver for projecting points onto manifold.

    Solves an equation of the form `r(λ) = c(q_ + M⁻¹ ∂c(q)ᵀλ) = 0` for the
    vector of Lagrange multipliers `λ` to project a point `q_` on to the
    manifold defined by the zero level set of `c`, with the projection performed
    with in the linear subspace defined by the rows of the Jacobian matrix
    `∂c(q)` evaluated at a previous point on the manifold `q`.

    The Jacobian of the residual function `r` is

        ∂r(λ) = ∂c(q_ + M⁻¹ ∂c(q)ᵀλ) M⁻¹ ∂c(q)ᵀ

    such that the full Newton update

        λ = λ - ∂r(λ)⁻¹ r(λ)

    requires evaluating `∂c` on each iteration. The symmetric quasi-Newton
    iteration instead uses the approximation

        ∂c(q_ + ∂c(q)ᵀλ) M⁻¹ ∂c(q)ᵀ ≈ ∂c(q) M⁻¹ ∂c(q)ᵀ

    with the corresponding update

        λ = λ - (∂c(q) M⁻¹ ∂c(q)ᵀ)⁻¹ r(λ)

    allowing a previously computed Cholesky decomposition of the Gram matrix
    `∂c(q) M⁻¹ ∂c(q)ᵀ` to be used to solve the linear system in each iteration
    with no requirement to evaluate `∂c` on each iteration.

    Compared to the inbuilt solver in `mici.solvers` this version exploits the
    structure in the constraint Jacobian `∂c` for conditioned diffusion systems
    and JIT compiles the iteration using JAX for better performance.
    """
    dc_du_prev, dc_dv_prev = system.jacob_constr_blocks(state_prev)
    chol_C_prev, chol_D_prev = system.chol_gram_blocks(state_prev)
    q, x_obs_seq, partition = state.pos, state.x_obs_seq, state.partition
    _, dh2_flow_mom_dmom = system.dh2_flow_dmom(dt)
    q_, mu, i, norm_delta_q, error = system.quasi_newton_projection(
        q,
        x_obs_seq,
        partition,
        dc_du_prev,
        dc_dv_prev,
        chol_C_prev,
        chol_D_prev,
        dt,
        convergence_tol,
        position_tol,
        divergence_tol,
        max_iters,
    )
    if state._call_counts is not None:
        key = _cache_key_func(system, system.constr)
        if key in state._call_counts:
            state._call_counts[key] += i
        else:
            state._call_counts[key] = i
    if error < convergence_tol and norm_delta_q < position_tol:
        state.pos = onp.array(q_)
        if state.mom is not None:
            state.mom -= dh2_flow_mom_dmom @ onp.asarray(mu)
        return state
    elif error > divergence_tol or np.isnan(error):
        raise ConvergenceError(
            f"Quasi-Newton iteration diverged on iteration {i}. "
            f"Last |c|={error:.1e}, |δq|={norm_delta_q}."
        )
    else:
        raise ConvergenceError(
            f"Quasi-Newton iteration did not converge. "
            f"Last |c|={error:.1e}, |δq|={norm_delta_q}."
        )


def jitted_solve_projection_onto_manifold_newton(
    state,
    state_prev,
    dt,
    system,
    convergence_tol=1e-8,
    position_tol=1e-8,
    divergence_tol=1e10,
    max_iters=50,
):
    """Newton solver for projecting points onto manifold.

    Solves an equation of the form `r(λ) = c(q_ + M⁻¹ ∂c(q)ᵀλ) = 0` for the
    vector of Lagrange multipliers `λ` to project a point `q_` on to the
    manifold defined by the zero level set of `c`, with the projection performed
    with in the linear subspace defined by the rows of the Jacobian matrix
    `∂c(q)` evaluated at a previous point on the manifold `q`.

    The Jacobian of the residual function `r` is

        ∂r(λ) = ∂c(q_ + M⁻¹ ∂c(q)ᵀλ) M⁻¹ ∂c(q)ᵀ

    such that the Newton update is

        λ = λ - ∂r(λ)⁻¹ r(λ)

    which requires evaluating `∂c` on each iteration.

    Compared to the inbuilt solver in `mici.solvers` this version exploits the
    structure in the constraint Jacobian `∂c` for conditioned diffusion systems
    and JIT compiles the iteration using JAX for better performance.
    """
    dc_du_prev, dc_dv_prev = system.jacob_constr_blocks(state_prev)
    q, x_obs_seq, partition = state.pos, state.x_obs_seq, state.partition
    _, dh2_flow_mom_dmom = system.dh2_flow_dmom(dt)
    q_, mu, i, norm_delta_q, error = system.newton_projection(
        q,
        x_obs_seq,
        partition,
        dc_du_prev,
        dc_dv_prev,
        dt,
        convergence_tol,
        position_tol,
        divergence_tol,
        max_iters,
    )
    if state._call_counts is not None:
        key = _cache_key_func(system, system.constr)
        if key in state._call_counts:
            state._call_counts[key] += i
        else:
            state._call_counts[key] = i
    if error < convergence_tol and norm_delta_q < position_tol:
        state.pos = onp.array(q_)
        if state.mom is not None:
            state.mom -= dh2_flow_mom_dmom @ onp.asarray(mu)
        return state
    elif error > divergence_tol or np.isnan(error):
        raise ConvergenceError(
            f"Newton iteration diverged on iteration {i}. "
            f"Last |c|={error:.1e}, |δq|={norm_delta_q}."
        )
    else:
        raise ConvergenceError(
            f"Newton iteration did not converge. "
            f"Last |c|={error:.1e}, |δq|={norm_delta_q}."
        )


def find_initial_state_by_linear_interpolation(
    system, rng, generate_x_obs_seq_init,
):
    """Find an initial constraint satisying state linearly interpolating noise sequence.

    Samples the parameters `z` and initial diffusion state `x_0` from their prior
    distributions and a sequence of diffusion states at the observation time indices
    `x_obs_seq` consistent with the observed sequence `y` (i.e. such that
    `y = obs_func(x_obs_seq)`) and then  solves for the sequence of noise vectors
    `v_seq` which map to a state sequence which linear interpolates between the states
    in `x_obs_seq`. It is assumed `forward_func` is linear in the noise vector argument
    `v` and that the Jacobian of `forward_func` wrt `v` is full row-rank.
    """

    def mean_and_sqrt_covar_step_diff(z, x, δ):
        v = np.zeros(system.dim_v)

        def step_diff_func(v):
            return system.forward_func(z, x, v, δ) - x

        return step_diff_func(v), api.jacobian(step_diff_func)(v)

    @api.jit
    def solve_for_v_seq(x_obs_seq, x_0, z):
        num_obs = x_obs_seq.shape[0]

        def solve_inner(x, Δx):
            mean_diff, sqrt_covar_diff = mean_and_sqrt_covar_step_diff(z, x, system.δ)
            return np.linalg.lstsq(sqrt_covar_diff, (Δx - mean_diff))[0]

        def solve_outer(x_0, x_1):
            Δx = (x_1 - x_0) / system.num_steps_per_obs
            x_seq = x_0[None] + np.arange(system.num_steps_per_obs)[:, None] * Δx[None]
            return api.vmap(solve_inner, (0, None))(x_seq, Δx)

        x_0_seq = np.concatenate((x_0[None], x_obs_seq[:-1]))
        x_1_seq = x_obs_seq

        return api.vmap(solve_outer)(x_0_seq, x_1_seq).reshape(
            (num_obs * system.num_steps_per_obs, system.dim_v)
        )

    u = rng.standard_normal(system.dim_u)
    z = system.generate_z(u)
    v_0 = rng.standard_normal(system.dim_v_0)
    x_0 = system.generate_x_0(z, v_0)
    x_obs_seq = generate_x_obs_seq_init(rng)
    v_seq = solve_for_v_seq(x_obs_seq, x_0, z)
    q = onp.concatenate([u, v_0, v_seq.flatten()])
    state = ConditionedDiffusionHamiltonianState(pos=q, x_obs_seq=x_obs_seq)
    assert onp.allclose(system.constr(state), 0)
    state.mom = system.sample_momentum(state, rng)
    return state


def find_initial_state_by_gradient_descent(
    system,
    rng,
    generate_x_obs_seq_init,
    tol=1e-9,
    adam_step_size=2e-1,
    reg_coeff=5e-2,
    coarse_tol=1e-1,
    max_iters=1000,
    max_num_tries=10,
    use_newton=True,
):
    """Find an initial constraint satisying state by a gradient descent based scheme.

    Uses a heuristic combination of gradient-based minimisation of the norm of a
    modified constraint function plus a subsequent projection step using a
    (quasi-)Newton method, to try to find an initial point `q` such that
    `max(abs(constr(q)) < tol`.
    """

    # Use optimizers to set optimizer initialization and update functions
    opt_init, opt_update, get_params = opt.adam(adam_step_size)

    # Define a compiled update step
    @api.jit
    def step(i, opt_state, x_obs_seq_init):
        (q,) = get_params(opt_state)
        (obj, constr), grad = system.value_and_grad_init_objective(
            q, x_obs_seq_init, reg_coeff
        )
        opt_state = opt_update(i, grad, opt_state)
        return opt_state, obj, constr

    projection_solver = (
        jitted_solve_projection_onto_manifold_newton
        if use_newton
        else jitted_solve_projection_onto_manifold_quasi_newton
    )

    for t in range(max_num_tries):
        logging.info(f"Starting try {t+1}")
        q_init = rng.standard_normal(system.dim_q)
        x_obs_seq_init = generate_x_obs_seq_init(rng)
        opt_state = opt_init((q_init,))
        for i in range(max_iters):
            opt_state_next, norm, constr = step(i, opt_state, x_obs_seq_init)
            if not np.isfinite(norm):
                logger.info("Adam iteration diverged")
                break
            max_abs_constr = maximum_norm(constr)
            if max_abs_constr < coarse_tol:
                logging.info("Within coarse_tol attempting projection.")
                (q_init,) = get_params(opt_state)
                state = ConditionedDiffusionHamiltonianState(
                    q_init, x_obs_seq=x_obs_seq_init, _call_counts={}
                )
                try:
                    state = projection_solver(state, state, 1.0, system, tol)
                except ConvergenceError:
                    logger.info("Quasi-Newton iteration diverged.")
                if np.max(np.abs(system.constr(state))) < tol:
                    logging.info("Found constraint satisfying state.")
                    state.mom = system.sample_momentum(state, rng)
                    return state
            if i % 100 == 0:
                logging.info(
                    f"Iteration {i: >6}: mean|constr|^2 = {norm:.3e} "
                    f"max|constr| = {max_abs_constr:.3e}"
                )
            opt_state = opt_state_next
    raise RuntimeError(f"Did not find valid state in {max_num_tries} tries.")
