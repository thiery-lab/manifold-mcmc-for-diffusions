import logging
import numpy as onp
import jax.numpy as np
import jax.numpy.linalg as nla
import jax.scipy.linalg as sla
from jax import lax, api
import jax.experimental.optimizers as opt
from mici.systems import (
    EuclideanMetricSystem, cache_in_state, multi_cache_in_state)
from mici.matrices import (
    SymmetricBlockDiagonalMatrix, IdentityMatrix, PositiveDiagonalMatrix)
from mici.transitions import Transition
from mici.states import ChainState
from mici.solvers import maximum_norm
from mici.errors import ConvergenceError


logger = logging.getLogger(__name__)


def split(v, lengths):
    """Split an array along first dimension into slices of specified lengths."""
    i = 0
    parts = []
    for l in lengths:
        parts.append(v[i:i+l])
        i += l
    if i < len(v):
        parts.append(v[i:])
    return parts


def standard_normal_neg_log_dens(q):
    """Unnormalised negative log density of standard normal vector."""
    return 0.5 * onp.sum(q**2)


def standard_normal_grad_neg_log_dens(q):
    """Gradient and value of negative log density of standard normal vector."""
    return q, 0.5 * onp.sum(q**2)


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
        raise ValueError(f'Unknown jax_pytree node type {type(jax_pytree)}')
    return wrapper


class ConditionedDiffusionConstrainedSystem(EuclideanMetricSystem):
    """Specialised mici system class for conditioned diffusion problems."""

    def __init__(self, obs_interval, num_steps_per_obs, num_obs_per_subseq,
                 y_seq, dim_z, dim_x, dim_v, forward_func, generate_x_0,
                 generate_z, obs_func, metric=None):
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
            dim_z(int): Dimension of parameter vector `z`.
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
            metric (Matrix): Metric matrix representation. Should be either an
                `mici.matrices.IdentityMatrix` or
                `mici.matrices.SymmetricBlockDiagonalMatrix` instance, with in
                the latter case the matrix having two blocks on the diagonal,
                the left most of size `dim_z x dim_z`, and the rightmost being
                positive diagonal. Defaults to `mici.matrices.IdentityMatrix`.
        """

        if metric is None or isinstance(metric, IdentityMatrix):
            metric_1 = np.eye(dim_z)
            log_det_sqrt_metric_1 = 0
        elif (isinstance(metric, SymmetricBlockDiagonalMatrix) and
                isinstance(metric.blocks[1], PositiveDiagonalMatrix)):
            metric_1 = metric.blocks[0].array
            log_det_sqrt_metric_1 = metric.blocks[0].log_abs_det_sqrt
            metric_2_diag = metric.blocks[1].diagonal
        else:
            raise NotImplementedError(
                'Only identity and block diagonal metrics with diagonal lower '
                'right block currently supported.')

        num_obs, dim_y = y_seq.shape
        δ = obs_interval / num_steps_per_obs
        dim_q = dim_z + dim_x + num_obs * dim_v * num_steps_per_obs
        if num_obs % num_obs_per_subseq != 0:
            raise NotImplementedError(
                 'Only cases where num_obs_per_subseq is a factor of num_obs '
                 'supported.')
        num_subseq = num_obs // num_obs_per_subseq
        obs_indices = slice(num_steps_per_obs - 1, None, num_steps_per_obs)
        num_step_per_subseq = num_obs_per_subseq * num_steps_per_obs
        y_subseqs_p0 = np.reshape(y_seq, (num_subseq, num_obs_per_subseq, -1))
        y_subseqs_p1 = split(
            y_seq, (num_obs_per_subseq // 2, num_obs - num_obs_per_subseq))
        y_subseqs_p1[1] = np.reshape(
            y_subseqs_p1[1], (num_subseq - 1, num_obs_per_subseq, dim_y))

        super().__init__(
            neg_log_dens=standard_normal_neg_log_dens,
            grad_neg_log_dens=standard_normal_grad_neg_log_dens, metric=metric)

        @api.jit
        def step_func(z, x, v):
            x_n = forward_func(z, x, v, δ)
            return (x_n, x_n)

        @api.jit
        def generate_x_obs_seq(q):
            u, v_0, v_seq_flat = split(q, (dim_z, dim_x))
            z = generate_z(u)
            x_0 = generate_x_0(z, v_0)
            v_seq = np.reshape(v_seq_flat, (-1, dim_v))
            _, x_seq = lax.scan(lambda x, v: step_func(z, x, v), x_0, v_seq)
            return x_seq[obs_indices]

        @api.partial(api.jit, static_argnums=(3,))
        def partition_into_subseqs(v_seq, v_0, x_obs_seq, partition=0):
            """Partition noise increment and observation sequences.

            Partitition sequences in to either `num_subseq` equally sized
            subsequences (`partition == 0`)  or `num_subseq - 1` equally sized
            subsequences plus initial and final 'half' subsequences.
            """
            if partition == 0:
                v_subseqs = v_seq.reshape(
                    (num_subseq, num_step_per_subseq, dim_v))
                v_subseqs = (v_subseqs[0], v_subseqs[1:-1], v_subseqs[-1])
                x_obs_subseqs = x_obs_seq.reshape(
                    (num_subseq, num_obs_per_subseq, dim_x))
                w_inits = (v_0, x_obs_subseqs[:-2, -1], x_obs_subseqs[-2, -1])
                y_bars = (
                    np.concatenate(
                        (y_subseqs_p0[0, :-1].flatten(), x_obs_subseqs[0, -1])),
                    np.concatenate(
                        (y_subseqs_p0[1:-1, :-1].reshape((num_subseq - 2, -1)),
                         x_obs_subseqs[1:-1, -1]), -1),
                    y_subseqs_p0[-1].flatten()
                )
            else:
                v_subseqs = split(
                    v_seq, ((num_obs_per_subseq // 2) * num_steps_per_obs,
                            num_step_per_subseq * (num_subseq - 1)))
                v_subseqs[1] = v_subseqs[1].reshape(
                    (num_subseq - 1, num_step_per_subseq, dim_v))
                x_obs_subseqs = split(x_obs_seq, (num_obs_per_subseq // 2,
                                                  num_obs - num_obs_per_subseq))
                x_obs_subseqs[1] = x_obs_subseqs[1].reshape(
                    (num_subseq - 1, num_obs_per_subseq, dim_x))
                w_inits = (
                    v_0,
                    np.concatenate((
                        x_obs_subseqs[0][-1:], x_obs_subseqs[1][:-1, -1]), 0),
                    x_obs_subseqs[1][-1, -1]
                )
                y_bars = (
                    np.concatenate(
                        (y_subseqs_p1[0][:-1].flatten(), x_obs_subseqs[0][-1])),
                    np.concatenate((
                        y_subseqs_p1[1][:, :-1].reshape((num_subseq - 1, -1)),
                        x_obs_subseqs[1][:, -1],
                    ), -1),
                    y_subseqs_p1[2].flatten()
                )
            return v_subseqs, w_inits, y_bars

        def generate_y_bar(z, w_0, v_seq, b):
            x_0 = generate_x_0(z, w_0) if b == 0 else w_0
            _, x_seq = lax.scan(lambda x, v: step_func(z, x, v), x_0, v_seq)
            y_seq = obs_func(x_seq[obs_indices])
            return y_seq.flatten() if b == 2 else np.concatenate(
                (y_seq[:-1].flatten(), x_seq[-1]))

        @api.partial(api.jit, static_argnums=(2,))
        def constr(q, x_obs_seq, partition=0):
            """Calculate constraint function for current partition."""
            u, v_0, v_seq_flat = split(q, (dim_z, dim_x,))
            v_seq = v_seq_flat.reshape((-1, dim_v))
            z = generate_z(u)
            (v_subseqs, w_inits, y_bars) = partition_into_subseqs(
                 v_seq, v_0, x_obs_seq, partition)
            gen_funcs = (
                generate_y_bar,
                api.vmap(generate_y_bar, (None, 0, 0, None)),
                generate_y_bar
            )
            return np.concatenate([
                (gen_funcs[b](z, w_inits[b], v_subseqs[b], b) -
                 y_bars[b]).flatten()
                for b in range(3)
            ])

        @api.jit
        def init_objective(q, x_obs_seq, reg_coeff):
            """Optimisation objective to find initial state on manifold."""
            u, v_0, v_seq_flat = split(q, (dim_z, dim_x,))
            v_subseqs = v_seq_flat.reshape((num_obs, num_steps_per_obs, dim_v))
            z = generate_z(u)
            x_0 = generate_x_0(z, v_0)
            x_inits = np.concatenate((x_0[None], x_obs_seq[:-1]), 0)

            def generate_final_state(z, v_seq, x_0):
                _, x_seq = lax.scan(lambda x, v: step_func(z, x, v), x_0, v_seq)
                return x_seq[-1]

            c = api.vmap(generate_final_state, in_axes=(None, 0, 0))(
                z, v_subseqs, x_inits) - x_obs_seq
            return 0.5 * np.mean(c**2) + 0.5 * reg_coeff * np.mean(q**2), c

        @api.partial(api.jit, static_argnums=(2,))
        def jacob_constr_blocks(q, x_obs_seq, partition=0):
            """Return non-zero blocks of constraint function Jacobian.

            Input state q can be decomposed into q = [u, v₀, v₁, v₂]
            where global latent state (parameters) are determined by u,
            initial subsequence by v₀, middle subsequences by v₁ and final
            subsequence by v₂.

            Constraint function can then be decomposed as

                c(q) = [c₀(u, v₀), c₁(u, v₁), c₂(u, v₂)]

            Constraint Jacobian ∂c(q) has block structure

                ∂c(q) = [[∂₀c₀(u, v₀), ∂₁c₀(u, v₀),     0,     ,     0      ]
                         [∂₀c₁(u, v₁),     0      , ∂₁c₁(u, v₁),     0      ]
                         [∂₀c₂(u, v₀),     0      ,     0      , ∂₁c₂(u, v₂)]]

            """

            def g_y_bar(u, v, w_0, b):
                z = generate_z(u)
                if b == 0:
                    w_0, v = split(v, (dim_x,))
                v_seq = np.reshape(v, (-1, dim_v))
                return generate_y_bar(z, w_0, v_seq, b)

            u, v_0, v_seq_flat = split(q, (dim_z, dim_x,))
            v_seq = np.reshape(v_seq_flat, (-1, dim_v))
            (v_subseqs, w_inits, y_bars) = partition_into_subseqs(
                 v_seq, v_0, x_obs_seq, partition)
            v_bars = (
                np.concatenate([v_0, v_subseqs[0].flatten()]),
                np.reshape(v_subseqs[1], (v_subseqs[1].shape[0], -1)),
                v_subseqs[2].flatten()
            )
            jac_g_y_bar = api.jacrev(g_y_bar, (0, 1))
            jacob_funcs = (
                jac_g_y_bar,
                api.vmap(jac_g_y_bar, (None, 0, 0, None)),
                jac_g_y_bar
            )
            return tuple(zip(*[
                jacob_funcs[b](u, v_bars[b], w_inits[b], b) for b in range(3)]))

        @api.jit
        def chol_gram_blocks(dc_du, dc_dv):
            """Calculate Cholesky factors of decomposition of Gram matrix. """
            if isinstance(metric, IdentityMatrix):
                D = tuple(np.einsum('...ij,...kj', dc_dv[i], dc_dv[i])
                          for i in range(3))
            else:
                m_v = split(
                    metric_2_diag, (dc_dv[0].shape[1],
                                    dc_dv[1].shape[0] * dc_dv[1].shape[2]))
                m_v[1] = m_v[1].reshape((dc_dv[1].shape[0], dc_dv[1].shape[2]))
                D = tuple(np.einsum('...ij,...kj',
                                    dc_dv[i] / m_v[i][..., None, :], dc_dv[i])
                          for i in range(3))
            chol_D = tuple(nla.cholesky(D[i]) for i in range(3))
            D_inv_dc_du = tuple(
                sla.cho_solve((chol_D[i], True), dc_du[i]) for i in range(3))
            chol_C = nla.cholesky(
                metric_1 +
                (dc_du[0].T @ D_inv_dc_du[0] +
                 np.einsum('ijk,ijl->kl', dc_du[1], D_inv_dc_du[1]) +
                 dc_du[2].T @ D_inv_dc_du[2]))
            return chol_C, chol_D

        @api.jit
        def log_det_sqrt_gram_from_chol(chol_C, chol_D):
            """Calculate log-det of Gram matrix from Cholesky factors."""
            return (
                sum(np.log(np.abs(chol_D[i].diagonal(0, -2, -1))).sum()
                    for i in range(3)) +
                np.log(np.abs(chol_C.diagonal())).sum() - log_det_sqrt_metric_1
            )

        @api.partial(api.jit, static_argnums=(2,))
        def log_det_sqrt_gram(q, x_obs_seq, partition=0):
            """Calculate log-determinant of constraint Jacobian Gram matrix."""
            dc_du, dc_dv = jacob_constr_blocks(q, x_obs_seq, partition)
            chol_C, chol_D = chol_gram_blocks(dc_du, dc_dv)
            return (
                log_det_sqrt_gram_from_chol(chol_C, chol_D),
                ((dc_du, dc_dv), (chol_C, chol_D)))

        @api.jit
        def lmult_by_jacob_constr(dc_du, dc_dv, vct):
            """Left-multiply vector by constraint Jacobian matrix."""
            vct_u, vct_v = split(vct, (dim_z,))
            j0, j1, j2 = dc_dv[0].shape[1], dc_dv[1].shape[0], dc_dv[2].shape[1]
            return (
                np.vstack((dc_du[0], dc_du[1].reshape((-1, dim_z)), dc_du[2])) @
                vct_u +
                np.concatenate((
                    dc_dv[0] @ vct_v[:j0],
                    np.einsum('ijk,ik->ij', dc_dv[1],
                              np.reshape(vct_v[j0:-j2], (j1, -1))).flatten(),
                    dc_dv[2] @ vct_v[-j2:]
                ))
            )

        @api.jit
        def rmult_by_jacob_constr(dc_du, dc_dv, vct):
            """Right-multiply vector by constraint Jacobian matrix."""
            vct_parts = split(
                vct, (dc_du[0].shape[0], dc_du[1].shape[0] * dc_du[1].shape[1]))
            vct_parts[1] = np.reshape(vct_parts[1], dc_du[1].shape[:2])
            return np.concatenate([
                vct_parts[0] @ dc_du[0] +
                np.einsum('ij,ijk->k', vct_parts[1], dc_du[1]) +
                vct_parts[2] @ dc_du[2],
                vct_parts[0] @ dc_dv[0],
                np.einsum('ij,ijk->ik', vct_parts[1], dc_dv[1]).flatten(),
                vct_parts[2] @ dc_dv[2]
            ])

        @api.jit
        def lmult_by_inv_gram(dc_du, dc_dv, chol_C, chol_D, vct):
            """Left-multiply vector by inverse Gram matrix."""
            vct_parts = split(
                vct, (dc_du[0].shape[0], dc_du[1].shape[0] * dc_du[1].shape[1]))
            vct_parts[1] = np.reshape(vct_parts[1], dc_du[1].shape[:2])
            D_inv_vct = [sla.cho_solve((chol_D[i], True), vct_parts[i])
                         for i in range(3)]
            dc_du_T_D_inv_vct = sum(
                np.einsum('...jk,...j->k', dc_du[i], D_inv_vct[i])
                for i in range(3))
            C_inv_dc_du_T_D_inv_vct = sla.cho_solve(
                (chol_C, True), dc_du_T_D_inv_vct)
            return np.concatenate([
                sla.cho_solve(
                    (chol_D[i], True),
                    vct_parts[i] - dc_du[i] @ C_inv_dc_du_T_D_inv_vct).flatten()
                for i in range(3)
            ])

        @api.jit
        def normal_space_component(vct, dc_du, dc_dv, chol_C, chol_D):
            return rmult_by_jacob_constr(
                dc_du, dc_dv, lmult_by_inv_gram(
                    dc_du, dc_dv, chol_C, chol_D, lmult_by_jacob_constr(
                        dc_du, dc_dv, vct)))

        @api.partial(api.jit, static_argnums=(2, 7, 8, 9, 10))
        def quasi_newton_projection(
                q, x_obs_seq, partition, dc_du_prev, dc_dv_prev, chol_C_prev,
                chol_D_prev, convergence_tol, position_tol, divergence_tol,
                max_iters):

            norm = lambda x: np.max(np.abs(x))

            def body_func(val):
                q, i, _, _ = val
                c = constr(q, x_obs_seq, partition)
                error = norm(c)
                delta_q = rmult_by_jacob_constr(
                    dc_du_prev, dc_dv_prev, lmult_by_inv_gram(
                        dc_du_prev, dc_dv_prev, chol_C_prev, chol_D_prev, c))
                q -= delta_q
                i += 1
                return q, i, norm(delta_q), error

            def cond_func(val):
                q, i, norm_delta_q, error, = val
                diverged = np.logical_or(
                    error > divergence_tol, np.isnan(error))
                converged = np.logical_and(
                    error < convergence_tol, norm_delta_q < position_tol)
                return np.logical_not(np.logical_or(
                    (i >= max_iters), np.logical_or(diverged, converged)))

            return lax.while_loop(cond_func, body_func, (q, 0, np.inf, -1.))

        self._generate_x_obs_seq = generate_x_obs_seq
        self._constr = constr
        self._jacob_constr_blocks = jacob_constr_blocks
        self._chol_gram_blocks = chol_gram_blocks
        self._log_det_sqrt_gram_from_chol = log_det_sqrt_gram_from_chol
        self._grad_log_det_sqrt_gram = api.jit(
            api.value_and_grad(log_det_sqrt_gram, has_aux=True), (2,))
        self.value_and_grad_init_objective = api.jit(
            api.value_and_grad(init_objective, (0,), has_aux=True))
        self._normal_space_component = normal_space_component
        self.quasi_newton_projection = quasi_newton_projection

    @cache_in_state('pos', 'x_obs_seq', 'partition')
    def constr(self, state):
        return convert_to_numpy_pytree(
            self._constr(state.pos, state.x_obs_seq, state.partition))

    @cache_in_state('pos', 'x_obs_seq', 'partition')
    def jacob_constr_blocks(self, state):
        return convert_to_numpy_pytree(self._jacob_constr_blocks(
            state.pos, state.x_obs_seq, state.partition))

    @cache_in_state('pos', 'x_obs_seq', 'partition')
    def chol_gram_blocks(self, state):
        return convert_to_numpy_pytree(
            self._chol_gram_blocks(*self.jacob_constr_blocks(state)))

    @cache_in_state('pos', 'x_obs_seq', 'partition')
    def log_det_sqrt_gram(self, state):
        return float(
            self._log_det_sqrt_gram_from_chol(*self.chol_gram_blocks(state)))

    @multi_cache_in_state(['pos', 'x_obs_seq', 'partition'],
                          ['grad_log_det_sqrt_gram', 'log_det_sqrt_gram',
                           'jacob_constr_blocks', 'chol_gram_blocks'])
    def grad_log_det_sqrt_gram(self, state):
        (val, (jacob_constr_blocks, chol_gram_blocks)), grad = (
           self._grad_log_det_sqrt_gram(
             state.pos, state.x_obs_seq, state.partition))
        return convert_to_numpy_pytree((
            grad, float(val), jacob_constr_blocks, chol_gram_blocks))

    def h1(self, state):
        return self.neg_log_dens(state) + self.log_det_sqrt_gram(state)

    def dh1_dpos(self, state):
        return (
            self.grad_neg_log_dens(state) + self.grad_log_det_sqrt_gram(state))

    def update_x_obs_seq(self, state):
        state.x_obs_seq = convert_to_numpy_pytree(
            self._generate_x_obs_seq(state.pos))

    def normal_space_component(self, state, vct):
        return convert_to_numpy_pytree(self._normal_space_component(
            self.metric.inv @ vct, *self.jacob_constr_blocks(state),
            *self.chol_gram_blocks(state)))

    def project_onto_cotangent_space(self, mom, state):
        mom -= self.normal_space_component(state, mom)
        return mom

    def sample_momentum(self, state, rng):
        mom = super().sample_momentum(state, rng)
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

    state_variables = {'partition', 'x_obs_seq'}
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

    def __init__(self, pos, x_obs_seq, partition=0, mom=None, dir=1,
                 _call_counts=None, _dependencies=None, _cache=None,
                 _read_only=False):
        if _call_counts is None:
            _call_counts = {}
        super().__init__(
            pos=pos, x_obs_seq=x_obs_seq, partition=partition,
            mom=mom, dir=dir, _call_counts=_call_counts,
            _dependencies=_dependencies, _cache=_cache, _read_only=_read_only)


def jitted_solve_projection_onto_manifold_quasi_newton(
        state, state_prev, dt, system,
        convergence_tol=1e-8, position_tol=1e-8,
        divergence_tol=1e10, max_iters=50, norm=maximum_norm):
    """Quasi-Newton iterative solver for projecting points onto manifold.

    Solves an equation of the form `c(q_ + ∂c(q)ᵀλ) = 0` for the vector of
    Lagrange multipliers `λ` to project a point `q_` on to the manifold defined
    by the zero level set of `c`, with the projection performed with in the
    linear subspace defined by the rows of the Jacobian matrix `∂c(q)` evaluated
    at a previous point on the manifold `q`.

    Compared to the inbuilt solver in `mici.solvers` this version exploits the
    structure in constraint Jacobian `∂c(q)` for conditioned diffusion systems
    and JIT compiles the iteration using JAX for better performance.
    """
    dc_du_prev, dc_dv_prev = system.jacob_constr_blocks(state_prev)
    chol_C_prev, chol_D_prev = system.chol_gram_blocks(state_prev)
    q, x_obs_seq, partition = state.pos, state.x_obs_seq, state.partition
    q_, i, norm_delta_q, error = system.quasi_newton_projection(
        q, x_obs_seq, partition, dc_du_prev, dc_dv_prev, chol_C_prev,
        chol_D_prev, convergence_tol, position_tol, divergence_tol, max_iters)
    if error < convergence_tol and norm_delta_q < position_tol:
        state.pos = convert_to_numpy_pytree(q_)
        state.mom = system.metric @ (state.pos - state_prev.pos) / dt
        return state
    elif error > divergence_tol or np.isnan(error):
        raise ConvergenceError(
            f'Quasi-Newton iteration diverged on iteration {i}. '
            f'Last |c|={error:.1e}, |δq|={norm_delta_q}.')
    else:
        raise ConvergenceError(
            f'Quasi-Newton iteration did not converge. '
            f'Last |c|={error:.1e}, |δq|={norm_delta_q}.')


def get_initial_state(system, rng, generate_x_obs_seq_init, dim_q, tol,
                      adam_step_size=2e-1, reg_coeff=5e-2, coarse_tol=1e-1,
                      max_iters=1000, max_num_tries=10):
    """Find an initial constraint satisying state.

    Uses a heuristic combination of gradient-based minimisation of the norm
    of a modified constraint function plus a subsequent projection step using a
    quasi-Newton method, to try to find an initial point `q` such that
    `max(abs(constr(q)) < tol`.
    """

    # Use optimizers to set optimizer initialization and update functions
    opt_init, opt_update, get_params = opt.adam(adam_step_size)

    # Define a compiled update step
    @api.jit
    def step(i, opt_state, x_obs_seq_init):
        q, = get_params(opt_state)
        (obj, constr), grad = system.value_and_grad_init_objective(
            q, x_obs_seq_init, reg_coeff)
        opt_state = opt_update(i, grad, opt_state)
        return opt_state, obj, constr

    for t in range(max_num_tries):
        logging.info(f'Starting try {t+1}')
        q_init = rng.standard_normal(dim_q)
        x_obs_seq_init = generate_x_obs_seq_init(rng)
        opt_state = opt_init((q_init,))
        for i in range(max_iters):
            opt_state_next, norm, constr = step(i, opt_state, x_obs_seq_init)
            if not np.isfinite(norm):
                logger.info('Adam iteration diverged')
                break
            max_abs_constr = maximum_norm(constr)
            if max_abs_constr < coarse_tol:
                logging.info('Within coarse_tol attempting projection.')
                q_init, = get_params(opt_state)
                state = ConditionedDiffusionHamiltonianState(
                    q_init, x_obs_seq=x_obs_seq_init)
                try:
                    state = jitted_solve_projection_onto_manifold_quasi_newton(
                        state, state, 1., system, tol)
                except ConvergenceError:
                    logger.info('Quasi-Newton iteration diverged.')
                if np.max(np.abs(system.constr(state))) < tol:
                    logging.info('Found constraint satisfying state.')
                    state.mom = system.sample_momentum(state, rng)
                    return state
            if i % 100 == 0:
                logging.info(f'Iteration {i: >6}: mean|constr|^2 = {norm:.3e} '
                             f'max|constr| = {max_abs_constr:.3e}')
            opt_state = opt_state_next
    raise RuntimeError(f'Did not find valid state in {max_num_tries} tries.')
