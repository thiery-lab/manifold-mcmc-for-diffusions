import logging
from numbers import Number
from functools import partial
import numpy as onp
import jax
import jax.numpy as np
import jax.numpy.linalg as nla
import jax.scipy.linalg as sla
from jax import lax
import jax.experimental.optimizers as opt
from mici.systems import (
    System,
    cache_in_state,
    cache_in_state_with_aux,
)
from mici.matrices import (
    DensePositiveDefiniteMatrix,
    PositiveDefiniteBlockDiagonalMatrix,
    IdentityMatrix,
)
from mici.transitions import Transition
from mici.adapters import Adapter, AdaptationError
from mici.states import ChainState, _cache_key_func
from mici.solvers import maximum_norm
from mici.errors import ConvergenceError, HamiltonianDivergenceError


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


def split_and_reshape(array, shapes):
    """Split an array along first dimension into subarrays of specified shapes."""
    i = 0
    parts = []
    for s in shapes:
        j = onp.product(s)
        parts.append(array[i : i + j].reshape(s + array.shape[1:]))
        i += j
    if i < array.shape[0]:
        parts.append(array[i:])
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
    elif isinstance(jax_pytree, (float, int, complex, bool, type(None))):
        return jax_pytree
    elif isinstance(jax_pytree, tuple):
        return tuple(convert_to_numpy_pytree(subtree) for subtree in jax_pytree)
    elif isinstance(jax_pytree, list):
        return [convert_to_numpy_pytree(subtree) for subtree in jax_pytree]
    elif isinstance(jax_pytree, dict):
        return {k: convert_to_numpy_pytree(v) for k, v in jax_pytree.items()}
    else:
        raise ValueError(f"Unknown jax_pytree node type {type(jax_pytree)}")


def conditioned_diffusion_neg_log_dens_and_grad(
    obs_interval,
    num_steps_per_obs,
    y_seq,
    dim_u,
    dim_v_0,
    dim_v,
    forward_func,
    generate_x_0,
    generate_z,
    generate_σ,
    obs_func,
    use_gaussian_splitting=False,
    return_jax_funcs=False,
):
    """Construct negative log target density + gradient functions for diffusion model.

    Constructs functions evaluating the negative logarithm of the unnormalised posterior
    density with respect to the Lebesgue measure for a partially observed diffusion
    model with additive Gaussian noise in the observations as well as a function to
    evaluate the gradient of this negative log density. These functions can be used
    with a Mici `EuclideanMetricSystem` system class to allow using HMC to perform
    inference in the model.

    A non-centred parameterisation of the model is used with the latent state `q`
    assumed to have a prior distribution specified by a product of independent standard
    normal factors, with `q` being the concatenation of a vector `u` mapping to the
    model parameters `z` and observation noise standard deviation `σ` via generator
    functions `generate_z` and `generate_σ` respectively, a vector `v_0` mapping to
    the initial diffusion state `x_0` via a generator function `generate_x_0` and a
    vector `v_seq_flat` which corresponds to a concatenatation of the standard normal
    vectors used to simulate the Wiener noise increments on each integrator step.

    Args:
        obs_interval (float): Interobservation time interval.
        num_steps_per_obs (int): Number of discrete time steps to simulate between each
            observation time.
        y_seq (array): Two-dimensional array containing observations at equally spaced
            time intervals, with first axis of array corresponding to observation time
            index (in order of increasing time) and second axis corresponding to
            dimension of each (vector-valued) observation.
        dim_u (int): Dimension of vector `u` mapping to parameter vector `z`.
        dim_v_0 (int): Dimension of vector used to generate initial state `x_0`.
        dim_v (int): Dimension of noise vector `v` consumed by `forward_func` to
            approximate time step.
        forward_func (Callable[[array, array, array, float], array]): Function
            implementing forward step of time-discretisation of diffusion such that
            `forward_func(z, x, v, δ)` for parameter vector `z`, current state `x` at
            time `t`, standard normal vector `v` and  small timestep `δ` and is
            distributed approximately according to `X(t + δ) | X(t) = x, Z = z`.
        generate_x_0 (Callable[[array, array], array]): Generator function for the
            initial state such that `generator_x_0(z, v_0)` for parameter vector `z` and
            standard normal vector `v_0` is distributed according to prior distribution
            on `X(0) | Z = z`.
        generate_z (Callable[[array], array]): Generator function for parameter vector
            such that `generator_z(u)` for standard normal vector `u` is distributed
            according to prior distribution on parameter vector `Z`.
        generate_σ (Callable[[array], array] or array): Function to generate
            standard-deviation(s) of independent and zero-mean Gaussian noise added to
            observations or a arrat specifying the observation noise standard
            deviation(s) directly if fixed. Function should accept a single array
            argument corresponding to the vector `u` mapping to the parameters which
            allows for variable observation noise standard-deviations.
        obs_func (Callable[[array], array]): Function mapping from state vector `x` at
            an observation time to the corresponding observed vector `y = obs_func(x)`.
        use_gaussian_splitting (bool): Whether to use Gaussian specific splitting
                h₁(q) = ½‖(y - g_y(q)) / σ‖² + D_y * log(σ), h₂(q, p) = ½qᵀq + ½pᵀM⁻¹p
            Or the more standard splitting
                h₁(q) = ½‖(y - g_y(q)) / σ‖² + D_y * log(σ) + ½qᵀq, h₂(q, p) = ½pᵀM⁻¹p
            to define the leapfrog integrator for the Hamiltonian dynamics.
        return_jax_funcs (bool): Whether to return original JAX function (True) suitable
            for further JAX transformation or wrapped versions which output NumPy arrays
            rather JAX `DeviceArray` instances.
    """

    num_obs, dim_y = y_seq.shape
    δ = obs_interval / num_steps_per_obs
    if isinstance(generate_σ, Number):
        σ = generate_σ

        def generate_σ(u):
            return σ

    @jax.jit
    def _neg_log_dens(q):
        num_step = num_steps_per_obs * num_obs
        u, v_0, v_seq_flat = split(q, (dim_u, dim_v_0, dim_v * num_step))
        z = generate_z(u)
        σ = generate_σ(u)
        x_0 = generate_x_0(z, v_0)
        v_seq = v_seq_flat.reshape((num_step, dim_v))

        def step_func(x, v):
            x_n = forward_func(z, x, v, δ)
            return x_n, x_n

        _, x_seq = lax.scan(step_func, x_0, v_seq)
        y_seq_mean = obs_func(x_seq[num_steps_per_obs - 1 :: num_steps_per_obs])
        return (
            0.5 * np.sum(((y_seq - y_seq_mean) / σ) ** 2)
            + num_obs * dim_y * np.log(σ)
            + (0 if use_gaussian_splitting else 0.5 * np.sum(q ** 2))
        )

    _grad_neg_log_dens = jax.jit(jax.value_and_grad(_neg_log_dens))

    if return_jax_funcs:
        return _neg_log_dens, lambda q: _grad_neg_log_dens(q)[::-1]

    def neg_log_dens(q):
        val = float(_neg_log_dens(q))
        if not onp.isfinite(val):
            raise HamiltonianDivergenceError("Hamiltonian non-finite")
        else:
            return val

    def grad_neg_log_dens(q):
        val, grad = _grad_neg_log_dens(q)
        if not onp.isfinite(val):
            raise HamiltonianDivergenceError("Hamiltonian non-finite")
        else:
            return onp.asarray(grad), float(val)

    return neg_log_dens, grad_neg_log_dens


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
        generate_σ=None,
        use_gaussian_splitting=False,
        metric=None,
        dim_v_0=None,
    ):
        """
        Args:
            obs_interval (float): Interobservation time interval.
            num_steps_per_obs (int): Number of discrete time steps to simulate
                between each observation time.
            num_obs_per_subseq (int or None): Number of observations per partitioned
                subsequence. Shorter initial / final subsequences will be used when
                the `num_obs_per_subseq` is not a factor of `num_obs`. If equal to
                `None` or `num_obs` no partitioning / blocking will be performed.
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
            generate_σ (None or Callable[[array], array] or array): Function to generate
                standard-deviation(s) of independent and zero-mean Gaussian noise added
                to observations. Function should accept a single array argument
                corresponding to the vector `u` mapping to the parameters which allows
                for variable observation noise standard-deviations. Alternatively if
                the standard-deviations are known and fixed a constant array may be
                specified instead of a function. If equal to `None` (the default)
                noiseless observations will be assumed (with the resulting latent state
                not including a component for the observation noise).
            use_gaussian_splitting (bool): Whether to use Gaussian specific splitting
                    h₁(q) = ½log(∂c(q)ᵀ∂c(q)), h₂(q, p) = ½qᵀq + ½pᵀp
                Or the more standard splitting
                    h₁(q) = ½qᵀq  + ½log(∂c(q)ᵀM⁻¹∂c(q)), h₂(q, p) =  ½pᵀM⁻¹p
                In the former case the metric matrix representation is required to be
                the identity matrix.
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
            metric.blocks[1], IdentityMatrix,
        ):
            log_det_sqrt_metric_0 = metric.blocks[0].log_abs_det / 2
        else:
            raise NotImplementedError(
                "Only identity and block diagonal metrics with identity lower "
                "right block currently supported."
            )

        num_obs, dim_y = y_seq.shape
        δ = obs_interval / num_steps_per_obs
        num_step = num_obs * num_steps_per_obs
        obs_indices = slice(num_steps_per_obs - 1, None, num_steps_per_obs)
        if num_obs_per_subseq is None or num_obs_per_subseq == num_obs:
            y_subseq_shapes = [((num_obs,),)]
            v_subseq_shapes = [((num_obs * num_steps_per_obs,),)]
            subseqs_are_batched = [(False,)]
        else:
            y_subseq_shapes, v_subseq_shapes, subseqs_are_batched = [], [], []
            for init_subseq_size in [num_obs_per_subseq, num_obs_per_subseq // 2]:
                num_full, num_remaining = divmod(
                    num_obs - init_subseq_size, num_obs_per_subseq
                )
                num_middle = num_full - 1 if num_remaining == 0 else num_full
                final_subseq_size = (
                    num_obs_per_subseq if num_remaining == 0 else num_remaining
                )
                y_subseq_shapes.append(
                    ((init_subseq_size,),)
                    + (((num_middle, num_obs_per_subseq),) if num_middle > 0 else ())
                    + ((final_subseq_size,),)
                )
                v_subseq_shapes.append(
                    ((init_subseq_size * num_steps_per_obs,),)
                    + (
                        ((num_middle, num_obs_per_subseq * num_steps_per_obs),)
                        if num_middle > 0
                        else ()
                    )
                    + ((final_subseq_size * num_steps_per_obs,),)
                )
                subseqs_are_batched.append(
                    (False, True, False) if num_middle > 0 else (False, False)
                )
        y_subseqs = [split_and_reshape(y_seq, shapes) for shapes in y_subseq_shapes]
        noisy_observations = generate_σ is not None
        if generate_σ is not None and isinstance(generate_σ, Number):
            σ = generate_σ

            def generate_σ(u):
                return σ

        dim_v_0 = dim_x if dim_v_0 is None else dim_v_0
        self.y_subseqs = y_subseqs
        self.num_partition = len(y_subseqs)
        self.model_dict = {
            "dim_u": dim_u,
            "dim_v": dim_v,
            "dim_v_0": dim_v_0,
            "dim_y": dim_y,
            "num_obs": num_obs,
            "num_steps_per_obs": num_steps_per_obs,
            "δ": δ,
            "generate_z": generate_z,
            "generate_x_0": generate_x_0,
            "generate_σ": generate_σ,
            "forward_func": forward_func,
            "obs_func": obs_func,
            "y_seq": y_seq,
        }

        @jax.jit
        def step_func(z, x, v):
            x_n = forward_func(z, x, v, δ)
            return (x_n, x_n)

        @jax.jit
        def generate_x_obs_seq(q):
            """Generate state sequence at observation time indices."""
            if noisy_observations:
                u, v_0, v_seq_flat, _ = split(
                    q, (dim_u, dim_v_0, num_obs * num_steps_per_obs * dim_v)
                )
            else:
                u, v_0, v_seq_flat = split(q, (dim_u, dim_v_0,))
            z = generate_z(u)
            x_0 = generate_x_0(z, v_0)
            v_seq = np.reshape(v_seq_flat, (-1, dim_v))
            _, x_seq = lax.scan(lambda x, v: step_func(z, x, v), x_0, v_seq)
            return x_seq[obs_indices]

        def generate_y_bar(z, w_0, v_seq, σ_n_seq, initial_subseq, final_subseq):
            """Generate partial observation subsequence."""
            x_0 = generate_x_0(z, w_0) if initial_subseq else w_0
            _, x_seq = lax.scan(lambda x, v: step_func(z, x, v), x_0, v_seq)
            y_seq = obs_func(x_seq[obs_indices])
            if noisy_observations:
                y_seq = y_seq + σ_n_seq
            if final_subseq:
                return y_seq.flatten()
            elif noisy_observations:
                return np.concatenate((y_seq.flatten(), x_seq[-1]))
            else:
                return np.concatenate((y_seq[:-1].flatten(), x_seq[-1]))

        @partial(jax.jit, static_argnames=("partition",))
        def partition_into_subseqs(v_seq, v_0, n_seq, x_obs_seq, partition=0):
            """Partition noise and observation sequences into subsequences.

            Partitition sequences in to either `num_subseq` equally sized
            subsequences (`partition == 0`)  or `num_subseq - 1` equally sized
            subsequences plus initial and final 'half' subsequences.
            """
            end_y = None if noisy_observations else -1
            partition_size = len(y_subseq_shapes[partition])
            v_subseqs = split_and_reshape(v_seq, v_subseq_shapes[partition])
            if noisy_observations:
                n_subseqs = split_and_reshape(n_seq, y_subseq_shapes[partition])
            else:
                n_subseqs = (None,) * partition_size
            x_obs_subseqs = split_and_reshape(x_obs_seq, y_subseq_shapes[partition])
            w_inits = [v_0]
            prev_batched = False
            for b in range(1, partition_size):
                if subseqs_are_batched[partition][b]:
                    w_inits.append(
                        np.vstack(
                            [
                                x_obs_subseqs[b - 1][(-1, -1) if prev_batched else -1],
                                x_obs_subseqs[b][:-1, -1],
                            ]
                        )
                    )
                    prev_batched = True
                else:
                    w_inits.append(
                        x_obs_subseqs[b - 1][(-1, -1) if prev_batched else (-1,)]
                    )
                    prev_batched = False
            y_bars = []
            for b in range(0, partition_size - 1):
                if subseqs_are_batched[partition][b]:
                    y_bars.append(
                        np.concatenate(
                            (
                                y_subseqs[partition][b][:, :end_y].reshape(
                                    (y_subseqs[partition][b].shape[0], -1)
                                ),
                                x_obs_subseqs[b][:, -1],
                            ),
                            -1,
                        )
                    )
                else:
                    y_bars.append(
                        np.concatenate(
                            (
                                y_subseqs[partition][b][:end_y].flatten(),
                                x_obs_subseqs[b][-1],
                            )
                        )
                    )
            y_bars.append(y_subseqs[partition][-1].flatten())
            return v_subseqs, n_subseqs, w_inits, y_bars

        @partial(jax.jit, static_argnames=("partition",))
        def constr(q, x_obs_seq, partition=0):
            """Calculate constraint function for current partition."""
            if noisy_observations:
                u, v_0, v_seq_flat, n_flat = split(
                    q, (dim_u, dim_v_0, num_step * dim_v, num_obs * dim_y)
                )
                n_seq = n_flat.reshape((-1, dim_y))
            else:
                u, v_0, v_seq_flat = split(q, (dim_u, dim_v_0,))
                n_seq = None
            v_seq = v_seq_flat.reshape((-1, dim_v))
            z = generate_z(u)
            (v_subseqs, n_subseqs, w_inits, y_bars) = partition_into_subseqs(
                v_seq, v_0, n_seq, x_obs_seq, partition
            )
            partition_size = len(v_subseqs)
            gen_funcs = [
                jax.vmap(
                    generate_y_bar,
                    (None, 0, 0, 0 if noisy_observations else None, None, None),
                )
                if is_batched
                else generate_y_bar
                for is_batched in subseqs_are_batched[partition]
            ]
            if noisy_observations:
                σ = generate_σ(u)
                σ_n_subseqs = [σ * n_subseq for n_subseq in n_subseqs]
            else:
                σ_n_subseqs = (None,) * partition_size
            return np.concatenate(
                [
                    (
                        gen_funcs[b](
                            z,
                            w_inits[b],
                            v_subseqs[b],
                            σ_n_subseqs[b],
                            b == 0,
                            b == partition_size - 1,
                        )
                        - y_bars[b]
                    ).flatten()
                    for b in range(partition_size)
                ]
            )

        @partial(jax.jit, static_argnames=("partition",))
        def jacob_constr_blocks(q, x_obs_seq, partition=0):
            """Return non-zero blocks of constraint function Jacobian.

            Input state `q` can be decomposed as

                q = [u, v₀, v₁, v₂, n₀, n₁, n₂]

            where global latent variables (parameters) are determined by `u`, Wiener
            noise increment subsequences by `(v₀, v₁, v₂)` (initial, middle, final
            subsequences) and observation noise subsequences by `(n₀, n₁, n₂)` (initial,
            middle, final subsequences) where present i.e. when observations are noisy.

            The constraint function can then be decomposed as

                c(q) = c̅(u, v, n) = [c₀(u, v₀, n₀), c₁(u, v₁, n₁), c₂(u, v₂, n₂)]

            abd the constraint Jacobian has the block structure

                ∂c([u, v, n]) = [∂₀c̅(u, v, n), ∂₁c̅(u, v, n), ∂₂c̅(u, v, n)]

            with

                ∂₀c̅(u, v, n) = [[∂₀c₀(u, v₀, n₀)]
                                [∂₀c₁(u, v₁, n₁)]
                                [∂₀c₂(u, v₀, n₁)]]

                ∂₁c̅(u, v, n) = [[∂₁c₀(u, v₀, n₀),         0       ,        0       ]
                                [       0       ,  ∂₁c₁(u, v₁, n₁),        0       ]
                                [       0       ,         0       , ∂₁c₂(u, v₂, n₂)]]

                ∂₂c̅(u, v, n) = [[∂₂c₀(u, v₀, n₀),         0       ,        0       ]
                                [       0       ,  ∂₂c₁(u, v₁, n₁),        0       ]
                                [       0       ,         0       , ∂₂c₂(u, v₂, n₂)]]

            where `∂₁c₁(u, v₁, n₁)` ad `∂₂c₁(u, v₁, n₁)` are both block diagonal.
            """

            def g_y_bar(u, v, n, w_0, initial_subseq, final_subseq):
                z = generate_z(u)
                if noisy_observations:
                    σ = generate_σ(u)
                    σ_n = σ * n
                else:
                    σ_n = None
                if initial_subseq:
                    w_0, v = split(v, (dim_v_0,))
                v_seq = v.reshape((-1, dim_v))
                return generate_y_bar(z, w_0, v_seq, σ_n, initial_subseq, final_subseq)

            if noisy_observations:
                u, v_0, v_seq_flat, n_flat = split(
                    q, (dim_u, dim_v_0, num_step * dim_v, num_obs * dim_y)
                )
                n_seq = n_flat.reshape((-1, dim_y))
            else:
                u, v_0, v_seq_flat = split(q, (dim_u, dim_v_0,))
                n_seq = None
            v_seq = v_seq_flat.reshape((-1, dim_v))
            (v_subseqs, n_subseqs, w_inits, _) = partition_into_subseqs(
                v_seq, v_0, n_seq, x_obs_seq, partition
            )
            partition_size = len(v_subseqs)
            v_bars = [np.concatenate([v_0, v_subseqs[0].flatten()])]
            for b in range(1, partition_size):
                v_bars.append(
                    v_subseqs[b].reshape((v_subseqs[b].shape[0], -1))
                    if subseqs_are_batched[partition][b]
                    else v_subseqs[b].flatten()
                )
            jacob_g_y_bar = jax.jacrev(g_y_bar, (0, 1))
            jacob_funcs = [
                jax.vmap(
                    jacob_g_y_bar,
                    (None, 0, 0 if noisy_observations else None, 0, None, None),
                )
                if is_batched
                else jacob_g_y_bar
                for is_batched in subseqs_are_batched[partition]
            ]
            if noisy_observations:
                σ = generate_σ(u)
                dc_dn_blocks = tuple(
                    (σ * np.ones_like(n_subseqs[b])).reshape(
                        (n_subseqs[b].shape[0], -1) if is_batched else (-1,)
                    )
                    for b, is_batched in enumerate(subseqs_are_batched[partition])
                )
            else:
                dc_dn_blocks = (None,) * partition_size
            dc_du_blocks, dc_dv_blocks = zip(
                *(
                    jacob_funcs[b](
                        u,
                        v_bars[b],
                        n_subseqs[b],
                        w_inits[b],
                        b == 0,
                        b == partition_size - 1,
                    )
                    for b in range(partition_size)
                )
            )
            return dc_du_blocks, dc_dv_blocks, dc_dn_blocks

        @jax.jit
        def chol_gram_blocks(dc_du_blocks, dc_dv_blocks, dc_dn_blocks):
            """Calculate Cholesky factors of decomposition of `∂c(q) M⁻¹ ∂c(q)ᵀ`.

            The constraint Jacobian decomposes as

                ∂c([u, v, n]) = [∂₀c̅(u, v, n), ∂₁c̅(u, v, n) ∂₂c̅(u, v, n)]

            where `c̅(u, v, n) = c([u, v, n])`, `∂₀c̅(u, v, n)` is a dense tall
            rectangular matrix, `∂₁c̅(u, v, n)` is a rectangular block diagonal matrix
            and `∂₂c̅(u, v, n)` is a rectangular block diagonal matrix. Similarly the
            metric matrix represention `M` is assumed to have the block structure `M =
            diag(M₀, M₁, M₂)` where `M₀` is a `dim_u × dim_u` square matrix, `M₁` is a
            `dim_v × dim_v` diagonal matrix and `M₂` is a `num_obs × dim_y` diagonal
            matrix.

            The Gram matrix can therefore be decomposed as

                ∂c([u, v, n]) M⁻¹ ∂c([u, v, n])ᵀ =
                ∂₀c̅(u, v, n) M₀⁻¹ ∂₀c̅(u, v, n)ᵀ +
                ∂₁c̅(u, v, n) M₁⁻¹ ∂₁c̅(u, v, n)ᵀ +
                ∂₂c̅(u, v, n) M₂⁻¹ ∂₂c̅(u, v, n)ᵀ

            denoting

                D(u, v, n) = ∂₁c̅(u, v, n) M₁⁻¹ ∂₁c̅(u, v, n)ᵀ +
                             ∂₂c̅(u, v, n) M₂⁻¹ ∂₂c̅(u, v, n)ᵀ

            with `D` square block diagonal and positive definite and

                C(u, v, n) = M₀ + ∂₀c̅(u, v, n)ᵀ D(u, v, n)⁻¹ ∂₀c̅(u, v, n)

            with `C` positive definite, by the Woodbury matrix identity we have

                (∂c([u, v, n]) M⁻¹ ∂c([u, v, n])ᵀ)⁻¹ =
                D(u, v, n)⁻¹ -
                D(u, v, n)⁻¹ ∂₀c̅(u, v, n) C(u, v, n)⁻¹ ∂₀c̅(u, v, n)ᵀ D(u, v, n)⁻¹

            Therefore by computing the Cholesky decompositions of `C(u, v, n)` and the
            blocks of `D(u, v, n)` we can solve for systems in the Gram matrix.
            """
            M_0 = get_M_0_matrix()
            D_blocks = compute_D_blocks(
                dc_dv_blocks, dc_dn_blocks, dc_dv_blocks, dc_dn_blocks
            )
            chol_D_blocks = tuple(nla.cholesky(D_block) for D_block in D_blocks)
            D_inv_dc_du_blocks = tuple(
                sla.cho_solve((chol_D_block, True), dc_du_block)
                for chol_D_block, dc_du_block in zip(chol_D_blocks, dc_du_blocks)
            )
            chol_C = nla.cholesky(
                M_0
                + sum(
                    dc_du_block.T @ D_inv_dc_du_block
                    if dc_du_block.ndim == 2
                    else np.einsum("ijk,ijl->kl", dc_du_block, D_inv_dc_du_block)
                    for dc_du_block, D_inv_dc_du_block in zip(
                        dc_du_blocks, D_inv_dc_du_blocks
                    )
                )
            )
            return chol_C, chol_D_blocks

        @jax.jit
        def lu_jacob_product_blocks(
            dc_du_l_blocks,
            dc_dv_l_blocks,
            dc_dn_l_blocks,
            dc_du_r_blocks,
            dc_dv_r_blocks,
            dc_dn_r_blocks,
        ):
            """Calculate LU factors of decomposition of ∂c(q) M⁻¹ ∂c(q')ᵀ

            The constraint Jacobian decomposes as

                ∂c([u, v, n]) = [∂₀c̅(u, v, n), ∂₁c̅(u, v, n) ∂₂c̅(u, v, n)]

            where `c̅(u, v, n) = c([u, v, n])`, `∂₀c̅(u, v, n)` is a dense tall
            rectangular matrix, `∂₁c̅(u, v, n)` is a rectangular block diagonal matrix
            and `∂₂c̅(u, v, n)` is a rectangular block diagonal matrix. Similarly the
            metric matrix represention `M` is assumed to have the block structure `M =
            diag(M₀, M₁, M₂)` where `M₀` is a `dim_u × dim_u` square matrix, `M₁` is a
            `dim_v × dim_v` diagonal matrix and `M₂` is a `num_obs × dim_y` diagonal
            matrix.

            The Jacobian product `∂c([u, v, n]) M⁻¹ ∂c([u', v', n'])ᵀ` can therefore be
            decomposed as

                ∂c([u, v, n]) M⁻¹ ∂c([u', v', n'])ᵀ =
                ∂₀c̅(u, v, n) M₀⁻¹ ∂₀c̅(u', v', n')ᵀ +
                ∂₁c̅(u, v, n) M₁⁻¹ ∂₁c̅(u', v', n')ᵀ +
                ∂₂c̅(u, v, n) M₂⁻¹ ∂₂c̅(u', v', n')ᵀ

            denoting

                D(u, v, n, u', v', n') = ∂₁c̅(u, v, n) M₁⁻¹ ∂₁c̅(u', v', n')ᵀ +
                                         ∂₂c̅(u, v, n) M₂⁻¹ ∂₂c̅(u', v', n')ᵀ

            with `D` square block diagonal and

                C(u, v, n, u', v', n') =
                M₀ + ∂₀c̅(u', v', n')ᵀ D(u, v, n, u', v', n')⁻¹ ∂₀c̅(u, v, n)

            we then have by the Woodbury matrix identity

                (∂c([u, v, n]) M⁻¹ ∂c([u', v', n'])ᵀ)⁻¹ =
                D(u, v, n, u', v', n')⁻¹ -
                D(u, v, n, u', v', n')⁻¹
                ∂₀c̅(u, v, n) C(u, v, n, u', v', n')⁻¹ ∂₀c̅(u', v', n')ᵀ
                D(u, v, n, u', v', n')⁻¹

            Therefore by computing LU factors of `C` and the blocks of `D` we can solve
            for systems in the Jacobian product.
            """
            M_0 = get_M_0_matrix()
            D_blocks = compute_D_blocks(
                dc_dv_l_blocks, dc_dn_l_blocks, dc_dv_r_blocks, dc_dn_r_blocks
            )
            lu_and_piv_D_blocks = tuple(sla.lu_factor(D_block) for D_block in D_blocks)
            D_inv_dc_du_l_blocks = tuple(
                sla.lu_solve(lu_and_piv_D_block, dc_du_l_block)
                for lu_and_piv_D_block, dc_du_l_block in zip(
                    lu_and_piv_D_blocks, dc_du_l_blocks
                )
            )
            lu_and_piv_C = sla.lu_factor(
                M_0
                + sum(
                    dc_du_r_block.T @ D_inv_dc_du_l_block
                    if dc_du_r_block.ndim == 2
                    else np.einsum("ijk,ijl->kl", dc_du_r_block, D_inv_dc_du_l_block)
                    for dc_du_r_block, D_inv_dc_du_l_block in zip(
                        dc_du_r_blocks, D_inv_dc_du_l_blocks
                    )
                )
            )
            return lu_and_piv_C, lu_and_piv_D_blocks

        def compute_D_blocks(
            dc_dv_l_blocks, dc_dn_l_blocks, dc_dv_r_blocks, dc_dn_r_blocks
        ):
            D_blocks = [
                np.einsum("...ij,...kj", dc_dv_l_block, dc_dv_r_block)
                for dc_dv_l_block, dc_dv_r_block in zip(dc_dv_l_blocks, dc_dv_r_blocks)
            ]
            if noisy_observations:
                for b, (D_block, dc_dn_l_block, dc_dn_r_block) in enumerate(
                    zip(D_blocks[:-1], dc_dn_l_blocks[:-1], dc_dn_r_blocks[:-1])
                ):
                    D_blocks[b] = D_block.at[
                        (...,) + np.diag_indices(D_block.shape[-2])
                    ].add(
                        np.concatenate(
                            [
                                dc_dn_l_block * dc_dn_r_block,
                                np.zeros(D_block.shape[-3:-2] + (dim_x,)),
                            ],
                            axis=-1,
                        )
                    )
                D_blocks[-1] = (
                    D_blocks[-1]
                    .at[np.diag_indices(D_blocks[-1].shape[0])]
                    .add(dc_dn_l_blocks[-1] * dc_dn_r_blocks[-1])
                )
            return D_blocks

        def get_M_0_matrix():
            if isinstance(metric, IdentityMatrix):
                return np.identity(dim_u)
            else:
                return metric.blocks[0].array

        @jax.jit
        def log_det_sqrt_gram_from_chol(chol_C, chol_D_blocks):
            """Calculate log-determinant of Gram matrix from block Cholesky factors."""
            return (
                sum(
                    np.log(np.abs(chol_D_block.diagonal(0, -2, -1))).sum()
                    for chol_D_block in chol_D_blocks
                )
                + np.log(np.abs(chol_C.diagonal())).sum()
                - log_det_sqrt_metric_0
            )

        @partial(jax.jit, static_argnames=("partition",))
        def log_det_sqrt_gram(q, x_obs_seq, partition=0):
            """Calculate log-determinant of constraint Jacobian Gram matrix."""
            jac_blocks = jacob_constr_blocks(q, x_obs_seq, partition)
            chol_blocks = chol_gram_blocks(*jac_blocks)
            return (
                log_det_sqrt_gram_from_chol(*chol_blocks),
                (jac_blocks, chol_blocks),
            )

        @jax.jit
        def lmult_by_jacob_constr(dc_du_blocks, dc_dv_blocks, dc_dn_blocks, vct):
            """Left-multiply vector by constraint Jacobian matrix."""
            if noisy_observations:
                vct_u, vct_v, vct_n = split(
                    vct, (dim_u, dim_v_0 + num_obs * num_steps_per_obs * dim_v,)
                )
            else:
                vct_u, vct_v = split(vct, (dim_u,))
            vct_v_parts = split_and_reshape(
                vct_v,
                [
                    dc_dv_block.shape[0:3:2]
                    if dc_dv_block.ndim == 3
                    else dc_dv_block.shape[1:2]
                    for dc_dv_block in dc_dv_blocks
                ],
            )
            dc_du = np.vstack(
                [
                    dc_du_block.reshape((-1, dim_u))
                    if dc_du_block.ndim == 3
                    else dc_du_block
                    for dc_du_block in dc_du_blocks
                ]
            )
            jacob_vct = dc_du @ vct_u + np.concatenate(
                [
                    np.einsum("ijk,ik->ij", dc_dv_block, vct_v_part,).flatten()
                    if dc_dv_block.ndim == 3
                    else dc_dv_block @ vct_v_part
                    for dc_dv_block, vct_v_part in zip(dc_dv_blocks, vct_v_parts)
                ]
            )
            if noisy_observations:
                vct_n_parts = split_and_reshape(
                    vct_n, [dc_dn_block.shape for dc_dn_block in dc_dn_blocks]
                )
                jacob_vct += np.concatenate(
                    [
                        np.concatenate(
                            [
                                dc_dn_block * vct_n_part,
                                np.zeros((dc_dn_block.shape[0], dim_x)),
                            ],
                            axis=1,
                        ).flatten()
                        if dc_dn_block.ndim == 2
                        else np.concatenate([dc_dn_block * vct_n_part, np.zeros(dim_x)])
                        for dc_dn_block, vct_n_part in zip(
                            dc_dn_blocks[:-1], vct_n_parts[:-1]
                        )
                    ]
                    + [dc_dn_blocks[-1] * vct_n_parts[-1]]
                )
            return jacob_vct

        @jax.jit
        def rmult_by_jacob_constr(dc_du_blocks, dc_dv_blocks, dc_dn_blocks, vct):
            """Right-multiply vector by constraint Jacobian matrix."""
            vct_parts = split_and_reshape(
                vct, [dc_du_block.shape[:-1] for dc_du_block in dc_du_blocks]
            )
            return np.concatenate(
                [
                    sum(
                        np.einsum("ij,ijk->k", vct_part, dc_du_block)
                        if vct_part.ndim == 2
                        else vct_part @ dc_du_block
                        for vct_part, dc_du_block in zip(vct_parts, dc_du_blocks)
                    )
                ]
                + [
                    np.einsum("ij,ijk->ik", vct_part, dc_dv_block).flatten()
                    if vct_part.ndim == 2
                    else vct_part @ dc_dv_block
                    for vct_part, dc_dv_block in zip(vct_parts, dc_dv_blocks)
                ]
                + (
                    [
                        (vct_part[:, :-dim_x] * dc_dn_block).flatten()
                        if vct_part.ndim == 2
                        else vct_part[:-dim_x] * dc_dn_block
                        for vct_part, dc_dn_block in zip(
                            vct_parts[:-1], dc_dn_blocks[:-1]
                        )
                    ]
                    + [vct_parts[-1] * dc_dn_blocks[-1]]
                    if noisy_observations
                    else []
                )
            )

        @jax.jit
        def lmult_by_inv_gram(
            dc_du_blocks, dc_dv_blocks, dc_dn_blocks, chol_C, chol_D_blocks, vct
        ):
            """Left-multiply vector by inverse Gram matrix."""
            vct_parts = split_and_reshape(
                vct, [dc_du_block.shape[:-1] for dc_du_block in dc_du_blocks]
            )
            D_inv_vct_blocks = [
                sla.cho_solve((chol_D_block, True), vct_part)
                for chol_D_block, vct_part in zip(chol_D_blocks, vct_parts)
            ]
            dc_du_T_D_inv_vct = sum(
                np.einsum("...jk,...j->k", dc_du_block, D_inv_vct_block)
                for dc_du_block, D_inv_vct_block in zip(dc_du_blocks, D_inv_vct_blocks)
            )
            C_inv_dc_du_T_D_inv_vct = sla.cho_solve((chol_C, True), dc_du_T_D_inv_vct)
            return np.concatenate(
                [
                    sla.cho_solve(
                        (chol_D_block, True),
                        vct_part - dc_du_block @ C_inv_dc_du_T_D_inv_vct,
                    ).flatten()
                    for chol_D_block, vct_part, dc_du_block in zip(
                        chol_D_blocks, vct_parts, dc_du_blocks
                    )
                ]
            )

        @jax.jit
        def lmult_by_inv_jacob_product(
            dc_du_l_blocks,
            dc_dv_l_blocks,
            dc_dn_l_blocks,
            dc_du_r_blocks,
            dc_dv_r_blocks,
            dc_dn_r_blocks,
            lu_and_piv_C,
            lu_and_piv_D_blocks,
            vct,
        ):
            """Left-multiply vector by inverse of Jacobian product matrix."""
            vct_parts = split_and_reshape(
                vct, [dc_du_l_block.shape[:-1] for dc_du_l_block in dc_du_l_blocks]
            )
            D_inv_vct_blocks = [
                sla.lu_solve(lu_and_piv_D_block, vct_part)
                for lu_and_piv_D_block, vct_part in zip(lu_and_piv_D_blocks, vct_parts)
            ]
            dc_du_r_T_D_inv_vct = sum(
                np.einsum("...jk,...j->k", dc_du_r_block, D_inv_vct_block)
                for dc_du_r_block, D_inv_vct_block in zip(
                    dc_du_r_blocks, D_inv_vct_blocks
                )
            )
            C_inv_dc_du_r_T_D_inv_vct = sla.lu_solve(lu_and_piv_C, dc_du_r_T_D_inv_vct)
            return np.concatenate(
                [
                    sla.lu_solve(
                        lu_and_piv_D_block,
                        vct_part - dc_du_l_block @ C_inv_dc_du_r_T_D_inv_vct,
                    ).flatten()
                    for lu_and_piv_D_block, vct_part, dc_du_l_block in zip(
                        lu_and_piv_D_blocks, vct_parts, dc_du_l_blocks
                    )
                ]
            )

        @jax.jit
        def normal_space_component(vct, jacob_constr_blocks, chol_gram_blocks):
            """Compute component of vector in normal space to manifold at a point."""
            return rmult_by_jacob_constr(
                *jacob_constr_blocks,
                lmult_by_inv_gram(
                    *jacob_constr_blocks,
                    *chol_gram_blocks,
                    lmult_by_jacob_constr(*jacob_constr_blocks, vct),
                ),
            )

        def norm(x):
            """Infinity norm of a vector."""
            return np.max(np.abs(x))

        @partial(
            jax.jit,
            static_argnames=(
                "partition",
                "constraint_tol",
                "position_tol",
                "divergence_tol",
                "max_iters",
            ),
        )
        def quasi_newton_projection(
            q,
            x_obs_seq,
            partition,
            jacob_constr_blocks_prev,
            chol_gram_blocks_prev,
            dt,
            constraint_tol,
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
                    *jacob_constr_blocks_prev,
                    lmult_by_inv_gram(
                        *jacob_constr_blocks_prev, *chol_gram_blocks_prev, c,
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
                    error < constraint_tol, norm_delta_q < position_tol
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

        @partial(
            jax.jit,
            static_argnames=(
                "partition",
                "constraint_tol",
                "position_tol",
                "divergence_tol",
                "max_iters",
            ),
        )
        def newton_projection(
            q,
            x_obs_seq,
            partition,
            jacob_constr_blocks_prev,
            dt,
            constraint_tol,
            position_tol,
            divergence_tol,
            max_iters,
        ):
            """Newton method to solve projection onto manifold."""

            def body_func(val):
                q, mu, i, _, _ = val
                c = constr(q, x_obs_seq, partition)
                jacob_constr_blocks_curr = jacob_constr_blocks(q, x_obs_seq, partition)
                lu_and_piv_jacob_product_blocks = lu_jacob_product_blocks(
                    *jacob_constr_blocks_curr, *jacob_constr_blocks_prev
                )
                error = norm(c)
                delta_mu = rmult_by_jacob_constr(
                    *jacob_constr_blocks_prev,
                    lmult_by_inv_jacob_product(
                        *jacob_constr_blocks_curr,
                        *jacob_constr_blocks_prev,
                        *lu_and_piv_jacob_product_blocks,
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
                    error < constraint_tol, norm_delta_q < position_tol
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
        self._grad_log_det_sqrt_gram = jax.jit(
            jax.value_and_grad(log_det_sqrt_gram, has_aux=True),
            static_argnames=("partition",),
        )
        self._normal_space_component = normal_space_component
        self._quasi_newton_projection = quasi_newton_projection
        self._newton_projection = newton_projection

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
            sin_dt, cos_dt = onp.sin(dt), onp.cos(dt)
            pos = state.pos.copy()
            state.pos *= cos_dt
            state.pos += sin_dt * state.mom
            state.mom *= cos_dt
            state.mom -= sin_dt * pos
        else:
            state.pos += dt * self.dh2_dmom(state)

    def dh2_flow_dmom(self, dt):
        if self.use_gaussian_splitting:
            sin_dt, cos_dt = onp.sin(dt), onp.cos(dt)
            return sin_dt * IdentityMatrix(), cos_dt * IdentityMatrix()
        else:
            return (dt * self.metric.inv, IdentityMatrix())

    def update_x_obs_seq(self, state):
        state.x_obs_seq = convert_to_numpy_pytree(self._generate_x_obs_seq(state.pos))

    def normal_space_component(self, state, vct):
        return convert_to_numpy_pytree(
            self._normal_space_component(
                self.metric.inv @ vct,
                self.jacob_constr_blocks(state),
                self.chol_gram_blocks(state),
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

    The `partition` integer variable in the chain state, which sets the current
    partition used when conditioning on values of the diffusion process at intermediate
    time, is deterministically circularly incremented on each transition as well as
    updating the cached observed state sequence based on the current position state
    component.
    """

    def __init__(self, system):
        self.system = system
        self.num_partition = system.num_partition

    state_variables = {"partition", "x_obs_seq"}
    statistic_types = None

    def sample(self, state, rng):
        state.partition = (state.partition + 1) % self.num_partition
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
    constraint_tol=1e-8,
    position_tol=1e-8,
    divergence_tol=1e10,
    max_iters=50,
):
    """Symmetric quasi-Newton solver for projecting points onto manifold.

    Solves an equation of the form `r(λ) = c(q_ + M⁻¹ ∂c(q)ᵀλ) = 0` for the vector of
    Lagrange multipliers `λ` to project a point `q_` on to the manifold defined by the
    zero level set of `c`, with the projection performed with in the linear subspace
    defined by the rows of the Jacobian matrix `∂c(q)` evaluated at a previous point on
    the manifold `q`.

    The Jacobian of the residual function `r` is

        ∂r(λ) = ∂c(q_ + M⁻¹ ∂c(q)ᵀλ) M⁻¹ ∂c(q)ᵀ

    such that the full Newton update

        λ = λ - ∂r(λ)⁻¹ r(λ)

    requires evaluating `∂c` on each iteration. The symmetric quasi-Newton iteration
    instead uses the approximation

        ∂c(q_ + ∂c(q)ᵀλ) M⁻¹ ∂c(q)ᵀ ≈ ∂c(q) M⁻¹ ∂c(q)ᵀ

    with the corresponding update

        λ = λ - (∂c(q) M⁻¹ ∂c(q)ᵀ)⁻¹ r(λ)

    allowing a previously computed Cholesky decomposition of the Gram matrix `∂c(q) M⁻¹
    ∂c(q)ᵀ` to be used to solve the linear system in each iteration with no requirement
    to evaluate `∂c` on each iteration.

    Compared to the inbuilt solver in `mici.solvers` this version exploits the structure
    in the constraint Jacobian `∂c` for conditioned diffusion systems and JIT compiles
    the iteration using JAX for better performance.
    """
    jacob_constr_blocks_prev = system.jacob_constr_blocks(state_prev)
    chol_gram_blocks_prev = system.chol_gram_blocks(state_prev)
    q, x_obs_seq, partition = state.pos, state.x_obs_seq, state.partition
    _, dh2_flow_mom_dmom = system.dh2_flow_dmom(dt)
    q_, mu, i, norm_delta_q, error = system._quasi_newton_projection(
        q,
        x_obs_seq,
        partition,
        jacob_constr_blocks_prev,
        chol_gram_blocks_prev,
        dt,
        constraint_tol,
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
    if error < constraint_tol and norm_delta_q < position_tol:
        state.pos = onp.array(q_)
        if state.mom is not None:
            state.mom -= dh2_flow_mom_dmom @ onp.asarray(mu)
        return state
    elif error > divergence_tol or onp.isnan(error):
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
    constraint_tol=1e-8,
    position_tol=1e-8,
    divergence_tol=1e10,
    max_iters=50,
):
    """Newton solver for projecting points onto manifold.

    Solves an equation of the form `r(λ) = c(q_ + M⁻¹ ∂c(q)ᵀλ) = 0` for the vector of
    Lagrange multipliers `λ` to project a point `q_` on to the manifold defined by the
    zero level set of `c`, with the projection performed with in the linear subspace
    defined by the rows of the Jacobian matrix `∂c(q)` evaluated at a previous point on
    the manifold `q`.

    The Jacobian of the residual function `r` is

        ∂r(λ) = ∂c(q_ + M⁻¹ ∂c(q)ᵀλ) M⁻¹ ∂c(q)ᵀ

    such that the Newton update is

        λ = λ - ∂r(λ)⁻¹ r(λ)

    which requires evaluating `∂c` on each iteration.

    Compared to the inbuilt solver in `mici.solvers` this version exploits the structure
    in the constraint Jacobian `∂c` for conditioned diffusion systems and JIT compiles
    the iteration using JAX for better performance.
    """
    jacob_constr_blocks_prev = system.jacob_constr_blocks(state_prev)
    q, x_obs_seq, partition = state.pos, state.x_obs_seq, state.partition
    _, dh2_flow_mom_dmom = system.dh2_flow_dmom(dt)
    q_, mu, i, norm_delta_q, error = system._newton_projection(
        q,
        x_obs_seq,
        partition,
        jacob_constr_blocks_prev,
        dt,
        constraint_tol,
        position_tol,
        divergence_tol,
        max_iters,
    )
    if state._call_counts is not None:
        for method in [
            system.constr,
            system.jacob_constr_blocks,
            "lu_jacob_product_blocks",
        ]:
            key = _cache_key_func(system, method)
            if key in state._call_counts:
                state._call_counts[key] += i
            else:
                state._call_counts[key] = i
    if error < constraint_tol and norm_delta_q < position_tol:
        state.pos = onp.array(q_)
        if state.mom is not None:
            state.mom -= dh2_flow_mom_dmom @ onp.asarray(mu)
        return state
    elif error > divergence_tol or onp.isnan(error):
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
    system, rng, generate_x_obs_seq_init, u=None, v_0=None, **model_dict
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

    model_dict = system.model_dict if not model_dict else model_dict

    def mean_and_sqrt_covar_step_diff(z, x, δ):
        v = np.zeros(model_dict["dim_v"])

        def step_diff_func(v):
            return model_dict["forward_func"](z, x, v, δ) - x

        return step_diff_func(v), jax.jacobian(step_diff_func)(v)

    @jax.jit
    def solve_for_v_seq(x_obs_seq, x_0, z):
        num_obs = x_obs_seq.shape[0]

        def solve_inner(x, Δx):
            mean_diff, sqrt_covar_diff = mean_and_sqrt_covar_step_diff(
                z, x, model_dict["δ"]
            )
            return np.linalg.lstsq(sqrt_covar_diff, (Δx - mean_diff))[0]

        def solve_outer(x_0, x_1):
            Δx = (x_1 - x_0) / model_dict["num_steps_per_obs"]
            x_seq = (
                x_0[None]
                + np.arange(model_dict["num_steps_per_obs"])[:, None] * Δx[None]
            )
            return jax.vmap(solve_inner, (0, None))(x_seq, Δx)

        x_0_seq = np.concatenate((x_0[None], x_obs_seq[:-1]))
        x_1_seq = x_obs_seq

        return jax.vmap(solve_outer)(x_0_seq, x_1_seq).reshape(
            (num_obs * model_dict["num_steps_per_obs"], model_dict["dim_v"])
        )

    u = rng.standard_normal(model_dict["dim_u"]) if u is None else u
    z = model_dict["generate_z"](u)
    v_0 = rng.standard_normal(model_dict["dim_v_0"]) if v_0 is None else v_0
    x_0 = model_dict["generate_x_0"](z, v_0)
    x_obs_seq = onp.asarray(generate_x_obs_seq_init(rng))
    v_seq = solve_for_v_seq(x_obs_seq, x_0, z)
    if (
        isinstance(system, ConditionedDiffusionConstrainedSystem)
        and model_dict["generate_σ"] is not None
    ):
        n = onp.zeros(model_dict["dim_y"] * model_dict["num_obs"])
        q = onp.concatenate([u, v_0, v_seq.flatten(), n])
    else:
        q = onp.concatenate([u, v_0, v_seq.flatten()])
    if isinstance(system, ConditionedDiffusionConstrainedSystem):
        state = ConditionedDiffusionHamiltonianState(pos=q, x_obs_seq=x_obs_seq)
    else:
        state = ChainState(pos=q, mom=None, dir=1, _call_counts={})
    state.mom = system.sample_momentum(state, rng)
    return state


def find_initial_state_by_gradient_descent(
    system,
    rng,
    generate_x_obs_seq_init,
    tol=1e-9,
    adam_step_size=2e-1,
    reg_coeff=2e-2,
    coarse_tol=1e-1,
    max_iters=1000,
    max_num_tries=10,
    use_newton=True,
    **model_dict,
):
    """Find an initial constraint satisying state by a gradient descent based scheme.

    Uses a heuristic combination of gradient-based minimisation of the norm of a
    modified constraint function plus a subsequent projection step using a
    (quasi-)Newton method, to try to find an initial point `q` such that
    `max(abs(constr(q)) < tol`.
    """

    model_dict = system.model_dict if not model_dict else model_dict
    num_step = model_dict["num_steps_per_obs"] * model_dict["num_obs"]
    noisy_observations = model_dict["generate_σ"] is not None
    dim_q = (
        model_dict["dim_u"]
        + model_dict["dim_v_0"]
        + model_dict["dim_v"] * num_step
        + (model_dict["num_obs"] * model_dict["dim_y"] if noisy_observations else 0)
    )

    @jax.jit
    def init_objective(q, x_obs_seq, reg_coeff):
        """Optimisation objective to find initial state on manifold."""
        if noisy_observations:
            u, v_0, v_seq_flat, _ = split(
                q,
                (
                    model_dict["dim_u"],
                    model_dict["dim_v_0"],
                    num_step * model_dict["dim_v"],
                ),
            )
        else:
            u, v_0, v_seq_flat = split(q, (model_dict["dim_u"], model_dict["dim_v_0"],))
        v_subseqs = v_seq_flat.reshape(
            (
                model_dict["num_obs"],
                model_dict["num_steps_per_obs"],
                model_dict["dim_v"],
            )
        )
        z = model_dict["generate_z"](u)
        x_0 = model_dict["generate_x_0"](z, v_0)
        x_inits = np.concatenate((x_0[None], x_obs_seq[:-1]), 0)

        def step_func(x, v):
            x_n = model_dict["forward_func"](z, x, v, model_dict["δ"])
            return (x_n, x_n)

        def generate_final_state(z, v_seq, x_0):
            _, x_seq = lax.scan(step_func, x_0, v_seq)
            return x_seq[-1]

        c = (
            jax.vmap(generate_final_state, in_axes=(None, 0, 0))(z, v_subseqs, x_inits)
            - x_obs_seq
        )
        return 0.5 * np.mean(c ** 2) + 0.5 * reg_coeff * np.mean(q ** 2), c

    value_and_grad_init_objective = jax.jit(
        jax.value_and_grad(init_objective, (0,), has_aux=True)
    )

    # Use optimizers to set optimizer initialization and update functions
    opt_init, opt_update, get_params = opt.adam(adam_step_size)

    # Define a compiled update step
    @jax.jit
    def step(i, opt_state, x_obs_seq_init):
        (q,) = get_params(opt_state)
        (obj, constr), grad = value_and_grad_init_objective(
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
        q_init = rng.standard_normal(dim_q)
        x_obs_seq_init = generate_x_obs_seq_init(rng)
        opt_state = opt_init((q_init,))
        for i in range(max_iters):
            opt_state_next, norm, constr = step(i, opt_state, x_obs_seq_init)
            if not onp.isfinite(norm):
                logger.info("Adam iteration diverged")
                break
            max_abs_constr = maximum_norm(constr)
            if max_abs_constr < coarse_tol:
                logging.info("Within coarse_tol attempting projection.")
                (q_init,) = get_params(opt_state)
                q_init = onp.asarray(q_init)
                state = ConditionedDiffusionHamiltonianState(
                    q_init, x_obs_seq=x_obs_seq_init, _call_counts={}
                )
                try:
                    state = projection_solver(state, state, 1.0, system, tol)
                except ConvergenceError as e:
                    logger.info(e)
                    break
                if onp.max(onp.abs(system.constr(state))) < tol:
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


def find_initial_state_by_gradient_descent_noisy_system(
    system,
    rng,
    adam_step_size=2e-2,
    max_iters=1000,
    max_init_tries=100,
    max_num_tries=10,
    threshold=1.0,
    slow_progress_ratio=0.8,
    check_iter=100,
    **model_dict,
):
    """Find an initial constraint satisying state by a gradient descent based scheme.

    Performs gradienty descent on negative log posterior density for noisy observation
    system until mean residual squared is less than one, with state on manifold then
    constructed by setting observation noise terms to residuals.
    """

    model_dict = system.model_dict if not model_dict else model_dict
    num_step = model_dict["num_steps_per_obs"] * model_dict["num_obs"]
    dim_u_v = (
        model_dict["dim_u"] + model_dict["dim_v_0"] + num_step * model_dict["dim_v"]
    )

    @jax.jit
    def init_objective(u_v):
        """Optimisation objective to find initial state for noisy systems."""
        num_step = model_dict["num_steps_per_obs"] * model_dict["num_obs"]
        u, v_0, v_flat = split(
            u_v,
            (
                model_dict["dim_u"],
                model_dict["dim_v_0"],
                num_step * model_dict["dim_v"],
            ),
        )
        v_seq = v_flat.reshape((num_step, model_dict["dim_v"]))
        z = model_dict["generate_z"](u)
        x_0 = model_dict["generate_x_0"](z, v_0)
        σ = model_dict["generate_σ"](u)

        def step_func(x, v):
            x_n = model_dict["forward_func"](z, x, v, model_dict["δ"])
            return (x_n, x_n)

        _, x_seq = lax.scan(step_func, x_0, v_seq)
        obs_slc = slice(
            model_dict["num_steps_per_obs"] - 1, None, model_dict["num_steps_per_obs"]
        )
        residuals = (model_dict["y_seq"] - model_dict["obs_func"](x_seq[obs_slc])) / σ
        return (
            0.5 * np.sum(residuals ** 2)
            + model_dict["num_obs"] * np.log(σ)
            + 0.5 * np.sum(u_v ** 2),
            residuals,
        )

    grad_init_objective = jax.jit(jax.grad(init_objective, has_aux=True))

    # Use optimizers to set optimizer initialization and update functions
    opt_init, opt_update, get_params = opt.adam(adam_step_size)

    # Define a compiled update step
    @jax.jit
    def step(i, opt_state):
        (u_v,) = get_params(opt_state)
        grad, residuals = grad_init_objective(u_v)
        opt_state = opt_update(i, (grad,), opt_state)
        return opt_state, residuals

    for t in range(max_num_tries):
        logging.info(f"Starting try {t+1}")
        found_valid_init_state, init_tries = False, 0
        while not found_valid_init_state and init_tries < max_init_tries:
            u_v = rng.standard_normal(dim_u_v)
            _, residuals = init_objective(u_v)
            if onp.all(onp.isfinite(residuals)):
                found_valid_init_state = True
            init_tries += 1
        if init_tries == max_init_tries:
            raise RuntimeError(
                f"Did not find valid initial state in {init_tries} tries."
            )
        opt_state = opt_init((u_v,))
        prev_mean_residual_sq = onp.mean(residuals ** 2)
        for i in range(max_iters):
            opt_state_next, residuals = step(i, opt_state)
            mean_residuals_sq = onp.mean(residuals ** 2)
            if not onp.isfinite(mean_residuals_sq):
                logger.info("Adam iteration diverged")
                break
            if mean_residuals_sq < threshold:
                logging.info(f"Found point with mean sq. residual < {threshold}")
                (u_v,) = get_params(opt_state)
                u_v = onp.asarray(u_v)
                if isinstance(system, ConditionedDiffusionConstrainedSystem):
                    state = ConditionedDiffusionHamiltonianState(
                        pos=onp.concatenate([u_v, residuals.flatten()]),
                        x_obs_seq=None,
                        _call_counts={},
                    )
                    system.update_x_obs_seq(state)
                else:
                    state = ChainState(pos=u_v, mom=None, dir=1, _call_counts={})
                state.mom = system.sample_momentum(state, rng)
                return state
            opt_state = opt_state_next
            if i % check_iter == 0:
                if (
                    i > 0
                    and i < max_iters // 2
                    and (mean_residuals_sq / prev_mean_residual_sq)
                    > slow_progress_ratio
                ):
                    logging.info("Slow progress, restarting")
                    break
                else:
                    logging.info(
                        f"Iteration {i: >6}: mean|residual^2| = {mean_residuals_sq:.3e}"
                    )
                    prev_mean_residual_sq = mean_residuals_sq
    raise RuntimeError(f"Did not find valid state in {max_num_tries} tries.")


class OnlineBlockDiagonalMetricAdapter(Adapter):
    """Block diagonal metric adapter using online covariance estimates.

    Uses Welford's algorithm [1] to stably compute an online estimate of the sample
    covariance matrix of the global parameter chain state position components (which are
    assume to form the first 'dim_param' components of the state position vector) during
    sampling. If online estimates are available from multiple independent chains, the
    final covariance matrix estimate is calculated from the per-chain statistics using a
    covariance variant due to Schubert and Gertz [2] of the parallel / batched
    incremental variance algorithm described by Chan et al. [3]. The covariance matrix
    estimates are optionally regularized towards a scaled identity matrix, with
    increasing weight for small number of samples, to decrease the effect of noisy
    estimates for small sample sizes, following the approach in Stan [4]. The metric
    matrix representation is set to a block diagonal matrix with uppper left block a
    dense positive definite matrix corresponding to the inverse of the (regularized)
    covariance matrix estimate and lower right block an identity matrix (representing a
    fixed metric for the remaining state components which are assumed to have standard
    normal priors and correspond to local latent variables which are less strongly
    informed by the observations and so remain close to their prior distribution under
    the posterior).


    References:

      1. Welford, B. P., 1962. Note on a method for calculating corrected sums
         of squares and products. Technometrics, 4(3), pp. 419–420.
      2. Schubert, E. and Gertz, M., 2018. Numerically stable parallel
         computation of (co-)variance. ACM. p. 10. doi:10.1145/3221269.3223036.
      3. Chan, T. F., Golub, G. H., LeVeque, R. J., 1979. Updating formulae and
         a pairwise algorithm for computing sample variances. Technical Report
         STAN-CS-79-773, Department of Computer Science, Stanford University.
      4. Carpenter, B., Gelman, A., Hoffman, M.D., Lee, D., Goodrich, B.,
         Betancourt, M., Brubaker, M., Guo, J., Li, P. and Riddell, A., 2017.
         Stan: A probabilistic programming language. Journal of Statistical
         Software, 76(1).
    """

    is_fast = False

    def __init__(self, dim_param, reg_iter_offset=5, reg_scale=1e-3):
        """
        Args:
            reg_iter_offset (int): Iteration offset used for calculating
                iteration dependent weighting between regularisation target and
                current covariance estimate. Higher values cause stronger
                regularisation during initial iterations.
            reg_scale (float): Positive scalar defining value variance estimates
                are regularized towards.
        """
        self.dim_param = dim_param
        self.reg_iter_offset = reg_iter_offset
        self.reg_scale = reg_scale

    def initialize(self, chain_state, transition):
        dtype = chain_state.pos.dtype
        return {
            "iter": 0,
            "mean": onp.zeros(shape=(self.dim_param,), dtype=dtype),
            "sum_diff_outer": onp.zeros(
                shape=(self.dim_param, self.dim_param), dtype=dtype
            ),
            "dim_pos": chain_state.pos.shape[0],
        }

    def update(self, adapt_state, chain_state, trans_stats, transition):
        # Use Welford (1962) incremental algorithm to update statistics to
        # calculate online covariance estimate
        # https://en.wikipedia.org/wiki/
        #  Algorithms_for_calculating_variance#Online
        adapt_state["iter"] += 1
        pos_minus_mean = chain_state.pos[: self.dim_param] - adapt_state["mean"]
        adapt_state["mean"] += pos_minus_mean / adapt_state["iter"]
        adapt_state["sum_diff_outer"] += (
            pos_minus_mean[None, :]
            * (chain_state.pos[: self.dim_param] - adapt_state["mean"])[:, None]
        )

    def _regularize_covar_est(self, covar_est, n_iter):
        """Update covariance estimate by regularising towards identity.

        Performed in place to prevent further array allocations.
        """
        covar_est *= n_iter / (self.reg_iter_offset + n_iter)
        covar_est_diagonal = onp.einsum("ii->i", covar_est)
        covar_est_diagonal += self.reg_scale * (
            self.reg_iter_offset / (self.reg_iter_offset + n_iter)
        )

    def finalize(self, adapt_state, transition):
        if isinstance(adapt_state, dict):
            n_iter = adapt_state["iter"]
            covar_est = adapt_state.pop("sum_diff_outer")
            dim_pos = adapt_state["dim_pos"]
        else:
            # Use Schubert and Gertz (2018) parallel covariance estimation
            # algorithm to combine per-chain statistics
            for i, a in enumerate(adapt_state):
                if i == 0:
                    n_iter = a["iter"]
                    mean_est = a.pop("mean")
                    covar_est = a.pop("sum_diff_outer")
                    dim_pos = a["dim_pos"]
                else:
                    n_iter_prev = n_iter
                    n_iter += a["iter"]
                    mean_diff = mean_est - a["mean"]
                    mean_est *= n_iter_prev
                    mean_est += a["iter"] * a["mean"]
                    mean_est /= n_iter
                    covar_est += a["sum_diff_outer"]
                    covar_est += (
                        onp.outer(mean_diff, mean_diff)
                        * (a["iter"] * n_iter_prev)
                        / n_iter
                    )
        if n_iter < 2:
            raise AdaptationError(
                "At least two chain samples required to compute a variance "
                "estimates."
            )
        covar_est /= n_iter - 1
        self._regularize_covar_est(covar_est, n_iter)
        transition.system.metric = PositiveDefiniteBlockDiagonalMatrix(
            (
                DensePositiveDefiniteMatrix(covar_est).inv,
                IdentityMatrix(dim_pos - self.dim_param),
            )
        )
