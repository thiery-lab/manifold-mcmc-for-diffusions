import logging
import numpy as onp
import scipy.linalg as osla
import jax.numpy as np
import jax.numpy.linalg as nla
import jax.scipy.linalg as sla
from jax import lax, api
import jax.experimental.optimizers as opt
from mici.systems import (
    EuclideanMetricSystem, cache_in_state, multi_cache_in_state)
from mici.matrices import (
    SymmetricBlockDiagonalMatrix, IdentityMatrix, PositiveDiagonalMatrix, 
    DensePositiveDefiniteMatrix, )
from mici.transitions import Transition
from mici.states import ChainState
from mici.solvers import maximum_norm
from mici.errors import ConvergenceError
from functools import partial


def split(v, lengths):
    i = 0
    for l in lengths:
        yield v[i:i+l]
        i += l
    if i < len(v):
        yield v[i:]


def standard_normal_neg_log_dens(q):
    return 0.5 * onp.sum(q**2)


def standard_normal_grad_neg_log_dens(q):
    return q, 0.5 * onp.sum(q**2)


class ConditionedDiffusionConstrainedSystem(EuclideanMetricSystem):
    
    def __init__(self, obs_interval, num_steps_per_obs, num_obs_per_subseq,
                 y_obs_seq, num_param, dim_state, dim_noise, forward_op_func, 
                 generate_init_state, generate_params, obs_func, metric=None, 
                 neg_log_input_density=standard_normal_neg_log_dens,
                 grad_neg_log_input_density=standard_normal_grad_neg_log_dens):
        
        if metric is None or isinstance(metric, IdentityMatrix):
            metric_1 = np.eye(num_param)
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
            
        self.dim_u_v0 = num_param + dim_state

        num_obs, dim_obs = y_obs_seq.shape
        delta = obs_interval / num_steps_per_obs
        dim_pos = (
            num_param + dim_state + num_obs * dim_noise * num_steps_per_obs)
        if num_obs % num_obs_per_subseq != 0:
            raise NotImplementedError(
                 'Only cases where num_obs_per_subseq is a factor of num_obs '
                 'supported.')
        num_block = num_obs // num_obs_per_subseq
        obs_indices = (1 + np.arange(num_obs)) * num_steps_per_obs - 1
        num_step_per_block = num_obs_per_subseq * num_steps_per_obs
        y_obs_seq_blk_p0 = np.reshape(
        y_obs_seq, (num_block, num_obs_per_subseq, -1))
        y_obs_seq_0_p1, y_obs_seq_1_p1, y_obs_seq_2_p1 = split(
            y_obs_seq, (num_obs_per_subseq // 2, 
                        num_obs_per_subseq  * (num_block - 1),))
        y_obs_seq_1_blk_p1 = np.reshape(
            y_obs_seq_1_p1, (num_block - 1, num_obs_per_subseq, dim_obs))
                               
        super().__init__(
            neg_log_dens=neg_log_input_density,
            grad_neg_log_dens=grad_neg_log_input_density, metric=metric)

        def step_func(x, v, params):
            x_n = forward_op_func(x, v, delta, **params) 
            return (x_n, x_n)

        def generate_x_obs_seq(q):
            u, v_0, v_r = np.split(q, (num_param, num_param + dim_state))
            params = generate_params(u)
            x_init = generate_init_state(v_0, params)
            v_seq = np.reshape(v_r, (-1, dim_noise))
            _, x_seq = lax.scan(
                partial(step_func, params=params), x_init, v_seq)
            return x_seq[num_steps_per_obs-1::num_steps_per_obs]

        def generate_obs(params, v_seq, x_init, to_end=False):
            _, x_seq = lax.scan(
                partial(step_func, params=params), x_init, v_seq)
            if to_end:
                return obs_func(
                    x_seq[num_steps_per_obs-1::num_steps_per_obs]).flatten()
            else:
                return np.concatenate(
                    [obs_func(x_seq[num_steps_per_obs-1:-num_steps_per_obs:
                                    num_steps_per_obs]).flatten(), x_seq[-1]])

        def vmapped_generate_obs(params, v_seq_blocks, x_init_blocks):
            
            def func(v_seq, x_init):
                return generate_obs(params, v_seq, x_init)
            vmapped_gen_func = api.vmap(partial(generate_obs, params))
            return vmapped_gen_func(v_seq_blocks, x_init_blocks)
        
        def partition_seq(v_seq, x_obs_seq, partition=0):
            """Partition noise increment and observation sequences.
            
            Partitition sequences in to either 
               0. equally sized blocks or 
               1. half-block offset equally sized blocks.
            """
            if partition == 0:
                v_seq_blk = np.reshape(
                    v_seq, (num_block, num_step_per_block, dim_noise))
                v_seq_0, v_seq_1, v_seq_2 = (
                    v_seq_blk[0], v_seq_blk[1:-1], v_seq_blk[-1])
                x_obs_seq_blk = np.reshape(
                    x_obs_seq, (num_block, num_obs_per_subseq, dim_state))
                x_init_1, x_init_2 = (
                    x_obs_seq_blk[:-2, -1], x_obs_seq_blk[-2, -1])
                y_tilde_0 = np.concatenate(
                    (y_obs_seq_blk_p0[0, :-1].flatten(), x_obs_seq_blk[0, -1]))
                y_tilde_1 = np.concatenate(
                    (y_obs_seq_blk_p0[1:-1, :-1].reshape((num_block - 2, -1)), 
                     x_obs_seq_blk[1:-1, -1]), -1)
                y_tilde_2 = y_obs_seq_blk_p0[-1].flatten()
            else:
                v_seq_0, v_seq_1, v_seq_2 = split(
                    v_seq, ((num_obs_per_subseq // 2) * num_steps_per_obs, 
                            num_step_per_block  * (num_block - 1),))
                v_seq_1 = np.reshape(
                    v_seq_1, (num_block - 1, num_step_per_block, dim_noise))
                x_obs_seq_0, x_obs_seq_1, x_obs_seq_2 = split(
                    x_obs_seq, (num_obs_per_subseq // 2, 
                                num_obs_per_subseq * (num_block - 1),))
                x_obs_seq_1_blk = np.reshape(
                    x_obs_seq_1, (num_block - 1, num_obs_per_subseq, dim_state))
                x_init_1 = np.concatenate(
                    (x_obs_seq_0[-1:], x_obs_seq_1_blk[:-1, -1]), 0)
                x_init_2 = x_obs_seq_1[-1]
                y_tilde_0 = np.concatenate(
                    (y_obs_seq_0_p1[:-1].flatten(), x_obs_seq_0[-1]))
                y_tilde_1 = np.concatenate(
                    (y_obs_seq_1_blk_p1[:, :-1].reshape((num_block - 1, -1)), 
                     x_obs_seq_1_blk[:, -1]), -1)
                y_tilde_2 = y_obs_seq_2_p1.flatten()
            return (
                v_seq_0, v_seq_1, v_seq_2, x_init_1, x_init_2, 
                y_tilde_0, y_tilde_1, y_tilde_2)

        def constr(q, x_obs_seq, partition=0):
            """Calculate constraint function for current partition."""
            u, v_init, v_seq_flat = split(q, (num_param, dim_state,))
            v_seq = np.reshape(v_seq_flat, (-1, dim_noise))
            params = generate_params(u)
            x_init_0 = generate_init_state(v_init, params)
            (v_seq_0, v_seq_1, v_seq_2, x_init_1, x_init_2, 
             y_tilde_0, y_tilde_1, y_tilde_2) = partition_seq(
                 v_seq, x_obs_seq, partition)
            c_0 = generate_obs(params, v_seq_0, x_init_0) - y_tilde_0
            c_1 = vmapped_generate_obs(params, v_seq_1, x_init_1) - y_tilde_1
            c_2 = generate_obs(params, v_seq_2, x_init_2, True) - y_tilde_2
            return np.concatenate([c_0, c_1.flatten(), c_2])
        
        def generate_final_state(params, v_seq, x_init):
            _, x_seq = lax.scan(partial(step_func, params=params), x_init, v_seq)
            return x_seq[-1]
        
        def constr_full(q, x_obs_seq):
            u, v_init, v_seq_flat = split(q, (num_param, dim_state,))
            v_seq_blk = np.reshape(
                v_seq_flat, (num_obs, num_steps_per_obs, dim_noise))
            params = generate_params(u)
            x_init_0 = generate_init_state(v_init, params)
            x_init_blk = np.concatenate((x_init_0[None], x_obs_seq[:-1]), 0)
            return api.vmap(generate_final_state, (None, 0, 0))(
                params, v_seq_blk, x_init_blk) - x_obs_seq
        
        def init_objective(q, x_obs_seq, reg_coeff=1e-2):
            c = constr_full(q, x_obs_seq)
            return 0.5 * np.mean(c**2) + 0.5 * reg_coeff * np.mean(q**2), c
     
        def jacob_constr_blocks(q, x_obs_seq, partition=0):
            """Return non-zero blocks of constraint function Jacobian.
            
            Input state q can be decomposed into q = (u, v0, v1, v2)
            where global latent state (parameters) are determined by u,
            initial sequence block by v0, middle sequence blocks by v1
            and final sequence block v2.
            
            Similarly constraints c can be decompsed as c = (c0, c1, c2)
            where c0 is constraints on initial sequence block, c1 is the
            constraints on the middle sequence blocks and c2 is the
            constraints on the final sequence block.
            
            Constraint Jacobian ∂c/∂q has block structure
            
                ∂c/∂q = ∂(c0, c1, c2)/∂(u, v0, v1, v2)
                      = ((∂c0/∂u, ∂c0/∂v0, 0,       0      )
                         (∂c1/∂u, 0,       ∂c1/∂v1, 0      )
                         (∂c2/∂u, 0,       0,       ∂c2/∂v2))
            
            """
            
            def gen_0(u, v):
                params = generate_params(u)
                v_init, v_seq_flat = split(v, (dim_state,))
                x_init = generate_init_state(v_init, params)
                return generate_obs(
                    params, np.reshape(v_seq_flat, (-1, dim_noise)), x_init)
            
            def gen_1(u, v, x_init):
                params = generate_params(u)
                return generate_obs(
                    params, np.reshape(v, (-1, dim_noise)), x_init)
            
            def gen_2(u, v, x_init):
                params = generate_params(u)
                return generate_obs(
                    params, np.reshape(v, (-1, dim_noise)), x_init, True)                
            
            u, v_init, v_seq_flat = split(q, (num_param, dim_state,))
            v_seq = np.reshape(v_seq_flat, (-1, dim_noise))
            (v_seq_0, v_seq_1, v_seq_2, x_init_1, x_init_2, 
             y_tilde_0, y_tilde_1, y_tilde_2) = partition_seq(
                 v_seq, x_obs_seq, partition)
            v_0 = np.concatenate((v_init, v_seq_0.flatten()))
            dc_0_du, dc_0_dv_0 = api.jacrev(gen_0, (0, 1))(u, v_0)
            v_1 = np.reshape(v_seq_1, (v_seq_1.shape[0], -1))
            dc_1_du, dc_1_dv_1 = api.vmap(
                api.jacrev(gen_1, (0, 1)), (None, 0, 0))(u, v_1, x_init_1)
            v_2 = v_seq_2.flatten()
            dc_2_du, dc_2_dv_2 = api.jacrev(gen_2, (0, 1))(u, v_2, x_init_2)
            return (
                (dc_0_du, dc_1_du, dc_2_du),  (dc_0_dv_0, dc_1_dv_1, dc_2_dv_2))
        
        def chol_gram_blocks(dc_du, dc_dv):
            """Calculate Cholesky factors of decomposition of Gram matrix. """
            if isinstance(metric, IdentityMatrix):
                D = tuple(np.einsum('...ij,...kj', dc_dv[i], dc_dv[i]) 
                          for i in range(3))  
            else:
                m_v = list(split(
                    metric_2_diag, (dc_dv[0].shape[1], 
                                    dc_dv[1].shape[0] * dc_dv[1].shape[2])))
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
            

        def log_det_sqrt_gram_from_chol(chol_C, chol_D):
            """Calculate log-det of Gram matrix from Cholesky factors."""
            return (
                sum(np.log(np.abs(chol_D[i].diagonal(0, -2, -1))).sum() 
                    for i in range(3)) +
                np.log(np.abs(chol_C.diagonal())).sum() - log_det_sqrt_metric_1
            )
        
        
        def log_det_sqrt_gram(q, x_obs_seq, partition=0):
            """Calculate log-determinant of constraint Jacobian Gram matrix."""
            dc_du, dc_dv = jacob_constr_blocks(q, x_obs_seq, partition)
            chol_C, chol_D = chol_gram_blocks(dc_du, dc_dv)
            return (
                log_det_sqrt_gram_from_chol(chol_C, chol_D), 
                ((dc_du, dc_dv), (chol_C, chol_D)))


        def lmult_by_jacob_constr(dc_du, dc_dv, vct):
            """Left-multiply vector by constraint Jacobian matrix."""
            vct_u, vct_v = split(vct, (num_param,))
            j0, j1, j2 = dc_dv[0].shape[1], dc_dv[1].shape[0], dc_dv[2].shape[1]
            return (
                np.vstack((dc_du[0], dc_du[1].reshape((-1, num_param)), 
                           dc_du[2])) @ vct_u +
                np.concatenate((
                    dc_dv[0] @ vct_v[:j0],
                    np.einsum('ijk,ik->ij', dc_dv[1], 
                              np.reshape(vct_v[j0:-j2], (j1, -1))).flatten(),
                    dc_dv[2] @ vct_v[-j2:]
                ))
            )

        def rmult_by_jacob_constr(dc_du, dc_dv, vct):
            """Right-multiply vector by constraint Jacobian matrix."""
            vct_0, vct_1, vct_2 = split(
                vct, (dc_du[0].shape[0], dc_du[1].shape[0] * dc_du[1].shape[1], 
                      dc_du[2].shape[0]))
            vct_1_blocks = np.reshape(vct_1, dc_du[1].shape[:2])
            return np.concatenate([
                dc_du[0].T @ vct_0 + 
                np.einsum('ijk,ij->k', dc_du[1], vct_1_blocks) + 
                dc_du[2].T @ vct_2,
                dc_dv[0].T @ vct_0,
                np.einsum('ijk,ij->ik', dc_dv[1], vct_1_blocks).flatten(),
                dc_dv[2].T @ vct_2
            ])
        
        def lmult_by_inv_gram(dc_du, dc_dv, chol_C, chol_D, vct):
            """Left-multiply vector by inverse Gram matrix."""
            vct = list(split(
                vct, (dc_du[0].shape[0], dc_du[1].shape[0] * dc_du[1].shape[1], 
                      dc_du[2].shape[0])))
            vct[1] = np.reshape(vct[1], dc_du[1].shape[:2])

            D_inv_vct = tuple(
                sla.cho_solve((chol_D[i], True), vct[i]) for i in range(3))
            
            dc_du_T_D_inv_vct = sum(
                np.einsum('...jk,...j->k', dc_du[i], D_inv_vct[i]) 
                for i in range(3))

            C_inv_dc_du_T_D_inv_vct = sla.cho_solve(
                (chol_C, True), dc_du_T_D_inv_vct)
            
            return np.concatenate([
                sla.cho_solve(
                    (chol_D[i], True), 
                    vct[i] - dc_du[i] @ C_inv_dc_du_T_D_inv_vct).flatten()
                for i in range(3)])

        self._generate_x_obs_seq = api.jit(generate_x_obs_seq)
        self._constr = api.jit(constr, (2,))
        self._jacob_constr_blocks = api.jit(jacob_constr_blocks, (2,))
        self._chol_gram_blocks = api.jit(chol_gram_blocks)
        self._log_det_sqrt_gram_from_chol = api.jit(log_det_sqrt_gram_from_chol)
        self._grad_log_det_sqrt_gram = api.jit(
            api.value_and_grad(log_det_sqrt_gram, has_aux=True), (2,))
        self._constr_full = api.jit(constr_full)
        self.value_and_grad_init_objective= api.jit(
            api.value_and_grad(init_objective, (0,), has_aux=True))
        self._lmult_by_jacob_constr = api.jit(lmult_by_jacob_constr)
        self._rmult_by_jacob_constr = api.jit(rmult_by_jacob_constr)
        self._lmult_by_inv_gram = api.jit(lmult_by_inv_gram)
    
    @cache_in_state('pos', 'x_obs_seq', 'partition')
    def constr(self, state):
        return onp.array(
            self._constr(state.pos, state.x_obs_seq, state.partition))
    
    @cache_in_state('pos', 'x_obs_seq', 'partition')
    def jacob_constr_blocks(self, state):
        dc_du, dc_dv = self._jacob_constr_blocks(
            state.pos, state.x_obs_seq, state.partition)
        return (
            tuple(onp.array(block) for block in dc_du),
            tuple(onp.array(block) for block in dc_dv)
        )
    
    @cache_in_state('pos', 'x_obs_seq', 'partition')
    def chol_gram_blocks(self, state):
        dc_du, dc_dv = self.jacob_constr_blocks(state)
        chol_C, chol_D = self._chol_gram_blocks(dc_du, dc_dv)
        return onp.array(chol_C), tuple(onp.array(chol) for chol in chol_D)
    
    @cache_in_state('pos', 'x_obs_seq', 'partition')
    def log_det_sqrt_gram(self, state):
        chol_C, chol_D = self.chol_gram_blocks(state)
        val = self._log_det_sqrt_gram_from_chol(chol_C, chol_D)
        return float(val)
       
    @multi_cache_in_state(['pos', 'x_obs_seq', 'partition'], 
                          ['grad_log_det_sqrt_gram', 'log_det_sqrt_gram', 
                           'jacob_constr_blocks', 'chol_gram_blocks'])
    def grad_log_det_sqrt_gram(self, state):
        (val, ((dc_du, dc_dv), 
               (chol_C, chol_D))), grad = self._grad_log_det_sqrt_gram(
            state.pos, state.x_obs_seq, state.partition)
        return (
            onp.array(grad), float(val), 
            (tuple(onp.array(block) for block in dc_du), 
             tuple(onp.array(block) for block in dc_dv)),
            (onp.array(chol_C), tuple(onp.array(chol) for chol in chol_D))
        )

    def h1(self, state):
        return self.neg_log_dens(state) + self.log_det_sqrt_gram(state)

    def dh1_dpos(self, state):
        return (
            self.grad_neg_log_dens(state) + 
            self.grad_log_det_sqrt_gram(state))
    
    def lmult_by_jacob_constr(self, state, vct):
        dc_du, dc_dv = self.jacob_constr_blocks(state)
        return onp.asarray(self._lmult_by_jacob_constr(dc_du, dc_dv, vct))
    
    def rmult_by_jacob_constr(self, state, vct):
        dc_du, dc_dv = self.jacob_constr_blocks(state)
        return onp.asarray(self._rmult_by_jacob_constr(dc_du, dc_dv, vct))
    
    def lmult_by_inv_gram(self, state, vct):
        dc_du, dc_dv = self.jacob_constr_blocks(state)
        chol_C, chol_D = self.chol_gram_blocks(state)
        return onp.asarray(
            self._lmult_by_inv_gram(dc_du, dc_dv, chol_C, chol_D, vct))
    
    def update_x_obs_seq(self, state):
        state.x_obs_seq = self._generate_x_obs_seq(state.pos)
        
    def project_onto_cotangent_space(self, mom, state):
        mom -= onp.array(
            self.rmult_by_jacob_constr(
                state, self.lmult_by_inv_gram(
                    state, self.lmult_by_jacob_constr(
                        state, self.metric.inv @ mom))))
        return mom

    def sample_momentum(self, state, rng):
        mom = super().sample_momentum(state, rng)
        mom = self.project_onto_cotangent_space(mom, state)
        return mom

    
def no_u_turn_criterion(system, state_1, state_2, sum_mom):
    return (
        np.sum(system.dh_dmom(state_1)[:system.dim_u_v0] * 
               sum_mom[:system.dim_u_v0]) < 0 or
        np.sum(system.dh_dmom(state_2)[:system.dim_u_v0] * 
               sum_mom[:system.dim_u_v0]) < 0)
        
        
class SwitchPartitionTransitionWrapper(Transition):
    """Markov transition that samples a base transition and switches paritition.
    
    The `partition` binary variable in the chain state, which sets the current
    partition used when conditioning on values of the diffusion process at
    intermediate time, is deterministically switched on each transition as well
    as sampling a new state from a base transition.
    """
    
    def __init__(self, system, base_transition):
        self.system = system
        self.base_transition = base_transition
        
    state_variables = {'pos', 'mom', 'partition', 'x_obs_seq', 'dir'}
    
    @property
    def statistic_types(self):
        return self.base_transition.statistic_types
        
    def sample(self, state, rng):
        state_next, trans_stats = self.base_transition.sample(state, rng)
        if state_next is not state:
            self.system.update_x_obs_seq(state_next)
        state_next.partition = 0 if state.partition == 1 else 1
        return state_next, trans_stats
        
        
class ConditionedDiffusionHamiltonianState(ChainState):
    
    def __init__(self, pos, x_obs_seq, partition=0, mom=None, dir=1, 
                 _call_counts=None, _dependencies=None, _cache=None):
        if _call_counts is None:
            _call_counts = {}
        super().__init__(
            pos=pos, x_obs_seq=x_obs_seq, partition=partition,
            mom=mom, dir=dir, _call_counts=_call_counts,
            _dependencies=_dependencies, _cache=_cache)

        
def solve_projection_onto_manifold_quasi_newton(
        state, state_prev, dt, system, 
        convergence_tol=1e-8, position_tol=1e-8,
        divergence_tol=1e10, max_iters=50, norm=maximum_norm):
    mu = onp.zeros_like(state.pos)
    for i in range(max_iters):
        try:
            constr = system.constr(state)
            error = norm(constr)
            delta_mu = system.rmult_by_jacob_constr(
                state_prev, system.lmult_by_inv_gram(state_prev, constr))
            delta_pos = system.metric.inv @ delta_mu
            if error > divergence_tol or onp.isnan(error):
                raise ConvergenceError(
                    f'Quasi-Newton iteration diverged. '
                    f'Last |c|={error:.1e}, |δq|={norm(delta_pos)}.')
            elif error < convergence_tol and norm(delta_pos) < position_tol:
                if state.mom is not None:
                    state.mom -= mu / dt
                return state
            mu += delta_mu
            state.pos -= delta_pos
        except ValueError as e:
            raise ConvergenceError(
                f'ValueError during Quasi-Newton iteration ({e}).')
    raise ConvergenceError(
        f'Quasi-Newton iteration did not converge. '
        f'Last |c|={error:.1e}, |δq|={norm(delta_pos)}.')


def get_initial_state(system, rng, generate_x_obs_seq_init, dim_q, tol, 
                      reg_coeff=5e-2, coarse_tol=1e-1, max_iters=1000):

    # Use optimizers to set optimizer initialization and update functions
    opt_init, opt_update, get_params = opt.adam(2e-1)

    # Define a compiled update step
    @api.jit
    def step(i, opt_state, x_obs_seq_init):
        q, = get_params(opt_state)
        (obj, constr), grad = system.value_and_grad_init_objective(
            q, x_obs_seq_init, reg_coeff)
        opt_state = opt_update(i, grad, opt_state)
        return opt_state, obj, constr

    converged = False
    while not converged:
        q_init = rng.standard_normal(dim_q)
        x_obs_seq_init = generate_x_obs_seq_init(rng)
        opt_state = opt_init((q_init,))
        for i in range(max_iters):
            opt_state_next, norm, constr = step(i, opt_state, x_obs_seq_init)
            if not np.isfinite(norm):
                logging.info('Diverged')
                break
            max_abs_constr = maximum_norm(constr)
            if max_abs_constr < coarse_tol:
                q_init, = get_params(opt_state)
                state = ConditionedDiffusionHamiltonianState(
                    q_init, x_obs_seq=x_obs_seq_init)
                try:
                    state = retract_onto_manifold_quasi_newton(
                        state, state, 1., system, tol)
                except ConvergenceError:
                    logging.info('Quasi-Newton iteration diverged.')
                if np.max(np.abs(system.constr(state))) < tol:
                    converged = True
                    break
            if i % 100 == 0:
                logging.info(f'Iteration {i: >6}: mean|constr|^2 = {norm:.3e} '
                             f'max|constr| = {max_abs_constr:.3e}')
            opt_state = opt_state_next
    return state