import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import optax

from numba.typed import List

# -----------------------------------------
#  Import your engine functions
# -----------------------------------------
import backgammon_engine as bge   # YOUR ENGINE
from feature_encoding import encode_board_batch, extract_aux_batch
from backgammon_value_net import BackgammonValueNet   # YOUR FLAX MODEL
# -----------------------------------------

class BackgammonAgent:
    def __init__(self, optimizer, key_seed=0):
        # RNG
        self.key = jax.random.PRNGKey(key_seed)

        # Network
        self.model = BackgammonValueNet()

        # Dummy inputs for shape inference
        sample_board = jnp.zeros((1, 24, 9), dtype=jnp.float32)
        sample_aux   = jnp.zeros((1, 6), dtype=jnp.float32)

        # Initialize parameters
        self.params = self.model.init(self.key, sample_board, sample_aux)

        # Optimizer
        self.optimizer = optimizer
        self.opt_state = optimizer.init(self.params)

        # Initialize eligibility traces
        self.grads_trace = jax.tree.map(jnp.zeros_like, self.params)

    # ----------------------------------------------------------
    #   Batch forward pass (JAX/JIT)
    # ----------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def batch_forward(self, params, boards, aux):
        values = self.model.apply(params, boards, aux)   # (N,1)
        return values.squeeze(-1)                       # (N,)

    # ----------------------------------------------------------
    #   TD-lambda update (JAX/JIT)
    # ----------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def td_lambda_update(self, params, opt_state, grads_trace,
                         state_encoded, aux, target, gamma, lam):

        def loss_fn(p):
            preds = self.model.apply(p, state_encoded, aux)
            preds = preds.squeeze()
            td_errors = preds - target                  # (B,)
            loss = 0.5 * jnp.mean(td_errors ** 2)       # <-- scalar
            return loss, preds
            #return 0.5 * (pred - target)**2, pred

        (loss, pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Update trace
        grads_trace = jax.tree.map(
            lambda old, new: gamma * lam * old + new,
            grads_trace, grads
        )

        updates, opt_state = self.optimizer.update(grads_trace, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, grads_trace, pred, loss

    # ----------------------------------------------------------
    #   Adapter function for Numba 2-ply search
    # ----------------------------------------------------------
    def batch_value_function(self, final_states_buffer):
        # Nothing to evaluate
        if len(final_states_buffer) == 0:
            return np.zeros(0, dtype=np.float32)

        # Convert Numba List → numpy → jax
        raw = np.stack(final_states_buffer).astype(np.int32)
        raw = jnp.array(raw)

        boards = encode_board_batch(raw)
        aux    = extract_aux_batch(raw)

        values = self.batch_forward(self.params, boards, aux)
        return np.array(values, dtype=np.float32)


    # ----------------------------------------------------------
    #   Run 2-ply search and choose moves (Numba interface)
    # ----------------------------------------------------------
    def select_action_batch(self, state_vector, player_vector, dice_vector):
        return bge._vectorized_2_ply_search(state_vector,
                                            player_vector,
                                            dice_vector,
                                            self.batch_value_function)
