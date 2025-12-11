import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np

import backgammon_engine as bge

from backgammon_value_net import BackgammonValueNet

optimizer = optax.adam(1e-4)

NUM_EPISODES = 1

@jax.jit
def td_lambda_update(params, opt_state, grads_trace, state_encoded, aux, target_value, gamma, lam):
    # Compute prediction + gradient wrt params
    def loss_fn(p):
        pred = value_fn(p, state_encoded, aux)
        return 0.5 * (pred - target_value)**2, pred

    (loss, pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Update eligibility trace
    grads_trace = jax.tree.map(lambda g_old, g_new: gamma * lam * g_old + g_new,
                               grads_trace, grads)

    # Apply SGD/Adam/etc with eligibility-weighted grads
    updates, opt_state = optimizer.update(grads_trace, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, grads_trace, pred

@jax.jit
def _batch_forward(params, boards, aux):
    # boards: (N, 24, 9)
    # aux:    (N, 6)
    values = model.apply(params, boards, aux)  # output (N,1)
    return values.squeeze(-1)                  # output (N,)

def batch_value_function(final_states_buffer):
    raw = np.stack(final_states_buffer)          # (N, 28)
    raw = jnp.array(raw, dtype=jnp.int32)        # JAX array

    boards = encode_board_batch(raw)             # (N, 24, 9)
    aux    = extract_aux_batch(raw)              # (N, 6)

    values = _batch_forward(params, boards, aux) # (N,)
    return np.array(values, dtype=np.float32)

def encode_board(state):
    """
    state: (28,) int8 JAX array
    returns: (24, 9) float32 JAX array
    """

    # Points are state[1:25], shape (24,)
    pts = state[1:25]

    # plane 0: empty
    empty = (pts == 0).astype(jnp.float32)

    # white planes
    white = pts > 0
    w1 = (pts == 1).astype(jnp.float32) * white
    w2 = (pts == 2).astype(jnp.float32) * white
    w3 = (pts == 3).astype(jnp.float32) * white
    w4 = (pts >= 4).astype(jnp.float32) * white

    # black planes
    neg = -pts
    black = pts < 0
    b1 = (neg == 1).astype(jnp.float32) * black
    b2 = (neg == 2).astype(jnp.float32) * black
    b3 = (neg == 3).astype(jnp.float32) * black
    b4 = (neg >= 4).astype(jnp.float32) * black

    # stack into (24, 9)
    return jnp.stack([empty,
                      w1, w2, w3, w4,
                      b1, b2, b3, b4], axis=1)

def extract_aux(state):
    w_bar = state[0]
    b_bar = state[25]
    w_off = state[26]
    b_off = state[27]

    return jnp.array([
        (w_bar > 0).astype(jnp.float32),
        w_bar / 15.0,
        w_off / 15.0,
        (b_bar > 0).astype(jnp.float32),
        b_bar / 15.0,
        b_off / 15.0
    ], dtype=jnp.float32)


# Vectorize across leading dimension N
encode_board_batch = jax.vmap(encode_board)
extract_aux_batch  = jax.vmap(extract_aux)

rand_key = jax.random.PRNGKey(0)
sample_board = jnp.zeros((1, 24, 9))
sample_aux = jnp.zeros((1, 6))

model = BackgammonValueNet()

params = model.init(rand_key, sample_board, sample_aux)

opt_state = optimizer.init(params)
grads_state = jax.tree.map(jnp.zeros_like, params)

def value_fn(params, board, aux):
    return model.apply(params, board, aux).squeeze()

def main():
    '''
    rand_key = jax.random.PRNGKey(0)
    sample_board = jnp.zeros((1, 24, 9))
    sample_aux = jnp.zeros((1, 6))

    model = BackgammonValueNet()

    params = model.init(rand_key, sample_board, sample_aux)

    opt_state = optimizer.init(params)
    grads_state = jax.tree.map(jnp.zeros_like, params)
    '''

    for episode in range(NUM_EPISODES):
        num_games = 1
        grads_state = jax.tree.map(jnp.zeros_like, params)
        done = np.zeros(num_games, dtype=bool)
        step = 0
        while not np.all(done):
            step += 1
            state_vector, player_vector, dice_vector = bge._vectorized_new_game(num_games)
            print("starting 2-ply search")
            opt_move = bge._vectorized_2_ply_search(state_vector, player_vector,
                                                    dice_vector,
                                                    batch_value_function)
            print("Made it to the optimal move")
            print(opt_move)

if __name__ == "__main__":
    main()
