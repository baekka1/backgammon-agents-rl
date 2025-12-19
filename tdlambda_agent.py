import jax
import jax.numpy as jnp
import numpy as np
import numba
from numba import njit, types, prange
from functools import partial
import optax

from numba.typed import List

# -----------------------------------------
#  Import your engine functions
# -----------------------------------------
import backgammon_engine as bge   # YOUR ENGINE
from backgammon_value_net import BackgammonValueNet   # YOUR FLAX MODEL
# -----------------------------------------

# 15 planes total:
# 0:  empty (shared)
# 1-7:  current player  (count==1..6, overflow)
# 8-14: opponent        (count==1..6, overflow)

# Hyperparameters
NUM_EPISODES = 1
NUM_GAMES = 4
GAMMA = 0.99
LAMBDA = 0.7
ALPHA = 1e-4

@njit(parallel=True)
def vectorized_is_terminal(state_vector):
    """
    state_vector: (N, 28) array
    returns boolean array of length N
    """
    N = state_vector.shape[0]
    done = np.zeros(N, dtype=np.bool_)

    for i in prange(N):
        s = state_vector[i]
        # White borne off all 15 OR Black borne off all 15
        done[i] = (s[26] == 15) or (s[27] == 15)

    return done

@njit(parallel=True)
def vectorized_reward(state_vector, player_vector):
    """
    Computes rewards for a batch of states from each player's perspective.
    
    state_vector: (N, 28) array of states
    player_vector: (N,) array of +1 or -1 indicating whose perspective
    
    Returns: (N,) float32 rewards
    """
    N = len(state_vector)
    out = np.empty(N, dtype=np.float32)

    for i in prange(N):
        out[i] = bge._reward(state_vector[i], player_vector[i])

    return out

def encode_board(state):
    """
    state: (28,) int8 JAX array
    returns: (24, 9) float32 JAX array
    """

    # Points are state[1:25], shape (24,)
    pts = state[1:25]

    # plane 0: empty
    empty = (pts == 0).astype(jnp.float32)

    w = jnp.maximum(pts, 0)
    b = jnp.maximum(-pts, 0)
    # white planes
    white = pts > 0
    w1 = (pts == 1).astype(jnp.float32) * white
    w2 = (pts == 2).astype(jnp.float32) * white
    w3 = (pts == 3).astype(jnp.float32) * white
    w4 = (pts == 4).astype(jnp.float32) * white
    w5 = (pts == 5).astype(jnp.float32) * white
    w6 = (pts == 6).astype(jnp.float32) * white
    w_over = jnp.maximum(w - 6, 0).astype(jnp.float32) / 9.0

    # black planes
    neg = -pts
    black = pts < 0
    b1 = (neg == 1).astype(jnp.float32) * black
    b2 = (neg == 2).astype(jnp.float32) * black
    b3 = (neg == 3).astype(jnp.float32) * black
    b4 = (neg == 4).astype(jnp.float32) * black
    b5 = (neg == 5).astype(jnp.float32) * black
    b6 = (neg == 6).astype(jnp.float32) * black
    b_over = jnp.maximum(b - 6, 0).astype(jnp.float32) / 9.0

    # stack into (24, 15)
    return jnp.stack([empty,
                      w1, w2, w3, w4, w5, w6, w_over,
                      b1, b2, b3, b4, b5, b6, b_over], axis=1)

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

encode_board_batch = jax.vmap(encode_board)
extract_aux_batch  = jax.vmap(extract_aux)

class BackgammonAgent:
    def __init__(self, optimizer, key_seed=0):
        # RNG
        self.key = jax.random.PRNGKey(key_seed)

        # Network
        self.model = BackgammonValueNet()

        # Dummy inputs for shape inference
        sample_board = jnp.zeros((1, 24, 15), dtype=jnp.float32)
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


def train():
    optimizer = optax.adam(ALPHA)
    agent = BackgammonAgent(optimizer)

    for episode in range(NUM_EPISODES):

        # ---- Initialize batch of games ----
        state_vector, player_vector, dice_vector = bge._vectorized_new_game(NUM_GAMES)
        done = np.zeros(NUM_GAMES, dtype=bool)

        total_loss = 0.0
        step = 0

        while not np.all(done):
            step += 1

            # 1. Select actions (batch)
            move_array = agent.select_action_batch(
                state_vector, player_vector, dice_vector
            )
            # move_array = _decode_move_batch(move_array)

            # 2. Apply actions
            next_state_vector = bge._vectorized_apply_move(
                state_vector, player_vector, move_array
            )

            # 3. Switch players, roll dice
            next_player_vector = -player_vector
            next_dice_vector = bge._vectorized_roll_dice(NUM_GAMES)

            # 4. Compute reward and terminal
            done = vectorized_is_terminal(next_state_vector)

            reward_vector = np.zeros(NUM_GAMES, dtype=np.float32)
            terminal_idx = np.where(done)[0]
            if len(terminal_idx) > 0:
                reward_vector[terminal_idx] = vectorized_reward(
                    next_state_vector[terminal_idx],
                    player_vector[terminal_idx]
                )

            # 5. Encode states for NN
            board_encoded = encode_board_batch(jnp.array(state_vector))
            aux_features = extract_aux_batch(jnp.array(state_vector))

            next_board_encoded = encode_board_batch(jnp.array(next_state_vector))
            next_aux          = extract_aux_batch(jnp.array(next_state_vector))

            # 6. Compute bootstrap value
            bootstrap = agent.batch_forward(agent.params,
                                            next_board_encoded,
                                            next_aux)

            # 7. TD target, respecting terminals
            targets = reward_vector + GAMMA * bootstrap * (~done)

            # 8. TD-λ parameter update
            agent.params, agent.opt_state, agent.grads_trace, preds, loss = \
                agent.td_lambda_update(agent.params,
                                       agent.opt_state,
                                       agent.grads_trace,
                                       board_encoded,
                                       aux_features,
                                       targets,
                                       GAMMA,
                                       LAMBDA)

            total_loss += float(loss)

            # 9. Move to next state
            state_vector = next_state_vector
            player_vector = next_player_vector
            dice_vector = next_dice_vector

        print(f"Episode {episode}: steps={step}  mean_reward={reward_vector.mean():.2f}  loss={total_loss:.4f}")


if __name__ == "__main__":
    train()
