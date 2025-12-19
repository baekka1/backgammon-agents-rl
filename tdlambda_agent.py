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
NUM_EPISODES = 5
NUM_GAMES = 5
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
        self.grads_trace = None

    def reset_traces(self, batch_size: int):
        # trace pytree with leading batch dimension: (B, *param.shape)
        self.grads_trace = jax.tree.map(
            lambda p: jnp.zeros((batch_size,) + p.shape, dtype=p.dtype),
            self.params
        )

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
    def td_lambda_update(self, params, opt_state, z,
                         boards, aux, targets, active_mask,
                         gamma, lam):
        """
        Semi-gradient TD(λ) with eligibility traces.

        z: pytree with leading batch dim (B, ...)
        active_mask: (B,) bool, True for games that are still running THIS step
                     (use ~done from start of loop)
        """

        B = boards.shape[0]

        # value function for a SINGLE sample (board: (24,15), aux: (6,))
        def v_fn(p, b, a):
            return self.model.apply(p, b[None, ...], a[None, ...]).squeeze()

        # per-sample value + grad: v_t (B,), g_t pytree with leading B
        (v_t, g_t) = jax.vmap(jax.value_and_grad(v_fn), in_axes=(None, 0, 0))(params, boards, aux)

        # TD error per sample
        delta = targets - v_t  # (B,)

        # Mask out finished games (don’t change their traces/updates)
        active = active_mask.astype(jnp.float32)  # (B,)

        # z_t = γλ z_{t-1} + g_t   (only for active games)
        z = jax.tree.map(
            lambda z_leaf, g_leaf: (gamma * lam) * z_leaf + g_leaf,
            z, g_t
        )
        z = jax.tree.map(
            lambda z_leaf: z_leaf * active.reshape((B,) + (1,) * (z_leaf.ndim - 1)),
            z
        )

        # Optax does gradient descent, but we want: w <- w + α * mean(delta * z)
        # So pass grad = - mean(delta * z)
        delta_bc = delta.astype(jnp.float32) * active  # zero out inactive
        grads_for_opt = jax.tree.map(
            lambda z_leaf: -jnp.mean(
                z_leaf * delta_bc.reshape((B,) + (1,) * (z_leaf.ndim - 1)),
                axis=0
            ),
            z
        )

        updates, opt_state = self.optimizer.update(grads_for_opt, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        loss = 0.5 * jnp.mean((delta_bc) ** 2)  # just for logging

        return params, opt_state, z, v_t, loss
    '''
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
    '''
    # ----------------------------------------------------------
    #   Adapter function for Numba 2-ply search
    # ----------------------------------------------------------
    '''
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
    '''
    def batch_value_function(self, final_states_buffer, chunk_size=2048):
        if len(final_states_buffer) == 0:
            return np.zeros(0, dtype=np.float32)

        raw = np.stack(final_states_buffer).astype(np.int8)
        L = raw.shape[0]

        out = np.empty((L,), dtype=np.float32)

        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            batch = jnp.asarray(raw[start:end])

            boards = encode_board_batch(batch)
            aux    = extract_aux_batch(batch)

            vals = self.batch_forward(self.params, boards, aux)  # (chunk,)
            out[start:end] = np.asarray(vals, dtype=np.float32)

        return out

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
        agent.reset_traces(NUM_GAMES)

        total_loss = 0.0
        step = 0

        while not np.all(done):
            step += 1
            if step % 50 == 0:
                print(
                    f"[ep {episode}] step={step} "
                    f"done={done.sum()}/{NUM_GAMES} "
                    f"avg_abs_pts={np.mean(np.abs(state_vector[:,1:25])):.2f} "
                    f"mean_off_w={state_vector[:,26].mean():.2f} "
                    f"mean_off_b={(-state_vector[:,27]).mean():.2f}"
                )
            active_mask = ~done 

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
            '''
            agent.params, agent.opt_state, agent.grads_trace, preds, loss = \
                agent.td_lambda_update(agent.params,
                                       agent.opt_state,
                                       agent.grads_trace,
                                       board_encoded,
                                       aux_features,
                                       targets,
                                       GAMMA,
                                       LAMBDA)
            '''
            agent.params, agent.opt_state, agent.grads_trace, preds, loss = agent.td_lambda_update(
                agent.params,
                agent.opt_state,
                agent.grads_trace,
                board_encoded,
                aux_features,
                jnp.asarray(targets, dtype=jnp.float32),
                jnp.asarray(active_mask, dtype=jnp.bool_),
                GAMMA,
                LAMBDA
            )
            total_loss += float(loss)

            # 9. Move to next state
            state_vector = next_state_vector
            player_vector = next_player_vector
            dice_vector = next_dice_vector

        print(f"Episode {episode}: steps={step}  mean_reward={reward_vector.mean():.2f}  loss={total_loss:.4f}")
    return agent

def is_terminal(state: np.ndarray) -> bool:
    # black borne-off is negative in your engine
    return (state[bge.W_OFF] == bge.NUM_CHECKERS) or (state[bge.B_OFF] == -bge.NUM_CHECKERS)

def simulate_game(agent, max_turns=2000, verbose=True, seed=0):
    np.random.seed(seed)

    player, dice, state = bge._new_game()  # player in {+1,-1}, dice shape (2,), state shape (28,)

    history = []  # (turn, player, dice, move, reward)
    turn = 0

    while (not is_terminal(state)) and (turn < max_turns):
        turn += 1

        # Use the already-rolled dice from _new_game on turn 1, otherwise roll new dice
        if turn > 1:
            dice = bge._roll_dice()

        # Vectorized interface with batch size 1
        state_vec  = np.expand_dims(state, axis=0).astype(np.int8)      # (1, 28)
        player_vec = np.array([player], dtype=np.int8)                  # (1,)
        dice_vec   = np.expand_dims(dice, axis=0).astype(np.int8)       # (1, 2)

        move_list = agent.select_action_batch(state_vec, player_vec, dice_vec)
        move = move_list[0]  # the Action for this single game

        next_state = bge._apply_move(state, player, move)
        r = bge._reward(next_state, player)

        history.append((turn, int(player), tuple(map(int, dice)), move, float(r)))

        if verbose:
            print(f"turn={turn:4d} player={int(player):+d} dice={tuple(map(int,dice))} "
                  f"move_len={len(move)} reward={r}")

        state = next_state
        if r != 0:
            break

        player = np.int8(-player)

    if turn >= max_turns:
        print(f"Hit max_turns={max_turns} without termination (likely a bug).")

    # Final winner info
    if state[bge.W_OFF] == bge.NUM_CHECKERS:
        winner = +1
    elif state[bge.B_OFF] == -bge.NUM_CHECKERS:
        winner = -1
    else:
        winner = 0

    return winner, state, history

if __name__ == "__main__":
    agent = train()
    winner, final_state, history = simulate_game(agent, verbose=True, seed=42)
    print("winner:", winner)
