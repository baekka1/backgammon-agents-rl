import numpy as np
import jax.numpy as jnp
import optax
from numba import njit
from numba.typed import List, Dict

import backgammon_engine as bge
from td_lambda_agent import BackgammonAgent
from feature_encoding import encode_board_batch, extract_aux_batch

# Hyperparameters
NUM_EPISODES = 1
NUM_GAMES = 5
GAMMA = 0.99
LAMBDA = 0.7
ALPHA = 1e-4

@njit
def _decode_move_batch(move_matrix):
    """
    move_matrix: (N, 4, 2)
    Return: List of Numba Lists, length N
    """
    N = move_matrix.shape[0]
    decoded = List()

    for i in range(N):
        arr = move_matrix[i]
        moves = List()
        for j in range(4):
            s = arr[j,0]
            e = arr[j,1]
            if s == -1:
                break
            moves.append((s, e))
        decoded.append(moves)

    return decoded

# with multiple games
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
            move_array = _decode_move_batch(move_array)

            # 2. Apply actions
            next_state_vector = bge._vectorized_apply_move(
                state_vector, player_vector, move_array
            )

            # 3. Switch players, roll dice
            next_player_vector = -player_vector
            next_dice_vector = bge._vectorized_roll_dice(NUM_GAMES)

            # 4. Compute reward and terminal
            done = bge._vectorized_is_terminal(next_state_vector)

            reward_vector = np.zeros(NUM_GAMES, dtype=np.float32)
            terminal_idx = np.where(done)[0]
            if len(terminal_idx) > 0:
                reward_vector[terminal_idx] = bge._vectorized_reward(
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

'''
With one game
def train():
    optimizer = optax.adam(ALPHA)
    agent = BackgammonAgent(optimizer)

    for episode in range(NUM_EPISODES):
        # Start a single game
        state_vector, player_vector, dice_vector = bge._vectorized_new_game(1)

        done = np.array([False])
        total_loss = 0.0
        step = 0

        while not done[0]:
            step += 1

            # ---------------------------
            # 1. Select action with 2-ply search
            # ---------------------------
            move_array = agent.select_action_batch(state_vector,
                                                   player_vector,
                                                   dice_vector)

            move_array = _decode_move_batch(move_array)
            # ---------------------------
            # 2. Apply move in engine
            # ---------------------------
            next_state_vector = bge._vectorized_apply_move(
                state_vector, player_vector, move_array
            )

            # Switch player (Numba engine will provide this soon)
            next_player_vector = -player_vector

            # Roll next dice
            next_dice_vector = bge._vectorized_roll_dice(1)

            # ---------------------------
            # 3. Compute reward + terminal
            # ---------------------------
            done = bge._vectorized_is_terminal(next_state_vector)
            if done[0]:
                reward_vector = bge._vectorized_reward(next_state_vector,
                                                       player_vector)
                reward = reward_vector[0]
            else:
                reward = 0.0

            # ---------------------------
            # 4. Encode states for network
            # ---------------------------
            board_encoded = encode_board_batch(jnp.array(state_vector))
            aux_features = extract_aux_batch(jnp.array(state_vector))

            next_board_encoded = encode_board_batch(jnp.array(next_state_vector))
            next_aux = extract_aux_batch(jnp.array(next_state_vector))

            # ---------------------------
            # 5. TD target
            # ---------------------------
            if done[0]:
                target = reward
            else:
                bootstrap = agent.batch_forward(agent.params,
                                                next_board_encoded,
                                                next_aux)[0]
                target = reward + GAMMA * bootstrap

            # ---------------------------
            # 6. Apply TD-lambda update
            # ---------------------------
            agent.params, agent.opt_state, agent.grads_trace, pred, loss = \
                agent.td_lambda_update(agent.params,
                                       agent.opt_state,
                                       agent.grads_trace,
                                       board_encoded,
                                       aux_features,
                                       target,
                                       GAMMA,
                                       LAMBDA)

            total_loss += float(loss)

            # Move to next state
            state_vector = next_state_vector
            player_vector = next_player_vector
            dice_vector = next_dice_vector

        print(f"Episode {episode}: steps={step} reward={reward} loss={total_loss:.4f}")

'''

if __name__ == "__main__":
    train()
