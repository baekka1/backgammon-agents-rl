# ppo_agent.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Tuple, Callable

import numpy as np
import jax
import jax.numpy as jnp
import functools
from flax.training.train_state import TrainState
import optax

import backgammon_engine as bge
from backgammon_value_net import (
    BackgammonActorCriticNet,
    BOARD_LENGTH,
    CONV_INPUT_CHANNELS,
    AUX_INPUT_SIZE,
)

# ---------- Feature encoding: 24 x 15 planes + 6 aux ----------

NUM_POINTS = int(bge.NUM_POINTS)
W_BAR = int(bge.W_BAR)
B_BAR = int(bge.B_BAR)
W_OFF = int(bge.W_OFF)
B_OFF = int(bge.B_OFF)
NUM_CHECKERS = int(bge.NUM_CHECKERS)


def canonical_state(state: np.ndarray, player: int) -> np.ndarray:
    return np.array(bge._to_canonical(state, np.int8(player)), dtype=np.int8)

def encode_board_planes(state_canon: np.ndarray) -> np.ndarray:
    planes = np.zeros((NUM_POINTS, CONV_INPUT_CHANNELS), dtype=np.float32)

    EMPTY_CH = 0
    WHITE_BASE = 1
    BLACK_BASE = 7
    WHITE_EX = 13
    BLACK_EX = 14

    for p in range(1, NUM_POINTS + 1):
        n = int(state_canon[p])
        idx = p - 1

        if n == 0:
            planes[idx, EMPTY_CH] = 1.0
            continue

        if n > 0:
            count = n
            if count <= 6:
                planes[idx, WHITE_BASE + (count - 1)] = 1.0
            else:
                planes[idx, WHITE_BASE + 5] = 1.0
                planes[idx, WHITE_EX] = min(count - 6, 4) / 4.0
        else:
            count = -n
            if count <= 6:
                planes[idx, BLACK_BASE + (count - 1)] = 1.0
            else:
                planes[idx, BLACK_BASE + 5] = 1.0
                planes[idx, BLACK_EX] = min(count - 6, 4) / 4.0

    return planes

def encode_aux_features(state_canon: np.ndarray) -> np.ndarray:
    aux = np.zeros((AUX_INPUT_SIZE,), dtype=np.float32)

    white_bar = max(0, int(state_canon[W_BAR]))
    white_off = max(0, int(state_canon[W_OFF]))
    aux[0] = 1.0 if white_bar > 0 else 0.0
    aux[1] = white_bar / float(NUM_CHECKERS)
    aux[2] = white_off / float(NUM_CHECKERS)

    black_bar = max(0, -int(state_canon[B_BAR]))
    black_off = max(0, -int(state_canon[B_OFF]))
    aux[3] = 1.0 if black_bar > 0 else 0.0
    aux[4] = black_bar / float(NUM_CHECKERS)
    aux[5] = black_off / float(NUM_CHECKERS)

    return aux

def encode_state_for_net(state: np.ndarray, player: int) -> Tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.int8)
    state_canon = canonical_state(state, player)
    planes = encode_board_planes(state_canon)
    aux = encode_aux_features(state_canon)
    return planes.reshape(-1).astype(np.float32), aux.astype(np.float32)

def encode_batch_states_for_net(
    states: np.ndarray, players: np.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    B = states.shape[0]
    board_batch = np.zeros((B, BOARD_LENGTH * CONV_INPUT_CHANNELS), dtype=np.float32)
    aux_batch = np.zeros((B, AUX_INPUT_SIZE), dtype=np.float32)

    for i in range(B):
        b_flat, a = encode_state_for_net(states[i], int(players[i]))
        board_batch[i] = b_flat
        aux_batch[i] = a

    return jnp.asarray(board_batch), jnp.asarray(aux_batch)

# ---------- PPO config and training state ----------

@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    num_epochs: int = 4
    minibatch_size: int = 256

class PPOTrainState(TrainState):
    pass

def create_ppo_train_state(
    rng: jax.random.PRNGKey,
    config: PPOConfig,
) -> PPOTrainState:
    model = BackgammonActorCriticNet()
    dummy_board = jnp.zeros((1, BOARD_LENGTH * CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    params = model.init(rng, dummy_board, dummy_aux)

    tx = optax.adam(config.lr)
    return PPOTrainState.create(apply_fn=model.apply, params=params, tx=tx)

# ---------- Action encoding on 25x25 grid ----------

def submove_to_indices(from_point: int, to_point: int, player: int) -> tuple[int, int]:
    bar_idx = W_BAR if player == 1 else B_BAR
    off_idx = W_OFF if player == 1 else B_OFF

    i_src = 0 if from_point == bar_idx else from_point
    i_dst = 24 if to_point == off_idx else (to_point - 1)
    return i_src, i_dst

def indices_to_flat(i_src: int, i_dst: int) -> int:
    return i_src * 25 + i_dst

# ---------- GAE and PPO loss ----------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - float(dones[t])
            next_value = 0.0
        else:
            next_nonterminal = 1.0 - float(dones[t + 1])
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae

    returns = values + adv
    return adv, returns

class PPOMiniBatch(NamedTuple):
    board_flat: jnp.ndarray
    aux: jnp.ndarray
    actions: jnp.ndarray
    logp_old: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray

def ppo_loss_fn(
    params: Any,
    apply_fn,
    batch: PPOMiniBatch,
    config: PPOConfig,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    values, policy_logits = apply_fn(params, batch.board_flat, batch.aux)

    logits_flat = policy_logits.reshape(policy_logits.shape[0], -1)  # (B,625)
    log_probs_all = jax.nn.log_softmax(logits_flat, axis=-1)

    idx = batch.actions
    logp = jnp.take_along_axis(log_probs_all, idx[:, None], axis=1).squeeze(-1)

    ratio = jnp.exp(logp - batch.logp_old)
    adv = batch.advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    unclipped = ratio * adv
    clipped = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    value_loss = jnp.mean((batch.returns - values) ** 2)

    entropy = -jnp.mean(jnp.sum(jnp.exp(log_probs_all) * log_probs_all, axis=-1))

    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
    }
    return loss, metrics

@functools.partial(jax.jit, static_argnames=("config",))
def ppo_update_step(
    state: PPOTrainState,
    batch: PPOMiniBatch,
    config: PPOConfig,
) -> Tuple[PPOTrainState, Dict[str, jnp.ndarray]]:
    grad_fn = jax.value_and_grad(ppo_loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params, state.apply_fn, batch, config)
    new_state = state.apply_gradients(grads=grads)
    metrics = {k: jax.lax.stop_gradient(v) for k, v in metrics.items()}
    return new_state, metrics

# ---------- Replay buffer ----------

class Transition(NamedTuple):
    state: np.ndarray
    player: int
    board_flat: np.ndarray
    aux: np.ndarray
    action: int
    logp: float
    value: float
    reward: float
    done: bool

class ReplayBuffer:
    def __init__(self):
        self.data: List[Transition] = []

    def add(self, tr: Transition):
        self.data.append(tr)

    def clear(self):
        self.data.clear()

    def as_arrays(self) -> Dict[str, np.ndarray]:
        states = np.stack([t.state for t in self.data], axis=0)
        players = np.asarray([t.player for t in self.data], dtype=np.int32)
        board_flat = np.stack([t.board_flat for t in self.data], axis=0)
        aux = np.stack([t.aux for t in self.data], axis=0)
        actions = np.asarray([t.action for t in self.data], dtype=np.int32)
        logp = np.asarray([t.logp for t in self.data], dtype=np.float32)
        values = np.asarray([t.value for t in self.data], dtype=np.float32)
        rewards = np.asarray([t.reward for t in self.data], dtype=np.float32)
        dones = np.asarray([t.done for t in self.data], dtype=bool)

        return dict(
            states=states,
            players=players,
            board_flat=board_flat,
            aux=aux,
            actions=actions,
            logp=logp,
            values=values,
            rewards=rewards,
            dones=dones,
        )

# ---------- 2-ply move choice with GPU-batched value eval ----------

def make_batch_value_function(
    params: Any,
    apply_fn,
    root_player: int,
):
    """
    Returns a batch_value_function compatible with bge._2_ply_search.

    Uses bge._hashing_batch_value_function to deduplicate
    leaf states before GPU evaluation.
    """

    def eval_unique_states(states_list) -> np.ndarray:
        n = len(states_list)
        if n == 0:
            return np.empty((0,), dtype=np.float32)

        # list-of-tuples -> (n,28) np.int8
        states_np = np.empty((n, 28), dtype=np.int8)
        for i in range(n):
            states_np[i, :] = np.asarray(states_list[i], dtype=np.int8)

        players_np = np.full((n,), int(root_player), dtype=np.int32)

        # ---- CHUNKED GPU EVAL ----
        CHUNK = 4096
        out = np.empty((n,), dtype=np.float32)

        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            board_flat, aux = encode_batch_states_for_net(states_np[s:e], players_np[s:e])
            v, _ = apply_fn(params, board_flat, aux)      # (chunk,)
            out[s:e] = np.asarray(jax.device_get(v), dtype=np.float32)

        return out

    def batch_value_function(states_buffer) -> np.ndarray:
        """
        Full leaf buffer -> dedup -> GPU eval -> expand back.
        """
        return bge._hashing_batch_value_function(eval_unique_states, states_buffer)

    return batch_value_function

def choose_move_via_2ply_and_score_policy(
    rng: np.random.Generator,
    state: np.ndarray,
    player: int,
    dice: np.ndarray,
    params: Any,
    apply_fn,
) -> tuple[bge.Action, int, float, float]:
    """
    1) Choose the *engine move* via 2-ply minimax using bge._2_ply_search
       with a GPU-batched value function.
    2) Compute (action_flat, logp, v_net) under the current policy/value net
       for PPO bookkeeping (first-submove encoding).
    """
    # If forced pass, treat as no-op
    moves, _ = bge._actions(state, player, dice)

    # Engine represents forced pass as a single empty move: [ [] ]
    if len(moves) == 0 or (len(moves) == 1 and len(moves[0]) == 0):
        board_flat_np, aux_np = encode_state_for_net(state, player)
        board_flat = jnp.asarray(board_flat_np[None, :])
        aux = jnp.asarray(aux_np[None, :])
        v_net, _ = apply_fn(params, board_flat, aux)
        return moves[0] if len(moves) else moves, 0, 0.0, float(v_net[0])

    # --- 2-ply move selection (tree in numba; leaves valued on GPU) ---
    batch_vf = make_batch_value_function(params, apply_fn, root_player=player)
    chosen_move = bge._2_ply_search(state, player, dice, batch_vf)

    # --- PPO bookkeeping: compute logp of chosen move's first submove ---
    # Evaluate policy/value ONCE for this state
    board_flat_np, aux_np = encode_state_for_net(state, player)
    board_flat = jnp.asarray(board_flat_np[None, :])
    aux = jnp.asarray(aux_np[None, :])
    v_net, policy_logits = apply_fn(params, board_flat, aux)
    v_net = float(v_net[0])

    logits_flat = policy_logits.reshape(1, -1)[0]  # (625,)

    # Build legal first-submove mask (CPU) and first-submove index of chosen_move
    legal_mask = np.zeros((25, 25), dtype=bool)

    move_first_idx: Dict[int, None] = {}
    for mv in moves:
        if len(mv) == 0:
            continue  # skip no-op move for policy bookkeeping
        fp, roll = mv[0]
        tp = int(bge._get_target_index(fp, roll, player))
        i_src, i_dst = submove_to_indices(fp, tp, player)
        legal_mask[i_src, i_dst] = True
        move_first_idx[indices_to_flat(i_src, i_dst)] = None

    # chosen move first submove
    fp, roll = chosen_move[0]
    tp = int(bge._get_target_index(fp, roll, player))
    i_src, i_dst = submove_to_indices(fp, tp, player)
    action_flat = indices_to_flat(i_src, i_dst)

    legal_flat_indices = np.where(legal_mask.reshape(-1))[0]
    if legal_flat_indices.size == 0:
        legal_flat_indices = np.array(list(move_first_idx.keys()), dtype=np.int32)

    # Mask + softmax on GPU
    legal_idx = jnp.asarray(legal_flat_indices, dtype=jnp.int32)
    masked = jnp.full((625,), -1e9, dtype=logits_flat.dtype)
    masked = masked.at[legal_idx].set(logits_flat[legal_idx])
    probs = jax.nn.softmax(masked)

    probs_host = np.asarray(jax.device_get(probs), dtype=np.float64)
    logp = float(np.log(probs_host[action_flat] + 1e-8))

    return chosen_move, int(action_flat), logp, v_net

# ---------- One PPO rollout (2-ply move choice) ----------

def rollout_episode_ppo(
    rng: np.random.Generator,
    train_state: PPOTrainState,
    max_steps: int = 2048,
) -> Tuple[ReplayBuffer, float]:
    rb = ReplayBuffer()
    total_reward_white = 0.0

    player, dice, state = bge._new_game()
    state = np.array(state, dtype=np.int8)
    player = int(player)

    for _t in range(max_steps):
        reward_white = float(bge._reward(state, bge.Player(1)))
        done = (reward_white != 0.0)
        if done:
            total_reward_white += reward_white
            break

        moves, _ = bge._actions(state, player, dice)

        # Treat forced pass as no-op turn: advance player/dice, but DO NOT add a transition
        if len(moves) == 0 or (len(moves) == 1 and len(moves[0]) == 0):
            player = -player
            dice = bge._roll_dice()
            continue

        # 2-ply chooses the move; compute policy logp/value for PPO
        move, action_flat, logp, v_net = choose_move_via_2ply_and_score_policy(
            rng, state, player, dice, train_state.params, train_state.apply_fn
        )

        board_flat_np, aux_np = encode_state_for_net(state, player)

        next_state = bge._apply_move(state, player, move)
        next_state = np.array(next_state, dtype=np.int8)
        next_player = -player
        next_dice = bge._roll_dice()

        reward_white_next = float(bge._reward(next_state, bge.Player(1)))
        done = (reward_white_next != 0.0)

        reward_canon = reward_white_next * float(player)

        rb.add(
            Transition(
                state=state.copy(),
                player=player,
                board_flat=board_flat_np,
                aux=aux_np,
                action=action_flat,
                logp=logp,
                value=float(v_net),
                reward=reward_canon,
                done=done,
            )
        )

        state = next_state
        player = next_player
        dice = next_dice

        if done:
            total_reward_white += reward_white_next
            break

    return rb, total_reward_white


# ---------- PPO update over one buffer ----------

def ppo_update_from_buffer(
    train_state: PPOTrainState,
    config: PPOConfig,
    rb: ReplayBuffer,
) -> Tuple[PPOTrainState, Dict[str, float]]:
    arrs = rb.as_arrays()
    rewards = arrs["rewards"]
    values = arrs["values"]
    dones = arrs["dones"]

    advantages, returns = compute_gae(rewards, values, dones, config.gamma, config.lam)

    board_flat = jnp.asarray(arrs["board_flat"], dtype=jnp.float32)
    aux = jnp.asarray(arrs["aux"], dtype=jnp.float32)
    actions = jnp.asarray(arrs["actions"], dtype=jnp.int32)
    logp_old = jnp.asarray(arrs["logp"], dtype=jnp.float32)
    advantages = jnp.asarray(advantages, dtype=jnp.float32)
    returns = jnp.asarray(returns, dtype=jnp.float32)

    N = board_flat.shape[0]
    idx_all = np.arange(N)

    metrics_accum: Dict[str, float] = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
    }
    num_updates = 0

    for _epoch in range(config.num_epochs):
        np.random.shuffle(idx_all)
        for start in range(0, N, config.minibatch_size):
            end = min(start + config.minibatch_size, N)
            mb_idx = idx_all[start:end]

            mb = PPOMiniBatch(
                board_flat=board_flat[mb_idx],
                aux=aux[mb_idx],
                actions=actions[mb_idx],
                logp_old=logp_old[mb_idx],
                advantages=advantages[mb_idx],
                returns=returns[mb_idx],
            )

            train_state, metrics = ppo_update_step(train_state, mb, config)
            num_updates += 1
            for k in metrics_accum:
                metrics_accum[k] += float(metrics[k])

    if num_updates > 0:
        for k in metrics_accum:
            metrics_accum[k] /= num_updates

    return train_state, metrics_accum

def train_ppo_agent(
    num_episodes: int = 50,
    max_steps_per_episode: int = 512,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    rng_jax = jax.random.PRNGKey(seed)

    config = PPOConfig(
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        lr=3e-4,
        num_epochs=4,
        minibatch_size=256,
    )
    train_state = create_ppo_train_state(rng_jax, config)

    print("Starting PPO training for Agent 3 (2-ply move choice)...")
    print(f"num_episodes={num_episodes}, max_steps_per_episode={max_steps_per_episode}")

    for ep in range(num_episodes):
        rb, total_reward_white = rollout_episode_ppo(
            rng=rng,
            train_state=train_state,
            max_steps=max_steps_per_episode,
        )

        ep_len = len(rb.data)
        if ep_len == 0:
            print(f"[Episode {ep}] Empty buffer, skipping PPO update.")
            continue

        train_state, metrics = ppo_update_from_buffer(train_state, config, rb)

        avg_value = float(np.mean([tr.value for tr in rb.data]))
        avg_reward = float(np.mean([tr.reward for tr in rb.data]))

        print(
            f"[Episode {ep:03d}] "
            f"len={ep_len:4d}  "
            f"white_return={total_reward_white:+.3f}  "
            f"avg_reward(canon)={avg_reward:+.3f}  "
            f"avg_value={avg_value:+.3f}  "
            f"loss={metrics['loss']:.4f}  "
            f"policy_loss={metrics['policy_loss']:.4f}  "
            f"value_loss={metrics['value_loss']:.4f}  "
            f"entropy={metrics['entropy']:.4f}"
        )

    print("Training finished.")

if __name__ == "__main__":
    train_ppo_agent(
        num_episodes=1,
        max_steps_per_episode=64,
    )
