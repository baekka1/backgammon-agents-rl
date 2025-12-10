# ppo_agent.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Tuple

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

def eval_value_from_root(
    state: np.ndarray,
    root_player: int,
    params: Any,
    apply_fn,
) -> float:
    """
    Evaluate leaf state with the value head from the root player's POV.

    We canonicalize the state with respect to `root_player`, then run the net
    and return the scalar value in [-3,3].
    """
    board_flat_np, aux_np = encode_state_for_net(state, root_player)
    board_flat = jnp.asarray(board_flat_np[None, :])
    aux = jnp.asarray(aux_np[None, :])
    value, _ = apply_fn(params, board_flat, aux)
    return float(value[0])


# ---------- Feature encoding: 24 x 9 planes + 6 aux ----------

NUM_POINTS = int(bge.NUM_POINTS)
W_BAR = int(bge.W_BAR)
B_BAR = int(bge.B_BAR)
W_OFF = int(bge.W_OFF)
B_OFF = int(bge.B_OFF)
NUM_CHECKERS = int(bge.NUM_CHECKERS)


def canonical_state(state: np.ndarray, player: int) -> np.ndarray:
    """
    Use the engine's canonical transform: from the current player's POV,
    they are 'white' moving P1 -> P24.
    """
    return np.array(bge._to_canonical(state, np.int8(player)), dtype=np.int8)


def encode_board_planes(state_canon: np.ndarray) -> np.ndarray:
    """
    state_canon: length-28 int8 array, canonical from 'white to move' POV.

    Returns: board_planes of shape (24, 9) float32.

    Channels:
      0: empty          (1 if no checker on that point)
      1: white blot     (exactly +1 checker)
      2: black blot     (exactly -1 checker)
      3: white made     (exactly +2)
      4: black made     (exactly -2)
      5: white builder  (exactly +3)
      6: black builder  (exactly -3)
      7: white deeper   (>= +4)
      8: black deeper   (<= -4)
    """
    planes = np.zeros((NUM_POINTS, 9), dtype=np.float32)

    for p in range(1, NUM_POINTS + 1):
        n = int(state_canon[p])
        idx = p - 1

        if n == 0:
            planes[idx, 0] = 1.0
        elif n > 0:
            # white checkers
            if n == 1:
                planes[idx, 1] = 1.0
            elif n == 2:
                planes[idx, 3] = 1.0
            elif n == 3:
                planes[idx, 5] = 1.0
            elif n >= 4:
                planes[idx, 7] = 1.0
        else:
            # black checkers
            n_abs = -n
            if n_abs == 1:
                planes[idx, 2] = 1.0
            elif n_abs == 2:
                planes[idx, 4] = 1.0
            elif n_abs == 3:
                planes[idx, 6] = 1.0
            elif n_abs >= 4:
                planes[idx, 8] = 1.0

    return planes


def encode_aux_features(state_canon: np.ndarray) -> np.ndarray:
    """
    6 aux features:
      For 'white' (current player):
        - bar_active_white
        - bar_scale_white  (bar / 15)
        - borne_off_white  (W_OFF / 15)
      For 'black' (opponent):
        - bar_active_black
        - bar_scale_black
        - borne_off_black

    Canonical white checkers are positive, black checkers are negative.
    """
    aux = np.zeros((AUX_INPUT_SIZE,), dtype=np.float32)

    # White (current player)
    white_bar = max(0, int(state_canon[W_BAR]))
    white_off = max(0, int(state_canon[W_OFF]))
    aux[0] = 1.0 if white_bar > 0 else 0.0
    aux[1] = white_bar / float(NUM_CHECKERS)
    aux[2] = white_off / float(NUM_CHECKERS)

    # Black (opponent) – note negative signs in canonical rep
    black_bar = max(0, -int(state_canon[B_BAR]))
    black_off = max(0, -int(state_canon[B_OFF]))
    aux[3] = 1.0 if black_bar > 0 else 0.0
    aux[4] = black_bar / float(NUM_CHECKERS)
    aux[5] = black_off / float(NUM_CHECKERS)

    return aux


def encode_state_for_net(state: np.ndarray, player: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a physical engine state (28,), and current player (+1 or -1),
    produce:
      board_flat: (216,) float32
      aux:        (6,)   float32
    """
    state = np.asarray(state, dtype=np.int8)
    state_canon = canonical_state(state, player)
    planes = encode_board_planes(state_canon)
    aux = encode_aux_features(state_canon)
    return planes.reshape(-1).astype(np.float32), aux.astype(np.float32)


def encode_batch_states_for_net(
    states: np.ndarray, players: np.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    states:  (B, 28) int8
    players: (B,)   int {+1,-1}

    Returns:
      board_flat_batch: (B, 216)
      aux_batch:        (B, 6)
    """
    B = states.shape[0]
    board_batch = np.zeros((B, BOARD_LENGTH * CONV_INPUT_CHANNELS), dtype=np.float32)
    aux_batch = np.zeros((B, AUX_INPUT_SIZE), dtype=np.float32)

    for i in range(B):
        b_flat, a = encode_state_for_net(states[i], int(players[i]))
        board_batch[i] = b_flat
        aux_batch[i] = a

    return jnp.asarray(board_batch), jnp.asarray(aux_batch)


# ---------- PPO config and training state ----------

@dataclass
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
    # params: actor-critic params (value + policy)
    pass


def create_ppo_train_state(
    rng: jax.random.PRNGKey,
    config: PPOConfig,
) -> PPOTrainState:
    """
    Initialize actor-critic net and optimizer.
    """
    model = BackgammonActorCriticNet()
    # Dummy input to initialize shapes
    dummy_board = jnp.zeros((1, BOARD_LENGTH * CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    params = model.init(rng, dummy_board, dummy_aux)

    tx = optax.adam(config.lr)

    return PPOTrainState.create(apply_fn=model.apply, params=params, tx=tx)


# ---------- Action encoding on 25x25 grid ----------

def submove_to_indices(from_point: int, to_point: int, player: int) -> tuple[int, int]:
    """
    Map engine (from_point, to_point, player) to (i_src, i_dst) in [0,24].

    Conventions (per current player):

      src:
        0      -> bar (their bar index)
        1..24  -> physical points 1..24

      dst:
        0..23  -> physical points 1..24
        24     -> their bear-off

    Engine indices:
      W_BAR=0, B_BAR=25, W_OFF=26, B_OFF=27
    """
    # Current player's bar/off indices
    bar_idx = W_BAR if player == 1 else B_BAR
    off_idx = W_OFF if player == 1 else B_OFF

    # --- source index ---
    if from_point == bar_idx:
        i_src = 0
    else:
        # from_point is 1..24
        i_src = from_point  # 1..24

    # --- destination index ---
    if to_point == off_idx:
        i_dst = 24
    else:
        # 1..24 -> 0..23
        i_dst = to_point - 1

    return i_src, i_dst

def indices_to_flat(i_src: int, i_dst: int) -> int:
    """
    Flatten (i_src, i_dst) into single integer in [0, 625).
    """
    return i_src * 25 + i_dst


def flat_to_indices(a: int) -> Tuple[int, int]:
    """
    Inverse of indices_to_flat.
    """
    return a // 25, a % 25

# ---------- GAE and PPO loss ----------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard GAE-Lambda for a single trajectory.

    rewards, values, dones: shape (T,)
    Returns:
      advantages: (T,)
      returns:    (T,)
    """
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

        delta = (
            rewards[t] + gamma * next_value * next_nonterminal - values[t]
        )
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae

    returns = values + adv
    return adv, returns


class PPOMiniBatch(NamedTuple):
    board_flat: jnp.ndarray  # (B, 216)
    aux: jnp.ndarray         # (B, 6)
    actions: jnp.ndarray     # (B,) int32, flat submove indices
    logp_old: jnp.ndarray    # (B,)
    advantages: jnp.ndarray  # (B,)
    returns: jnp.ndarray     # (B,)


def ppo_loss_fn(
    params: Any,
    apply_fn,
    batch: PPOMiniBatch,
    config: PPOConfig,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Clipped PPO objective + value loss + entropy bonus.
    """
    values, policy_logits = apply_fn(
        params, batch.board_flat, batch.aux
    )  # values: (B,), logits: (B,25,25)

    # --- policy loss ---
    logits_flat = policy_logits.reshape(policy_logits.shape[0], -1)  # (B,625)
    log_probs_all = jax.nn.log_softmax(logits_flat, axis=-1)         # (B,625)

    idx = batch.actions  # (B,)
    logp = jnp.take_along_axis(
        log_probs_all, idx[:, None], axis=1
    ).squeeze(-1)

    ratio = jnp.exp(logp - batch.logp_old)  # (B,)
    adv = batch.advantages
    # normalize advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    unclipped = ratio * adv
    clipped = jnp.clip(
        ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps
    ) * adv
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    # --- value loss ---
    value_loss = jnp.mean((batch.returns - values) ** 2)

    # --- entropy bonus ---
    entropy = -jnp.mean(
        jnp.sum(jnp.exp(log_probs_all) * log_probs_all, axis=-1)
    )

    loss = (
        policy_loss
        + config.value_coef * value_loss
        - config.entropy_coef * entropy
    )

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
    (loss, metrics), grads = grad_fn(
        state.params, state.apply_fn, batch, config
    )
    new_state = state.apply_gradients(grads=grads)
    metrics = {k: jax.lax.stop_gradient(v) for k, v in metrics.items()}
    return new_state, metrics


# ---------- Replay buffer structure (simple list-based) ----------

class Transition(NamedTuple):
    state: np.ndarray        # (28,) int8
    player: int              # +1 or -1
    board_flat: np.ndarray   # (216,) float32
    aux: np.ndarray          # (6,) float32
    action: int              # flat submove index [0,625)
    logp: float
    value: float
    reward: float            # from canonical 'white' POV
    done: bool


class ReplayBuffer:
    def __init__(self):
        self.data: List[Transition] = []

    def add(self, tr: Transition):
        self.data.append(tr)

    def clear(self):
        self.data.clear()

    def as_arrays(self) -> Dict[str, np.ndarray]:
        """
        Stack into numpy arrays for a single trajectory / batch.
        """
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


# ---------- Using engine + policy: picking/pruning moves ----------

def top_k_moves_by_policy(
    state: np.ndarray,
    player: int,
    dice: np.ndarray,
    params: Any,
    apply_fn,
    k: int = 5,
) -> Tuple[List[bge.Action], List[np.ndarray]]:
    """
    Return up to K legal moves, ranked by the policy's logits on their first submove.

    state   : engine state (28,)
    player  : +1 or -1 (side to move)
    dice    : np.ndarray shape (2,) int8, sorted as in engine
    params  : actor-critic params
    apply_fn: Flax apply function

    Returns:
      candidate_moves      : list of engine moves (each a list[(from_point, roll)])
      candidate_afterstates: list of resulting states (np.ndarray int8) after applying each move
                             (same length as candidate_moves)
    """
    moves, afterstates = bge._actions(state, player, dice)
    if len(moves) == 0:
        return [], []

    # Featurize from this player's POV
    board_flat_np, aux_np = encode_state_for_net(state, player)
    board_flat = jnp.asarray(board_flat_np[None, :])
    aux = jnp.asarray(aux_np[None, :])

    _, policy_logits = apply_fn(params, board_flat, aux)  # (1,25,25)
    logits_flat = np.array(policy_logits.reshape(-1), dtype=np.float64)  # (625,)

    scores: List[float] = []
    for mv in moves:
        from_point, roll = mv[0]
        to_point = int(bge._get_target_index(from_point, roll, player))
        i_src, i_dst = submove_to_indices(from_point, to_point, player)
        flat_idx = indices_to_flat(i_src, i_dst)
        scores.append(float(logits_flat[flat_idx]))

    # Sort moves by score descending
    idx_sorted = np.argsort(-np.asarray(scores))
    k_eff = min(k, len(moves))
    chosen_moves: List[bge.Action] = []
    chosen_after: List[np.ndarray] = []
    for i in idx_sorted[:k_eff]:
        chosen_moves.append(moves[i])
        chosen_after.append(np.array(afterstates[i], dtype=np.int8))

    return chosen_moves, chosen_after

def pick_action_from_policy(
    rng: np.random.Generator,
    state: np.ndarray,
    player: int,
    dice: np.ndarray,
    params: Any,
    apply_fn,
) -> tuple[bge.Action, int, float, float]:
    """
    Sample a legal move according to the policy over submoves.

    Returns:
      chosen_move       : engine move (list of (from_point, roll))
      action_flat_index : int in [0, 625) for PPO
      logp              : log π(a|s) for that submove (used in PPO)
      value             : V(s) from the network (canonical current-player POV)
    """
    moves, afterstates = bge._actions(state, player, dice)
    if len(moves) == 0:
        # forced pass, no action; treat as no-op
        # value still useful, but no policy term
        board_flat_np, aux_np = encode_state_for_net(state, player)
        board_flat = jnp.asarray(board_flat_np[None, :])
        aux = jnp.asarray(aux_np[None, :])
        value, _ = apply_fn(params, board_flat, aux)
        value = float(value[0])
        return moves, 0, 0.0, value

    # ---- featurize state for net ----
    board_flat_np, aux_np = encode_state_for_net(state, player)
    board_flat = jnp.asarray(board_flat_np[None, :])  # (1,216)
    aux = jnp.asarray(aux_np[None, :])                # (1,6)

    value, policy_logits = apply_fn(params, board_flat, aux)
    value = float(value[0])                            # scalar
    logits_flat = policy_logits.reshape(1, -1)[0]      # (625,) as jnp
    logits_np = np.array(logits_flat, dtype=np.float64)

    # ---- build mask over the 25x25 grid ----
    # legal_mask[i_src, i_dst] = True if that submove is allowed for at least one move
    legal_mask = np.zeros((25, 25), dtype=bool)

    move_first_submove_idx: list[int] = []

    for mv in moves:
        from_point, roll = mv[0]
        to_point = int(bge._get_target_index(from_point, roll, player))
        i_src, i_dst = submove_to_indices(from_point, to_point, player)
        legal_mask[i_src, i_dst] = True
        flat_idx = indices_to_flat(i_src, i_dst)
        move_first_submove_idx.append(flat_idx)

    legal_flat_indices = np.where(legal_mask.reshape(-1))[0]
    if legal_flat_indices.size == 0:
        # should not really happen if moves non-empty
        legal_flat_indices = np.unique(np.array(move_first_submove_idx))

    masked_logits = np.full_like(logits_np, -1e9)
    masked_logits[legal_flat_indices] = logits_np[legal_flat_indices]

    # softmax
    shifted = masked_logits - masked_logits.max()
    probs = np.exp(shifted)
    probs_sum = probs.sum()
    if probs_sum <= 0.0:
        # numeric safety: fall back to uniform over legal
        probs = np.zeros_like(masked_logits)
        probs[legal_flat_indices] = 1.0
        probs_sum = probs.sum()
    probs /= probs_sum

    # sample a submove index
    a = int(rng.choice(len(probs), p=probs))
    logp = float(np.log(probs[a] + 1e-8))

    # among moves whose first submove maps to this submove index, pick one
    candidates = [
        mv for mv, idx in zip(moves, move_first_submove_idx) if idx == a
    ]
    if not candidates:
        chosen_move = moves[0]
    else:
        chosen_move = candidates[rng.integers(len(candidates))]

    return chosen_move, a, logp, value

def two_ply_value_pruned(
    state: np.ndarray,
    root_player: int,
    dice_root: np.ndarray,
    params: Any,
    apply_fn,
    k_root: int = 5,
) -> float:
    """
    Approximate V(S) with a 2-ply search:

      - At the root (player = root_player, dice_root), use the policy to pick
        the top-k_root moves.
      - For each candidate root move, consider all 21 sorted opponent dice.
      - For each opponent roll, enumerate ALL legal opponent moves, and take
        the worst (minimum) value from the root player's POV.
      - Return the maximum expected value over root's candidate moves.

    This mirrors Agent 1/2's logic but with policy-guided pruning at the root.
    """
    # Root's candidate moves (pruned)
    candidate_moves, candidate_afterstates = top_k_moves_by_policy(
        state, root_player, dice_root, params, apply_fn, k=k_root
    )

    if len(candidate_moves) == 0:
        # Forced pass or no legal moves: just evaluate current state
        return eval_value_from_root(state, root_player, params, apply_fn)

    best_value = -np.inf

    for S1 in candidate_afterstates:
        # Evaluate opponent response from state S1
        opp = -root_player
        expected = 0.0

        d_idx = 0
        for r1 in range(1, 7):
            for r2 in range(1, r1 + 1):
                dice_opp = np.array([r1, r2], dtype=np.int8)
                p_roll = 1.0 / 36.0 if r1 == r2 else 2.0 / 36.0

                opp_moves, opp_after = bge._actions(S1, opp, dice_opp)
                if len(opp_moves) == 0:
                    # opponent forced pass: leaf is S1
                    leaf_val = eval_value_from_root(S1, root_player, params, apply_fn)
                    expected += p_roll * leaf_val
                    continue

                # Full opponent search (no pruning) but you could apply top_k_moves_by_policy
                worst_val = np.inf
                for S2 in opp_after:
                    v = eval_value_from_root(np.array(S2, dtype=np.int8), root_player, params, apply_fn)
                    if v < worst_val:
                        worst_val = v

                expected += p_roll * worst_val
                d_idx += 1

        if expected > best_value:
            best_value = expected

    return best_value

# ---------- One PPO rollout over many steps ----------

def rollout_episode_ppo(
    rng: np.random.Generator,
    train_state: PPOTrainState,
    max_steps: int = 2048,
) -> Tuple[ReplayBuffer, float]:
    """
    Generate one on-policy trajectory via self-play, storing both players'
    transitions (canonicalized) in the buffer.

    - Actions are sampled from the policy over submoves (via pick_action_from_policy).
    - State values V(S_t) are estimated using a 2-ply, policy-pruned search
      (two_ply_value_pruned) from the current player's perspective.
    """
    rb = ReplayBuffer()
    total_reward_white = 0.0

    # Start new game
    player, dice, state = bge._new_game()
    state = np.array(state, dtype=np.int8)
    player = int(player)  # +1 or -1

    for t in range(max_steps):
        # Check for terminal before acting
        reward_white = float(bge._reward(state, bge.Player(1)))
        done = (reward_white != 0.0)

        if done:
            total_reward_white += reward_white
            break

            # Check legal moves
        moves, _ = bge._actions(state, player, dice)
        if len(moves) == 0:
            # Forced pass: no action taken
            player = -player
            dice = bge._roll_dice()
            continue

        # --------- Choose action from policy ---------
        # Sample a legal move according to the policy over submoves
        move, action_flat, logp, _ = pick_action_from_policy(
            rng, state, player, dice, train_state.params, train_state.apply_fn
        )

        # Featurize current state from this player's POV for storage
        board_flat_np, aux_np = encode_state_for_net(state, player)

        # Use pruned 2-ply search to estimate V(S_t) from this player's POV
        v_two_ply = two_ply_value_pruned(
            state=state,
            root_player=player,
            dice_root=dice,
            params=train_state.params,
            apply_fn=train_state.apply_fn,
            k_root=5,  # can tune this
        )

        # --------- Environment step ---------
        next_state = bge._apply_move(state, player, move)
        next_state = np.array(next_state, dtype=np.int8)
        next_player = -player
        next_dice = bge._roll_dice()

        # Reward from WHITE's POV after the move
        reward_white = float(bge._reward(next_state, bge.Player(1)))
        done = (reward_white != 0.0)

        # Convert to reward from *current* player's canonical POV
        reward_canon = reward_white * float(player)

        # Store transition
        tr = Transition(
            state=state.copy(),
            player=player,
            board_flat=board_flat_np,
            aux=aux_np,
            action=action_flat,
            logp=logp,
            value=v_two_ply,
            reward=reward_canon,
            done=done,
        )
        rb.add(tr)

        # Advance
        state = next_state
        player = next_player
        dice = next_dice

        if done:
            total_reward_white += reward_white
            # For multi-episode rollouts, you could restart a new game here
            break

    return rb, total_reward_white

# ---------- PPO training loop over one buffer ----------

def ppo_update_from_buffer(
    train_state: PPOTrainState,
    config: PPOConfig,
    rb: ReplayBuffer,
) -> Tuple[PPOTrainState, Dict[str, float]]:
    """
    Run multiple PPO epochs over a single replay buffer.
    """
    arrs = rb.as_arrays()
    rewards = arrs["rewards"]
    values = arrs["values"]
    dones = arrs["dones"]

    advantages, returns = compute_gae(
        rewards, values, dones, config.gamma, config.lam
    )

    # Prepare big batch as jnp
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

    for epoch in range(config.num_epochs):
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
    """
    Simple training driver for PPO Agent 3.

    - Runs `num_episodes` self-play episodes.
    - After each episode, runs PPO updates over that episode's buffer.
    - Prints basic sanity metrics.
    """
    # --- RNG setup ---
    rng = np.random.default_rng(seed)
    rng_jax = jax.random.PRNGKey(seed)

    # --- PPO config & state ---
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

    print("Starting PPO training for Agent 3...")
    print(f"num_episodes={num_episodes}, max_steps_per_episode={max_steps_per_episode}")

    for ep in range(num_episodes):
        # ---- Rollout one on-policy episode ----
        rb, total_reward_white = rollout_episode_ppo(
            rng=rng,
            train_state=train_state,
            max_steps=max_steps_per_episode,
        )

        ep_len = len(rb.data)

        # Sanity: skip empty buffer (shouldn't usually happen)
        if ep_len == 0:
            print(f"[Episode {ep}] Empty buffer, skipping PPO update.")
            continue

        # ---- PPO update over this buffer ----
        train_state, metrics = ppo_update_from_buffer(
            train_state=train_state,
            config=config,
            rb=rb,
        )

        # ---- Basic logging ----
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
        num_episodes=1,          # start small
        max_steps_per_episode=16 # keep this modest at first
    )
