# ppo_agent.py
#
# Usage (example):
#   python ppo_agent.py --num_envs 128 --steps_per_rollout 256 --total_updates 2000
#
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

import backgammon_engine as bge
from backgammon_value_net import (
    BackgammonActorCriticNet,
    BOARD_LENGTH,
    CONV_INPUT_CHANNELS,
    AUX_INPUT_SIZE,
)

# -------------------------
# Constants / shapes
# -------------------------
NUM_POINTS = int(bge.NUM_POINTS)
W_BAR = int(bge.W_BAR)
B_BAR = int(bge.B_BAR)
W_OFF = int(bge.W_OFF)
B_OFF = int(bge.B_OFF)
NUM_CHECKERS = int(bge.NUM_CHECKERS)

SUBMOVE_GRID = 25 * 25
MAX_SUBMOVES = 4

# Policy-grid indexing (25 values each axis)
# From-axis: 0=BAR, 1..24=points
# To-axis:   0=OFF, 1..24=points


def eng_from_to_grid_from(fp_eng: int) -> int:
    # canonical player is +1, so bar is W_BAR in canonical state
    return 0 if fp_eng == W_BAR else fp_eng  # points already 1..24


def eng_to_to_grid_to(tp_eng: int) -> int:
    # bearing off goes to W_OFF in canonical state
    return 0 if tp_eng == W_OFF else tp_eng  # points already 1..24


def grid_from_to_eng_from(fp_grid: int) -> int:
    return W_BAR if fp_grid == 0 else fp_grid


def grid_to_to_eng_to(tp_grid: int) -> int:
    return W_OFF if tp_grid == 0 else tp_grid


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class PPOConfig:
    seed: int = 0

    # Vectorized self-play
    num_envs: int = 128
    steps_per_rollout: int = 256
    gamma: float = 1.0
    gae_lambda: float = 0.95

    # PPO
    lr: float = 3e-4
    clip_eps: float = 0.2
    vf_coef: float = 1.0
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0

    ppo_epochs: int = 4
    minibatch_size: int = 4096

    # Logging
    log_every: int = 10

    # Policy pruning (used for competition search in the assignment)
    top_k_prune: int = 5


# ============================================================
# Feature Encoding
# ============================================================

def to_canonical_state(state: np.ndarray, player: int) -> np.ndarray:
    """
    Convert (state, player) -> canonical state where current player is +1 (white).
    """
    return np.asarray(bge._to_canonical(state, np.int8(player)), dtype=np.int8)


def encode_board_planes(state_canon: np.ndarray) -> np.ndarray:
    """
    24 x 15 planes (float32), assignment semantic buckets.

    Channel 0: Empty (shared) 1 if both players have count==0 at point i

    For each player (canonical white = current player; canonical black = opponent):
      Blot     (count==1)               -> binary
      Made     (count==2)               -> binary
      Builder  (count==3)               -> binary
      Basic    (count==4)               -> binary
      Deep     (count==5)               -> binary
      Permanent(count==6)               -> binary
      Overflow (count>6) normalized by 9 -> float in [0,1] ( (count-6)/9 )

    Layout:
      0: Empty
      1..7:  white buckets (blot, made, builder, basic, deep, permanent, overflow)
      8..14: black buckets (same)
    """
    planes = np.zeros((NUM_POINTS, CONV_INPUT_CHANNELS), dtype=np.float32)

    EMPTY = 0
    W0 = 1
    B0 = 8

    for p in range(1, NUM_POINTS + 1):
        v = int(state_canon[p])
        w = max(v, 0)
        b = max(-v, 0)

        planes[p - 1, EMPTY] = 1.0 if (w == 0 and b == 0) else 0.0

        # white buckets
        if w == 1:
            planes[p - 1, W0 + 0] = 1.0
        elif w == 2:
            planes[p - 1, W0 + 1] = 1.0
        elif w == 3:
            planes[p - 1, W0 + 2] = 1.0
        elif w == 4:
            planes[p - 1, W0 + 3] = 1.0
        elif w == 5:
            planes[p - 1, W0 + 4] = 1.0
        elif w == 6:
            planes[p - 1, W0 + 5] = 1.0
        elif w > 6:
            planes[p - 1, W0 + 6] = min((w - 6) / 9.0, 1.0)

        # black buckets
        if b == 1:
            planes[p - 1, B0 + 0] = 1.0
        elif b == 2:
            planes[p - 1, B0 + 1] = 1.0
        elif b == 3:
            planes[p - 1, B0 + 2] = 1.0
        elif b == 4:
            planes[p - 1, B0 + 3] = 1.0
        elif b == 5:
            planes[p - 1, B0 + 4] = 1.0
        elif b == 6:
            planes[p - 1, B0 + 5] = 1.0
        elif b > 6:
            planes[p - 1, B0 + 6] = min((b - 6) / 9.0, 1.0)

    return planes


def encode_aux_features(state_canon: np.ndarray) -> np.ndarray:
    """
    6 aux features (float32)

    White (current player in canonical):
      0: Bar active (1 if bar>0 else 0)
      1: Bar scale (bar/15)
      2: Borne-off scale (off/15)

    Black (opponent in canonical):
      3: Bar active
      4: Bar scale
      5: Borne-off scale
    """
    aux = np.zeros((AUX_INPUT_SIZE,), dtype=np.float32)

    w_bar = float(max(int(state_canon[W_BAR]), 0))
    b_bar = float(max(int(-state_canon[B_BAR]), 0))
    w_off = float(max(int(state_canon[W_OFF]), 0))
    b_off = float(max(int(-state_canon[B_OFF]), 0))

    aux[0] = 1.0 if w_bar > 0 else 0.0
    aux[1] = w_bar / float(NUM_CHECKERS)
    aux[2] = w_off / float(NUM_CHECKERS)

    aux[3] = 1.0 if b_bar > 0 else 0.0
    aux[4] = b_bar / float(NUM_CHECKERS)
    aux[5] = b_off / float(NUM_CHECKERS)

    return aux


def encode_batch_for_net(states_canon: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    states_canon: (B, 28) int8, canonical where current player is +1
    Returns:
      board_flat: (B, 24*15) float32
      aux: (B, 6) float32
    """
    B = states_canon.shape[0]
    boards = np.zeros((B, BOARD_LENGTH * CONV_INPUT_CHANNELS), dtype=np.float32)
    auxs = np.zeros((B, AUX_INPUT_SIZE), dtype=np.float32)

    for i in range(B):
        planes = encode_board_planes(states_canon[i])
        aux = encode_aux_features(states_canon[i])
        boards[i] = planes.reshape(-1)
        auxs[i] = aux

    return jnp.asarray(boards), jnp.asarray(auxs)


# ============================================================
# PPO data structures
# ============================================================

@dataclass
class BatchRollout:
    """
    Stores what happened while playing many games at once.
    """
    states: np.ndarray        # (T, B, 28) int8  canonical at decision time
    dices: np.ndarray         # (T, B, 2)  int8

    submoves: np.ndarray      # (T, B, MAX_SUBMOVES, 2) int16 (chosen move, padded -1)
    mask0: np.ndarray         # (T, B, 625) uint8 (legal first-submoves)
    logp_old: np.ndarray      # (T, B) float32 (logp of FIRST submove under rollout policy)

    values: np.ndarray        # (T, B) float32
    rewards: np.ndarray       # (T, B) float32
    dones: np.ndarray         # (T, B) uint8

    def as_flat(self) -> Dict[str, np.ndarray]:
        T, B = self.logp_old.shape
        N = T * B
        return {
            "states": self.states.reshape(N, -1),
            "dices": self.dices.reshape(N, -1),
            "submoves": self.submoves.reshape(N, MAX_SUBMOVES, 2),
            "mask0": self.mask0.reshape(N, SUBMOVE_GRID),
            "logp_old": self.logp_old.reshape(N),
            "values": self.values.reshape(N),
            "rewards": self.rewards.reshape(N),
            "dones": self.dones.reshape(N),
        }


# ============================================================
# JAX model / train state
# ============================================================

class PPOTrainState(TrainState):
    pass


def make_train_state(rng_key: jax.Array, lr: float, max_grad_norm: float) -> PPOTrainState:
    model = BackgammonActorCriticNet()
    dummy_board = jnp.zeros((1, BOARD_LENGTH * CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    params = model.init(rng_key, dummy_board, dummy_aux)

    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )
    return PPOTrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def model_forward(params: Any, board_flat: jnp.ndarray, aux: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      values: (B,)
      logits: (B, 25, 25)
    """
    v, logits = BackgammonActorCriticNet().apply(params, board_flat, aux)
    return v, logits


# ============================================================
# Masked softmax utilities (numpy-side sampling)
# ============================================================

def masked_log_softmax(logits: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    """
    logits: (625,) float32
    mask01: (625,) uint8 {0,1}
    returns logprobs for all positions (masked entries get large negative)
    """
    if mask01.sum() == 0:
        l = logits - logits.max()
        logz = np.log(np.exp(l).sum() + 1e-9)
        return l - logz

    neg_inf = -1e30
    masked = np.where(mask01.astype(bool), logits, neg_inf)
    m = masked.max()
    ex = np.exp(masked - m) * mask01
    z = ex.sum() + 1e-9
    return (masked - m) - np.log(z)


def sample_from_logprobs(rng: np.random.Generator, logp: np.ndarray, mask01: np.ndarray) -> int:
    p = np.exp(logp) * mask01
    s = p.sum()
    if s <= 0:
        legal = np.flatnonzero(mask01)
        return int(legal[np.argmax(logp[legal])])
    p = p / s
    return int(rng.choice(np.arange(p.shape[0]), p=p))


# ============================================================
# Move helpers (convert engine Action -> (from,to) submoves)
# ============================================================

def action_to_submoves(action: Any, player: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for (from_point, roll) in action:
        fp_eng = int(from_point)
        r = int(roll)
        tp_eng = int(bge._get_target_index(np.int8(fp_eng), np.int8(r), np.int8(player)))

        fp_grid = eng_from_to_grid_from(fp_eng)
        tp_grid = eng_to_to_grid_to(tp_eng)

        out.append((fp_grid, tp_grid))
    return out


def build_stage0_mask_from_candidate_moves(candidate_moves_sub: List[List[Tuple[int, int]]]) -> np.ndarray:
    """
    Build a (625,) uint8 mask for legal FIRST submoves.
    """
    mask = np.zeros((25, 25), dtype=np.uint8)
    for mv in candidate_moves_sub:
        if len(mv) >= 1:
            f, t = mv[0]
            if 0 <= f < 25 and 0 <= t < 25:
                mask[f, t] = 1
    return mask.reshape(-1)


def filter_candidates_by_choice(
    candidate_moves_sub: List[List[Tuple[int, int]]],
    stage: int,
    chosen_flat_idx: int,
) -> List[List[Tuple[int, int]]]:
    f = chosen_flat_idx // 25
    t = chosen_flat_idx % 25
    kept = []
    for mv in candidate_moves_sub:
        if stage < len(mv) and mv[stage][0] == f and mv[stage][1] == t:
            kept.append(mv)
    return kept


def apply_one_submove(state: np.ndarray, player: int, from_pt: int, to_pt: int) -> np.ndarray:
    # grid -> engine
    fp_eng = grid_from_to_eng_from(int(from_pt))
    tp_eng = grid_to_to_eng_to(int(to_pt))

    ns = bge._apply_sub_move(state, np.int8(player), np.int8(fp_eng), np.int8(tp_eng))
    if ns is None:
        raise RuntimeError(
            f"_apply_sub_move returned None for from={from_pt} to={to_pt} "
            f"(engine from={fp_eng} to={tp_eng})"
        )
    return np.asarray(ns, dtype=np.int8)


# ============================================================
# Vectorized reward for canonical POV
# ============================================================

def reward_canonical(states_after: np.ndarray) -> np.ndarray:
    B = states_after.shape[0]
    r = np.zeros((B,), dtype=np.float32)
    p = np.int8(1)
    for i in range(B):
        # bge._reward returns win/loss (including gammon/backgammon magnitudes)
        r[i] = float(bge._reward(states_after[i], p))
    return r


# ============================================================
# Vectorized rollout with sequential submove sampling
# ============================================================

def rollout_vectorized(
    rng: np.random.Generator,
    train_state: PPOTrainState,
    cfg: PPOConfig,
    states: np.ndarray,
    dices: np.ndarray,
) -> Tuple[BatchRollout, np.ndarray, np.ndarray]:
    B = cfg.num_envs
    T = cfg.steps_per_rollout

    buf_states = np.zeros((T, B, 28), dtype=np.int8)
    buf_dices = np.zeros((T, B, 2), dtype=np.int8)
    buf_submoves = np.full((T, B, MAX_SUBMOVES, 2), -1, dtype=np.int16)

    buf_mask0 = np.zeros((T, B, SUBMOVE_GRID), dtype=np.uint8)
    buf_logp0 = np.zeros((T, B), dtype=np.float32)

    buf_vals = np.zeros((T, B), dtype=np.float32)
    buf_rew = np.zeros((T, B), dtype=np.float32)
    buf_done = np.zeros((T, B), dtype=np.uint8)

    cur_states = states.copy()
    cur_dices = dices.copy()

    for t in range(T):
        buf_states[t] = cur_states
        buf_dices[t] = cur_dices

        # Forward pass
        board_flat, aux = encode_batch_for_net(cur_states)
        v_jax, logits_jax = model_forward(train_state.params, board_flat, aux)
        v = np.asarray(v_jax, dtype=np.float32)
        logits = np.asarray(logits_jax, dtype=np.float32).reshape(B, SUBMOVE_GRID)
        buf_vals[t] = v

        # Parse through legal moves
        players = np.ones((B,), dtype=np.int8)  # canonical player always +1
        legal_moves, _legal_afterstates = bge._vectorized_actions_parallel(cur_states, players, cur_dices)

        chosen_actions: List[Any] = [None] * B

        # Build per-env candidate lists
        candidates_sub: List[List[List[Tuple[int, int]]]] = []
        candidates_raw: List[List[Any]] = []
        for i in range(B):
            mv_list = list(legal_moves[i])
            candidates_raw.append(mv_list)
            mv_sub = [action_to_submoves(mv, player=1) for mv in mv_list]
            candidates_sub.append(mv_sub)

        # Build mask0, sample first submove, store logp_old
        stage0_masks = np.zeros((B, SUBMOVE_GRID), dtype=np.uint8)
        chosen0_flat = np.full((B,), -1, dtype=np.int32)
        logp0 = np.zeros((B,), dtype=np.float32)

        for i in range(B):
            if len(candidates_sub[i]) == 0:
                continue
            m0 = build_stage0_mask_from_candidate_moves(candidates_sub[i])
            stage0_masks[i] = m0
            if m0.sum() == 0:
                continue

            log_soft = masked_log_softmax(logits[i], m0)
            a0 = sample_from_logprobs(rng, log_soft, m0)
            chosen0_flat[i] = a0
            logp0[i] = float(log_soft[a0])

            f0 = a0 // 25
            t0p = a0 % 25
            buf_submoves[t, i, 0, 0] = f0
            buf_submoves[t, i, 0, 1] = t0p

            candidates_sub[i] = filter_candidates_by_choice(candidates_sub[i], stage=0, chosen_flat_idx=int(a0))

        buf_mask0[t] = stage0_masks
        buf_logp0[t] = logp0

        inter_states = cur_states.copy()
        for i in range(B):
            if chosen0_flat[i] >= 0:
                f0 = int(chosen0_flat[i] // 25)
                t0p = int(chosen0_flat[i] % 25)
                inter_states[i] = apply_one_submove(inter_states[i], player=1, from_pt=f0, to_pt=t0p)

        # Keep sampling submoves to get a legal full move for environment transition
        unresolved = np.array([len(candidates_sub[i]) > 1 for i in range(B)], dtype=bool)

        for stage in range(1, MAX_SUBMOVES):
            idxs = np.where(unresolved)[0]
            if len(idxs) == 0:
                break

            bf, ax = encode_batch_for_net(inter_states[idxs])
            _v2, lg2 = model_forward(train_state.params, bf, ax)
            stage_logits = np.asarray(lg2, dtype=np.float32).reshape(len(idxs), SUBMOVE_GRID)

            for j, i in enumerate(idxs):
                if len(candidates_sub[i]) == 0:
                    unresolved[i] = False
                    continue

                # stage legality mask
                mask = np.zeros((25, 25), dtype=np.uint8)
                for mv in candidates_sub[i]:
                    if stage < len(mv):
                        f, tt = mv[stage]
                        if 0 <= f < 25 and 0 <= tt < 25:
                            mask[f, tt] = 1
                mask = mask.reshape(-1)

                if mask.sum() == 0:
                    unresolved[i] = False
                    continue

                log_soft = masked_log_softmax(stage_logits[j], mask)
                a = sample_from_logprobs(rng, log_soft, mask)

                f = a // 25
                to = a % 25
                buf_submoves[t, i, stage, 0] = f
                buf_submoves[t, i, stage, 1] = to

                candidates_sub[i] = filter_candidates_by_choice(candidates_sub[i], stage, int(a))

                # update intermediate state
                inter_states[i] = apply_one_submove(inter_states[i], player=1, from_pt=int(f), to_pt=int(to))

                unresolved[i] = (len(candidates_sub[i]) > 1)

        # Finalize chosen action per env
        for i in range(B):
            if len(candidates_raw[i]) == 0:
                chosen_actions[i] = bge.List.empty_list(bge.TwoInt8Tuple)
                continue

            # If we filtered down to exactly one candidate, pick it.
            if len(candidates_sub[i]) == 1:
                target = candidates_sub[i][0]
                found = None
                for mv in candidates_raw[i]:
                    if action_to_submoves(mv, player=1) == target:
                        found = mv
                        break
                chosen_actions[i] = found if found is not None else candidates_raw[i][0]
            else:
                # still ambiguous -> choose first deterministically
                chosen_actions[i] = candidates_raw[i][0]

        # Apply moves in batch
        move_seq = bge.List()  # numba typed list
        for i in range(B):
            move_seq.append(chosen_actions[i])
        next_states = bge._vectorized_apply_move(cur_states, players, move_seq)

        # Reward + done
        next_states_np = np.asarray(next_states, dtype=np.int8)
        r = reward_canonical(next_states_np)
        done = (r != 0.0).astype(np.uint8)

        buf_rew[t] = r
        buf_done[t] = done

        # Advance envs: reset terminals; otherwise canonicalize to opponent-to-move
        rolled = bge._vectorized_roll_dice(B)
        next_dices = np.asarray(rolled, dtype=np.int8)

        for i in range(B):
            if done[i]:
                p, d, s = bge._new_game()
                s = np.asarray(s, dtype=np.int8)
                cur_states[i] = to_canonical_state(s, int(p))
                cur_dices[i] = np.asarray(d, dtype=np.int8)
            else:
                cur_states[i] = to_canonical_state(next_states_np[i], player=-1)
                cur_dices[i] = next_dices[i]

    rollout = BatchRollout(
        states=buf_states,
        dices=buf_dices,
        submoves=buf_submoves,
        mask0=buf_mask0,
        logp_old=buf_logp0,
        values=buf_vals,
        rewards=buf_rew,
        dones=buf_done,
    )
    return rollout, cur_states, cur_dices


# ============================================================
# GAE
# ============================================================

def compute_gae(
    rollout: BatchRollout,
    last_values: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    T, B = rollout.values.shape
    adv = np.zeros((T, B), dtype=np.float32)
    last_gae = np.zeros((B,), dtype=np.float32)

    for t in reversed(range(T)):
        done = rollout.dones[t].astype(np.float32)
        not_done = 1.0 - done

        v = rollout.values[t]
        v_next = last_values if t == T - 1 else rollout.values[t + 1]

        delta = rollout.rewards[t] + gamma * v_next * not_done - v
        last_gae = delta + gamma * lam * not_done * last_gae
        adv[t] = last_gae

    ret = adv + rollout.values
    return adv, ret


# ============================================================
# PPO update (ratio uses only first submove)
# ============================================================

def ppo_update(
    train_state: PPOTrainState,
    cfg: PPOConfig,
    rollout: BatchRollout,
) -> Tuple[PPOTrainState, Dict[str, float]]:
    # Bootstrap last values
    last_states = rollout.states[-1]
    board_flat, aux = encode_batch_for_net(last_states)
    v_last_jax, _ = model_forward(train_state.params, board_flat, aux)
    last_values = np.asarray(v_last_jax, dtype=np.float32)

    adv, ret = compute_gae(rollout, last_values, cfg.gamma, cfg.gae_lambda)

    flat = rollout.as_flat()
    N = flat["logp_old"].shape[0]

    adv_f = adv.reshape(N)
    ret_f = ret.reshape(N)

    # Advantage normalization
    adv_mean = float(adv_f.mean())
    adv_std = float(adv_f.std() + 1e-8)
    adv_f = (adv_f - adv_mean) / adv_std

    # Chosen first-submove index: from,to -> f*25 + t
    f0 = flat["submoves"][:, 0, 0].astype(np.int32)
    t0 = flat["submoves"][:, 0, 1].astype(np.int32)
    chosen0 = np.full((N,), -1, dtype=np.int32)
    ok0 = (f0 >= 0) & (t0 >= 0)
    chosen0[ok0] = f0[ok0] * 25 + t0[ok0]

    idxs = np.arange(N)

    def loss_fn(params: Any, mb: Dict[str, Any]) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        # Value + policy logits at state0
        v_pred, logits = model_forward(params, mb["board0"], mb["aux0"])
        logits_flat = logits.reshape((logits.shape[0], SUBMOVE_GRID))

        mask = jnp.asarray(mb["mask0"], dtype=jnp.float32)

        neg_inf = -1e30
        masked = jnp.where(mask > 0.5, logits_flat, neg_inf)
        m = jnp.max(masked, axis=1, keepdims=True)
        ex = jnp.exp(masked - m) * mask
        z = jnp.sum(ex, axis=1, keepdims=True) + 1e-9
        log_soft = (masked - m) - jnp.log(z)

        chosen = jnp.asarray(mb["chosen0"], dtype=jnp.int32)
        gather = jnp.take_along_axis(
            log_soft,
            jnp.clip(chosen[:, None], 0, SUBMOVE_GRID - 1),
            axis=1,
        ).squeeze(1)
        logp_new = jnp.where(chosen >= 0, gather, 0.0)

        # Entropy of legal first-submove distribution
        p = jnp.exp(log_soft) * mask
        ps = jnp.sum(p, axis=1, keepdims=True) + 1e-9
        p = p / ps
        entropy = jnp.mean(-jnp.sum(p * jnp.where(p > 0, jnp.log(p + 1e-9), 0.0), axis=1))

        # PPO ratio using ONLY first submove logp
        logp_old = jnp.asarray(mb["logp_old"], dtype=jnp.float32)
        ratio = jnp.exp(logp_new - logp_old)

        adv_mb = jnp.asarray(mb["adv"], dtype=jnp.float32)
        unclipped = ratio * adv_mb
        clipped = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_mb
        pg_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

        ret_mb = jnp.asarray(mb["ret"], dtype=jnp.float32)
        vf_loss = jnp.mean((v_pred - ret_mb) ** 2)

        loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * entropy

        info = {
            "pg_loss": pg_loss,
            "vf_loss": vf_loss,
            "entropy": entropy,
            "approx_kl": 0.5 * jnp.mean((logp_new - logp_old) ** 2),
            "ratio_mean": jnp.mean(ratio),
        }
        return loss, info

    @jax.jit
    def train_step(state: PPOTrainState, mb: Dict[str, Any]) -> Tuple[PPOTrainState, Dict[str, jax.Array]]:
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, mb)
        new_state = state.apply_gradients(grads=grads)
        info = dict(info)
        info["loss"] = loss
        return new_state, info

    info_accum: Dict[str, float] = {
        "loss": 0.0,
        "pg_loss": 0.0,
        "vf_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "ratio_mean": 0.0,
    }
    n_steps = 0

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(idxs)
        for start in range(0, N, cfg.minibatch_size):
            mb_idx = idxs[start : start + cfg.minibatch_size]
            if len(mb_idx) == 0:
                continue

            mb_states0 = flat["states"][mb_idx].astype(np.int8)
            board0, aux0 = encode_batch_for_net(mb_states0)

            mb = {
                "board0": board0,
                "aux0": aux0,
                "mask0": jnp.asarray(flat["mask0"][mb_idx].astype(np.uint8)),
                "chosen0": jnp.asarray(chosen0[mb_idx].astype(np.int32)),
                "logp_old": jnp.asarray(flat["logp_old"][mb_idx].astype(np.float32)),
                "adv": jnp.asarray(adv_f[mb_idx].astype(np.float32)),
                "ret": jnp.asarray(ret_f[mb_idx].astype(np.float32)),
            }

            train_state, info = train_step(train_state, mb)

            info_np = {k: float(np.asarray(v)) for k, v in info.items()}
            for k in info_accum:
                info_accum[k] += info_np.get(k, 0.0)
            n_steps += 1

    for k in info_accum:
        info_accum[k] /= max(n_steps, 1)

    logs: Dict[str, float] = dict(info_accum)
    logs["adv_mean"] = adv_mean
    logs["adv_std"] = adv_std
    logs["return_mean"] = float(ret_f.mean())
    logs["episode_done_rate"] = float(flat["dones"].mean())
    return train_state, logs


# ============================================================
# Main training loop (vectorized self-play)
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--steps_per_rollout", type=int, default=256)
    parser.add_argument("--total_updates", type=int, default=2000)
    args = parser.parse_args()

    cfg = PPOConfig(
        num_envs=args.num_envs,
        steps_per_rollout=args.steps_per_rollout,
    )

    rng = np.random.default_rng(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)

    train_state = make_train_state(key, cfg.lr, cfg.max_grad_norm)

    # Initialize vectorized games
    x0, x1, x2 = bge._vectorized_new_game(cfg.num_envs)
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    # Dice is the (B,2) array
    dices = next(x for x in (x0, x1, x2) if x.ndim == 2 and x.shape[1] == 2).astype(np.int8)

    # There may be multiple (B,28) arrays; pick one as the board state
    state_candidates = [x for x in (x0, x1, x2) if x.ndim == 2 and x.shape[1] == 28]
    states = np.asarray(state_candidates[0], dtype=np.int8)

    # Canonicalize initial states: force player=+1 POV
    states_canon = np.zeros_like(states)
    for i in range(cfg.num_envs):
        states_canon[i] = to_canonical_state(states[i], player=1)
    states = states_canon

    t0 = time.time()

    for update in range(1, args.total_updates + 1):
        rollout, states, dices = rollout_vectorized(rng, train_state, cfg, states, dices)
        train_state, logs = ppo_update(train_state, cfg, rollout)

        if update % cfg.log_every == 0:
            dt = time.time() - t0
            sps = (cfg.num_envs * cfg.steps_per_rollout * cfg.log_every) / max(dt, 1e-9)
            t0 = time.time()
            print(
                f"[upd {update:5d}] "
                f"loss={logs['loss']:.4f} pg={logs['pg_loss']:.4f} vf={logs['vf_loss']:.4f} "
                f"ent={logs['entropy']:.4f} kl={logs['approx_kl']:.6f} "
                f"ret={logs['return_mean']:.3f} done_rate={logs['episode_done_rate']:.3f} "
                f"sps={sps:,.0f}"
            )


if __name__ == "__main__":
    main()
