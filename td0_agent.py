import backgammon_engine as bge
import numpy as np

# --- Hyperparameters for TD(0) agent ---
GAMMA = 1.0      # episodic game, no discounting
ALPHA = 1e-3     # learning rate (tune this)
FEATURE_DIM = 28 # start with simple 28-dim features (board itself)

# =========================
# Feature representation
# =========================

# Handcrafted feature set:
#  - 28 raw board features
#  - 2 (blot counts)
#  - 2 (>=3 stacks in home)
#  - 2 (longest prime on full board)
#  - 2 (longest prime in home)
#  - 2 (# opp checkers trapped behind longest prime)
#  - 2 (made points in home)
#  - 2 (# checkers within 6 pips of opp blot)
#  - 1 (x_race, contact/race phase, from white's POV)
#  - 1 (pip count difference: white - black)
#  - 8 product features:
#       2: (opp blot count) * (own #points within 6 pips of an opp blot)
#       2: (longest prime length) * (# opp checkers trapped)
#       2: x_race * pip_diff and (1 - x_race) * pip_diff
#       2: (made_points_home)^2 for each player
FEATURE_DIM = 52

W_BAR = int(bge.W_BAR)
B_BAR = int(bge.B_BAR)
W_OFF = int(bge.W_OFF)
B_OFF = int(bge.B_OFF)
NUM_POINTS = int(bge.NUM_POINTS)
HOME_BOARD_SIZE = int(bge.HOME_BOARD_SIZE)

def _home_range(player: int):
    """
    Return (start, end) inclusive indices for the home board
    for the given player (+1 for white, -1 for black).
    """
    if player == 1: # white home: points 19-24
        return 19, 24
    else:           # black home: points 1-6
        return 1, 6
    
def _pip_count(state: np.ndarray, player: int) -> int:
    """
    Total pip count for the given player, using standard backgammon convention:
      - For white: distance from point i to bear off = 25 - i
      - For black: distance from point i to bear off = i
      - Bar checkers count as 25 pips.
    """
    total = 0
    # On-board checkers
    for i in range(1, NUM_POINTS + 1):
        n = state[i] * player
        if n > 0:
            if player == 1:
                dist = (NUM_POINTS + 1) - i # 25 - i
            else:
                dist = i
            total += int(n) * dist

    # Bar checkers
    bar_idx = W_BAR if player == 1 else B_BAR
    n_bar = state[bar_idx] * player
    if n_bar > 0:
        total += int(n_bar) * (NUM_POINTS + 1) # 25 pips for each bar checker
    
    return total

def _longest_prime(state: np.ndarray, player: int, start: int, end: int):
    """
    Find the longest consecutive sequence of points in [start, end] (inclusive)
    where the player has at least 2 checkers on each point.

    Returns (length, prime_start, prime_end). If no prime, returns (0, -1, -1).
    """
    max_len = 0
    max_start = -1
    max_end = -1

    current_len = 0
    current_start = -1

    for i in range(start, end + 1):
        n = state[i] * player
        if n >= 2:
            if current_len == 0:
                current_start = i
                current_len = 1
            else:
                current_len += 1
            if current_len > max_len:
                max_len = current_len
                max_start = current_start
                max_end = i
        else:
            current_len = 0
            current_start = -1
    
    return max_len, max_start, max_end

def _opp_trapped_behind_prime(state: np.ndarray, player: int, prime_start: int, prime_end: int) -> int:
    """
    Number of opponent checkers 'trapped behind' the given prime segment
    for the player. If no prime (prime_start == -1), returns 0.

    For white (player = +1, moving 1 -> 24):
        - Check opponent checkers on points > prime_end.

    For black (player = -1, moving 24 -> 1):
        - Check opponent checkers on points < prime_start.
    """
    if prime_start < 0 or prime_end < 0:
        return 0
    
    opp = -player
    trapped = 0

    if player == 1:
        # points strictly "behind" prime, towards black bear-off
        for i in range(prime_end + 1, NUM_POINTS + 1):
            n_opp = state[i] * opp
            if n_opp > 0:
                trapped += int(n_opp)
    else:
        # black moves down; white behind prime are on points < prime_start
        for i in range(1, prime_start):
            n_opp = state[i] * opp
            if n_opp > 0:
                trapped += int(n_opp)
    
    return trapped

def _blot_positions(state: np.ndarray, player: int):
    """
    Return the list of point indices (1..24) where `player` has a blot (exactly 1 checker).
    """
    blots = []
    for i in range(1, NUM_POINTS + 1):
        n = state[i] * player
        if n == 1:
            blots.append(i)
    return blots

def _checkers_and_points_near_opp_blots(state: np.ndarray, player: int, opp_blots: list[int]):
    """
    For the given player and a list of opponent blot positions,
    compute:
      - total number of player's checkers within 6 pips of ANY opponent blot
      - number of distinct points (with at least one checker) that are within 6 pips
    """
    checkers_near = 0
    points_near = 0

    if not opp_blots:
        return 0, 0
    
    for i in range(1, NUM_POINTS + 1):
        n = state[i] * player
        if n <= 0:
            continue

        # Is this point within 6 pips of any opp blot?
        near = False
        for j in opp_blots:
            if player == 1:
                # white moves up: can hit blot at j if 1 <= j - 1 <= 6
                if 1 <= j - i <= 6:
                    near = True
                    break
            else:
                # black moves down: can hit blot at j if 1 <= i - j <= 6
                if 1 <= i - j <= 6:
                    near = True
                    break

        if near:
            points_near += 1
            checkers_near += int(n)

    return checkers_near, points_near

def feature_function(state: np.ndarray) -> np.ndarray:
    """
    Map a backgammon state (length-28 int8 array) to a feature vector φ(s).

    Features implemented per the assignment:

      1. Raw board encoding (28 features):
         [W_BAR, P1..P24, B_BAR, W_OFF, B_OFF].

      2. Blot counts (2 features): #blots for white, #blots for black.

      3. Number of checkers stacked >= 3 in home quadrant (2 features).

      4. Length of longest prime (>=2 checkers per point) anywhere on board (2 features).

      5. Length of longest prime inside home board (2 features).

      6. Number of opponent checkers trapped behind longest prime (2 features).

      7. Number of home-board points with at least 2 checkers ("made points") (2 features).

      8. Number of checkers within 6 pips of any opponent blot (2 features).

      9. x_race (1 feature): from WHITE's perspective, in [0,1],
         x_race = (number_of_white_checkers_past_point_12) / 15.

     10. Pip count difference (1 feature): pip_white - pip_black.

     11. Product features (8 features):
         - For each player: (opp blot count) * (own #points within 6 pips of any opp blot)  (2)
         - For each player: (longest prime length) * (opp trapped behind that prime)        (2)
         - x_race * pip_diff and (1 - x_race) * pip_diff                                   (2)
         - For each player: (made_points_home)^2                                          (2)
    """
    state = np.asarray(state, dtype=np.int8)
    
    feats: list[float] = []

    # Raw board features (28)
    feats.extend(state.astype(np.float32))

    players = [1, -1] # white, black

    # Per-player metrics we will need also for products
    blot_counts = {}
    stacked3_home = {}
    longest_prime_len = {}
    longest_prime_len_home = {}
    trapped_opp_behind_prime = {}
    made_points_home = {}
    checkers_near_blots = {}
    points_near_blots = {}
    opp_blot_counts = {}

    # Compute blot positions for both players
    blot_positions = {
        1: _blot_positions(state, 1),
        -1: _blot_positions(state, -1)
    }

    # Compute main symmetric features per player
    for player in players:
        opp = -player

        # Blot count
        blots = len(blot_positions[player])
        blot_counts[player] = blots
        feats.append(float(blots))

        # Number of checkers stacked >= 3 in home quadrant
        home_start, home_end = _home_range(player)
        stacked = 0
        for i in range(home_start, home_end + 1):
            n = state[i] * player
            if n >= 3:
                stacked += int(n)
        stacked3_home[player] = stacked
        feats.append(float(stacked))

        # Longest prime on full board and in home
        lp_len, lp_start, lp_end = _longest_prime(state, player, 1, NUM_POINTS)
        longest_prime_len[player] = lp_len
        feats.append(float(lp_len))

        lp_home_len, lp_home_start, lp_home_end = _longest_prime(
            state, player, home_start, home_end
        )
        longest_prime_len_home[player] = lp_home_len
        feats.append(float(lp_home_len))

        # Opponent checkers trapped behind longest prime (full board prime)
        trapped = _opp_trapped_behind_prime(state, player, lp_start, lp_end)
        trapped_opp_behind_prime[player] = trapped
        feats.append(float(trapped))

        # Number of made points in home board (>= 2 checkers)
        made = 0
        for i in range(home_start, home_end + 1):
            n = state[i] * player
            if n >= 2:
                made += 1
        made_points_home[player] = made
        feats.append(float(made))

        # Number of checkers within 6 pips of any opponent blot
        opp_blots = blot_positions[opp]
        checkers_near, points_near = _checkers_and_points_near_opp_blots(
            state, player, opp_blots
        )
        checkers_near_blots[player] = checkers_near
        points_near_blots[player] = points_near
        opp_blot_counts[player] = len(opp_blots)
        feats.append(float(checkers_near))

    # x_race (white's contact / race phase indicator)
    # Let c = number of white checkers on points 1-12.
    # Then x_race = (15 - c) / 15, in [0,1].
    white_contact = 0
    for i in range(1, 12 + 1):
        n_white = state[i]
        if n_white > 0:
            white_contact += int(n_white)
    x_race = (15.0 - white_contact) / 15.0
    x_race = float(np.clip(x_race, 0.0, 1.0))
    feats.append(x_race)

    # Pip count difference (white - black)
    pip_white = _pip_count(state, 1)
    pip_black = _pip_count(state, -1)
    pip_diff = float(pip_white - pip_black)
    feats.append(pip_diff)

    # Product features (8)

    # (a) For each player: (opp blot count) * (own #points within 6 pips of any opp blot)
    for player in players:
        opp = -player
        opp_blots_count = blot_counts[opp]
        own_points_near = points_near_blots[player]
        feats.append(float(opp_blots_count * own_points_near))

    # (b) For each player: (longest prime length) * (#opp checkers trapped behind that prime)
    for player in players:
        lp_len = longest_prime_len[player]
        trapped = trapped_opp_behind_prime[player]
        feats.append(float(lp_len * trapped))

    # (c) x_race * pip_diff and (1 - x_race) * pip_diff
    feats.append(x_race * pip_diff)
    feats.append((1.0 - x_race) * pip_diff)

    # (d) For each player: (made_points_home)^2
    for player in players:
        made = made_points_home[player]
        feats.append(float(made * made))

    # Convert to Numpy array
    return np.asarray(feats, dtype=np.float32)

def init_weights(feature_dim: int = FEATURE_DIM) -> np.ndarray:
    """
    Initialize linear value function weights.
    """
    # Small random initialization
    return np.zeros(feature_dim, dtype=np.float32)

# =======================================
# Value function and batch evaluation
# =======================================

def value_white(weights: np.ndarray, state: np.ndarray) -> float:
    """
    V_white(s) = w^T φ(s), interpreted as value from white's perspective.
    """
    phi = feature_function(state)
    return float(np.dot(weights, phi))

def make_linear_batch_value_function(weights: np.ndarray, root_player: int):
    """
    Construct batch_value_function to pass into _2_ply_search.

    - Internally, we evaluate V_white(s) for each state.
    - We then multiply by root_player ∈ {+1, -1} to interpret it
      as value from the root player's perspective:

        V_root(s) = root_player * V_white(s)

    This way, the same value function can be used for either side.
    """

    def batch_value_function(state_buffer):
        # state_buffer is a numba List or Python sequence of np.ndarray states
        values = np.empty(len(state_buffer), dtype=np.float64)
        for i in range(len(state_buffer)):
            s = np.array(state_buffer[i], dtype=np.int8)
            v_white = value_white(weights, s)
            values[i] = root_player * v_white
        return values
    return batch_value_function

# ==============================
# TD(0) update for one step
# ==============================

def td0_update(weights: np.ndarray,
               state_t: np.ndarray,
               reward_tp1_white: float,
               state_tp1: np.ndarray | None,
               alpha: float = ALPHA,
               gamma: float = GAMMA) -> None:
    """
    Perform an in-place TD(0) update on weights for a single transition.

      s_t -> s_{t+1}, reward r_{t+1} (from white's perspective)

    If state_tp1 is None, we treat V(s_{t+1}) = 0 (terminal).
    """
    v_t = value_white(weights, state_t)

    if state_tp1 is None:
        v_tp1 = 0.0
    else: 
        v_tp1 = value_white(weights, state_tp1)

    delta = reward_tp1_white + gamma * v_tp1 - v_t
    phi_t = feature_function(state_t)

    # In-place weight update
    weights += alpha * delta * phi_t

# ==========================================
# Play a single self-play game with TD(0)
# ==========================================

def play_one_game_td0(weights: np.ndarray,
                      alpha: float = ALPHA,
                      gamma: float = GAMMA,
                      rng: np.random.Generator | None = None):
    """
    Play one self-play backgammon game.

    - Both sides choose moves using the same linear value function + 2-ply search.
    - After each move, we perform a TD(0) update on V_white.
    - Returns: final reward from white's perspective, and the updated weights (in-place).
    """
    global GAMMA, ALPHA
    GAMMA = gamma
    ALPHA = alpha

    if rng is None:
        rng = np.random.default_rng()

    # --- Initialize a new game ---
    # player ∈ {+1 (white), -1 (black)}
    player, dice, state = bge._new_game()
    state = np.array(state, dtype=np.int8)

    # Game loop
    step = 0
    while True:
        step += 1

        # Check for legal moves for this player + dice
        player_moves, player_afterstates = bge._actions(state, player, dice)

        if len(player_moves) == 0:
            # ----- FORCED PASS -----
            # No legal moves: state does not change, turn passes to opponent.

            # Reward from WHITE's perspective (usually 0 unless game already over)
            reward_white = bge._reward(state, 1)
            terminal = (reward_white != 0)

            # If somehow terminal here, do one last TD update and stop
            if terminal:
                td0_update(weights, state, reward_white, None, alpha, gamma)
                return reward_white, step

            # Otherwise, just pass the turn: no state change, no TD update
            player = bge.Player(-player) if hasattr(bge, "Player") else -player
            dice = bge._roll_dice()
            continue

        # Build a batch_value_function for this root player
        batch_v = make_linear_batch_value_function(weights, int(player))

        # Use 2-ply search to pick the best move from this state and dice
        move_sequence = bge._2_ply_search(state, player, dice, batch_v)

        # Apply that move to get the next state
        next_state = bge._apply_move(state, player, move_sequence)
        next_state = np.array(next_state, dtype=np.int8)

        # Roll dice for the next player
        dice_next = bge._roll_dice()

        # Compute reward from WHITE'S perspective at next_state
        reward_white = bge._reward(next_state, bge.Player(1))

        # Determine if the game is over (non-zero reward from white's POV)
        terminal = (reward_white != 0.0)

        # TD(0) update on V_white, using (state -> next_state, reward_white)
        if terminal:
            # No next state's value beyond this, so pass None
            td0_update(weights, state, reward_white, None, alpha, gamma)
        else:
            td0_update(weights, state, reward_white, next_state, alpha, gamma)

        # Transition to next step or terminate
        if terminal:
            # Game over; return final outcome from white's perspective
            return reward_white, step
        
        # Continue game
        state = next_state
        player = bge.Player(-player) # switch side to move
        dice = dice_next
