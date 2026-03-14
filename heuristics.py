import numpy as np
from othello import (
    BLACK, WHITE, EMPTY, BOARD_SIZE,
    get_legal_moves, get_score, is_game_over
)

# Hand-crafted feature weights
# These weights define how much each strategic feature contributes to the
# final evaluation score.  They are based on Othello domain knowledge and
# serve as the BASELINE that the GA will later try to improve upon.
#
# Weight interpretation:
#   - Positive weight: feature benefits the maximising player
#   - Higher magnitude:  feature is considered more strategically important
#
# Tuning rationale (drawn from Alliot & Durand 1996 and standard Othello theory):
#   CORNER_WEIGHT   : Corners are permanent and anchor entire edges by far
#                     the most valuable positional feature.
#   STABILITY_WEIGHT: Stable discs (can never be flipped) provide lasting
#                     positional advantage; second only to corners.
#   MOBILITY_WEIGHT : Restricting the opponent's moves is critical in the
#                     early/mid game; less important in the endgame.
#   PIECE_WEIGHT    : Raw disc count matters only in the endgame — chasing
#                     pieces early is a well-known beginner mistake.
# ---------------------------------------------------------------------------
CORNER_WEIGHT   = 25.0
STABILITY_WEIGHT = 10.0
MOBILITY_WEIGHT  = 5.0
PIECE_WEIGHT     = 1.0

CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]

# X-squares: diagonally adjacent to corners giving these up is dangerous
X_SQUARES = [(1, 1), (1, 6), (6, 1), (6, 6)]

# C-squares: edge squares adjacent to corners also risky before corner taken
C_SQUARES = [(0, 1), (1, 0), (0, 6), (1, 7),
             (6, 0), (7, 1), (6, 7), (7, 6)]

# Static positional values for every square on the board.
# These encode the relative long-term value of each square based on
# standard Othello theory (similar to Alliot & Durand's Table 1).
# Corners > edges > interior > X/C-squares (negative: dangerous early on).
POSITION_TABLE = np.array([
    [ 500, -150,  30,  10,  10,  30, -150,  500],
    [-150, -250,   0,   0,   0,   0, -250, -150],
    [  30,    0,   1,   2,   2,   1,    0,   30],
    [  10,    0,   2,  16,  16,   2,    0,   10],
    [  10,    0,   2,  16,  16,   2,    0,   10],
    [  30,    0,   1,   2,   2,   1,    0,   30],
    [-150, -250,   0,   0,   0,   0, -250, -150],
    [ 500, -150,  30,  10,  10,  30, -150,  500],
], dtype=float)


# Feature 1: Corner control
def corner_score(board, player):
    """
    Compute the corner control score for `player`.

    Corners are permanent — once captured they can never be flipped.
    This function rewards owning corners and penalises:
      - X-squares (diagonal to empty corners): dangerously likely to gift
        the corner to the opponent.
      - C-squares (edge squares adjacent to empty corners): risky for the
        same reason, though less severely.

    Returns a float representing net corner advantage for `player`.
    """
    opponent = -player
    score = 0.0

    for i, corner in enumerate(CORNERS):
        cr, cc = corner
        corner_owner = board[cr][cc]

        if corner_owner == player:
            score += 1.0
        elif corner_owner == opponent:
            score -= 1.0
        else:

            x_r, x_c = X_SQUARES[i]
            if board[x_r][x_c] == player:
                score -= 0.5
            elif board[x_r][x_c] == opponent:
                score += 0.5

            c1 = C_SQUARES[i * 2]
            c2 = C_SQUARES[i * 2 + 1]
            for c_sq in (c1, c2):
                if board[c_sq[0]][c_sq[1]] == player:
                    score -= 0.25
                elif board[c_sq[0]][c_sq[1]] == opponent:
                    score += 0.25

    return score


# Feature 2: Stability
def _is_stable(board, row, col, player):
    """
    A disc is STABLE if it cannot be flipped for the rest of the game.

    A disc at (row, col) is stable when, in every axis (horizontal,
    vertical, and both diagonals), at least one of the following holds:
      (a) The entire row/column/diagonal in that axis is filled, OR
      (b) The disc is on the board edge in that axis, OR
      (c) All discs between this disc and both edges in that axis belong
          to the same player (they form an unbroken friendly line).

    This is an approximation of full stability analysis it correctly
    identifies the most common stable configurations without requiring
    the expensive recursive propagation used in world-class programs.
    """
    if board[row][col] != player:
        return False

    axes = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in axes:
        stable_in_axis = False

        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                stable_in_axis = True
                break
            all_friendly = True
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if board[r][c] == EMPTY:
                    all_friendly = False
                    break
                r += sign * dr
                c += sign * dc
            if all_friendly:
                stable_in_axis = True
                break

        if not stable_in_axis:
            return False 

    return True


def stability_score(board, player):
    """
    Return the net number of stable discs: player's stable discs minus
    the opponent's stable discs.
    """
    opponent = -player
    player_stable   = 0
    opponent_stable = 0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == player and _is_stable(board, r, c, player):
                player_stable += 1
            elif board[r][c] == opponent and _is_stable(board, r, c, opponent):
                opponent_stable += 1

    return float(player_stable - opponent_stable)


# Feature 3: Mobility
def mobility_score(board, player):
    """
    Mobility measures how many legal moves each player has.

    High mobility = more options = strategic flexibility.
    Restricting the opponent's mobility is a key mid-game strategy.

    Returns the normalised mobility advantage for `player`:
        (player_moves - opponent_moves) / (player_moves + opponent_moves + 1)

    Normalisation prevents the raw move count from dominating other features
    early in the game when many moves are available.
    """
    opponent      = -player
    player_moves  = len(get_legal_moves(board, player))
    opponent_moves = len(get_legal_moves(board, opponent))

    total = player_moves + opponent_moves
    if total == 0:
        return 0.0
    return (player_moves - opponent_moves) / (total + 1)


# Feature 4: Piece count
def piece_count_score(board, player):
    """
    Raw disc count difference: player's discs minus opponent's discs.

    This is intentionally given the lowest weight because chasing pieces
    early is a well-known strategic mistake in Othello — fewer pieces
    often means more mobility. Piece count becomes dominant only in the
    endgame when the board is nearly full.
    """
    black_count, white_count = get_score(board)
    if player == BLACK:
        return float(black_count - white_count)
    else:
        return float(white_count - black_count)


# Bonus feature: Static positional evaluation
def positional_score(board, player):
    """
    Score the board using the static POSITION_TABLE.

    Each occupied square contributes its table value — positive for the
    player's discs, negative for the opponent's.  This encodes standard
    Othello positional wisdom (corners good, X-squares bad, etc.) and
    was inspired directly by Alliot & Durand's evaluation function
    structure.
    """
    opponent = -player
    score = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == player:
                score += POSITION_TABLE[r][c]
            elif board[r][c] == opponent:
                score -= POSITION_TABLE[r][c]
    return score


# Combined hand-crafted evaluation function
def hand_crafted_eval(board, player):
    """
    The hand-crafted evaluation function — the BASELINE agent.

    Combines all four strategic features using hand-tuned weights.
    This is the evaluation function that the GA will attempt to beat by
    evolving better weights for the same feature set.

    Parameters
    ----------
    board  : numpy array — current board state
    player : BLACK or WHITE — the player to evaluate for

    Returns
    -------
    float : higher values are better for `player`
    """
    if is_game_over(board):
        black_count, white_count = get_score(board)
        player_count   = black_count if player == BLACK else white_count
        opponent_count = white_count if player == BLACK else black_count
        if player_count > opponent_count:
            return 1_000_000.0
        elif player_count < opponent_count:
            return -1_000_000.0
        else:
            return 0.0

    score = (
        CORNER_WEIGHT    * corner_score(board, player)    +
        STABILITY_WEIGHT * stability_score(board, player) +
        MOBILITY_WEIGHT  * mobility_score(board, player)  +
        PIECE_WEIGHT     * piece_count_score(board, player)
    )
    return score


# GA-compatible weight vector interface
# The GA will evolve a weight vector [w0, w1, w2, w3] and call
# make_weighted_eval(weights) to get a drop-in replacement for
# hand_crafted_eval.  This keeps the feature set fixed while allowing
# the GA to discover better weights.

def make_weighted_eval(weights):
    """
    Factory function: given a weight vector, return an evaluation function
    with the same signature as hand_crafted_eval.

    Parameters
    ----------
    weights : sequence of 4 floats
        [corner_w, stability_w, mobility_w, piece_count_w]

    Returns
    -------
    callable(board, player) → float
    """
    w_corner, w_stability, w_mobility, w_piece = weights

    def weighted_eval(board, player):
        if is_game_over(board):
            black_count, white_count = get_score(board)
            player_count   = black_count if player == BLACK else white_count
            opponent_count = white_count if player == BLACK else black_count
            if player_count > opponent_count:
                return 1_000_000.0
            elif player_count < opponent_count:
                return -1_000_000.0
            else:
                return 0.0

        return (
            w_corner    * corner_score(board, player)    +
            w_stability * stability_score(board, player) +
            w_mobility  * mobility_score(board, player)  +
            w_piece     * piece_count_score(board, player)
        )

    return weighted_eval


# test run
if __name__ == "__main__":
    from othello import get_initial_board, apply_move, display_board
    from search import get_best_move

    board = get_initial_board()
    print("=== Testing hand_crafted_eval on starting position ===")
    score = hand_crafted_eval(board, BLACK)
    print(f"Starting position score for Black: {score:.2f}  (expected ~0.0)")
    assert score == 0.0, "Starting position should be perfectly symmetric."

    print("\n=== Testing make_weighted_eval factory ===")
    custom_weights = [25.0, 10.0, 5.0, 1.0]
    custom_eval    = make_weighted_eval(custom_weights)
    score2 = custom_eval(board, BLACK)
    print(f"Custom eval score (same weights): {score2:.2f}  (should match: {score:.2f})")
    assert score2 == score, "make_weighted_eval with default weights should match hand_crafted_eval."

    print("\n=== Playing one move and checking score changes ===")
    move  = get_best_move(board, BLACK, hand_crafted_eval, depth=3)
    board = apply_move(board, move, BLACK)
    print(f"Black plays {move}")
    display_board(board)
    score_after = hand_crafted_eval(board, BLACK)
    print(f"Score for Black after move: {score_after:.2f}")

    print("test passed.")