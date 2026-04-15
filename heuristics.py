import numpy as np
from othello import (
    BLACK, WHITE, EMPTY, BOARD_SIZE,
    get_legal_moves, get_score, is_game_over
)

CORNER_WEIGHT   = 25.0
STABILITY_WEIGHT = 10.0
MOBILITY_WEIGHT  = 5.0
PIECE_WEIGHT     = 1.0

CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]

X_SQUARES = [(1, 1), (1, 6), (6, 1), (6, 6)]


C_SQUARES = [(0, 1), (1, 0), (0, 6), (1, 7),
             (6, 0), (7, 1), (6, 7), (7, 6)]


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


def corner_score(board, player):

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


def _is_stable(board, row, col, player):
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


def mobility_score(board, player):

    opponent      = -player
    player_moves  = len(get_legal_moves(board, player))
    opponent_moves = len(get_legal_moves(board, opponent))

    total = player_moves + opponent_moves
    if total == 0:
        return 0.0
    return (player_moves - opponent_moves) / (total + 1)


def piece_count_score(board, player):

    black_count, white_count = get_score(board)
    if player == BLACK:
        return float(black_count - white_count)
    else:
        return float(white_count - black_count)


def positional_score(board, player):
    opponent = -player
    score = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == player:
                score += POSITION_TABLE[r][c]
            elif board[r][c] == opponent:
                score -= POSITION_TABLE[r][c]
    return score


def hand_crafted_eval(board, player):
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


def make_weighted_eval(weights):

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