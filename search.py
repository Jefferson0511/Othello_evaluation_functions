from othello import (
    apply_move, get_legal_moves, get_next_player,
    is_game_over, get_winner, BLACK, WHITE, EMPTY
)

# ---------------------------------------------------------------------------
# Score constants for terminal states
# ---------------------------------------------------------------------------
WIN_SCORE  =  1_000_000   # Returned when the maximising player wins outright
LOSS_SCORE = -1_000_000   # Returned when the maximising player loses outright
DRAW_SCORE = 0            # Returned on a draw

DEFAULT_DEPTH = 7         # Search depth used when none is specified


# ---------------------------------------------------------------------------
# Core minimax with alpha-beta pruning
# ---------------------------------------------------------------------------
def _minimax(board, depth, alpha, beta, maximising_player, root_player, eval_fn):
    """
    Minimax search with alpha-beta pruning.

    Parameters
    ----------
    board             : current board state (numpy array)
    depth             : remaining search depth (0 = evaluate immediately)
    alpha             : best score the maximising player is guaranteed so far
    beta              : best score the minimising player is guaranteed so far
    maximising_player : True  → current node maximises score
                        False → current node minimises score
    root_player       : the player for whom we are choosing a move (BLACK or
                        WHITE).  The eval_fn always scores from this player's
                        perspective.
    eval_fn           : callable(board, player) → float.  Higher is better for
                        `player`.

    Returns
    -------
    float : the minimax value of this board state
    """
    # --- Terminal check ---
    if is_game_over(board):
        winner = get_winner(board)
        if winner == root_player:
            return WIN_SCORE
        elif winner == EMPTY:
            return DRAW_SCORE
        else:
            return LOSS_SCORE

    # --- Depth limit: evaluate the position statically ---
    if depth == 0:
        return eval_fn(board, root_player)

    current_player = BLACK if maximising_player else WHITE
    # Correct for when root_player is WHITE: maximising maps to WHITE then
    if root_player == WHITE:
        current_player = WHITE if maximising_player else BLACK

    legal_moves = get_legal_moves(board, current_player)

    # --- Forced pass: no legal moves but game not over ---
    # Switch perspective without decrementing depth so we don't waste a ply
    # on a pass that the opponent was forced to make.
    if not legal_moves:
        return _minimax(
            board, depth, alpha, beta,
            not maximising_player, root_player, eval_fn
        )

    if maximising_player:
        best = LOSS_SCORE
        for move in legal_moves:
            child_board = apply_move(board, move, current_player)
            score = _minimax(
                child_board, depth - 1, alpha, beta,
                False, root_player, eval_fn
            )
            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:       # Beta cut-off: minimiser won't allow this
                break
        return best

    else:  # minimising
        best = WIN_SCORE
        for move in legal_moves:
            child_board = apply_move(board, move, current_player)
            score = _minimax(
                child_board, depth - 1, alpha, beta,
                True, root_player, eval_fn
            )
            best = min(best, score)
            beta = min(beta, best)
            if beta <= alpha:       # Alpha cut-off: maximiser won't allow this
                break
        return best


# ---------------------------------------------------------------------------
# Public interface: choose the best move
# ---------------------------------------------------------------------------
def get_best_move(board, player, eval_fn, depth=DEFAULT_DEPTH):
    """
    Return the best move for `player` using minimax with alpha-beta pruning.

    Parameters
    ----------
    board   : current board state (numpy array)
    player  : BLACK or WHITE — the player choosing a move
    eval_fn : callable(board, player) → float.  Any evaluation function with
              this signature can be passed in — hand-crafted, GA-evolved, or
              neural network.
    depth   : search depth (default DEFAULT_DEPTH = 7)

    Returns
    -------
    tuple (row, col) : the best move found, or None if no legal moves exist.
    """
    legal_moves = get_legal_moves(board, player)
    if not legal_moves:
        return None

    best_move  = legal_moves[0]   # Default to first move — ensures we always
    best_score = LOSS_SCORE       # return something even in a losing position
    alpha      = LOSS_SCORE
    beta       = WIN_SCORE

    for move in legal_moves:
        child_board = apply_move(board, move, player)
        score = _minimax(
            child_board,
            depth - 1,
            alpha, beta,
            maximising_player=False,   # Opponent moves next → minimising
            root_player=player,
            eval_fn=eval_fn
        )
        if score > best_score:
            best_score = score
            best_move  = move
        alpha = max(alpha, best_score)

    return best_move


# ---------------------------------------------------------------------------
# Game simulation utility
# ---------------------------------------------------------------------------
def play_game(black_eval_fn, white_eval_fn,
              black_depth=DEFAULT_DEPTH, white_depth=DEFAULT_DEPTH,
              verbose=False):
    """
    Simulate a full game between two agents and return the result.

    Each agent is defined by its evaluation function and search depth.
    This is the function tournament.py will call for every match-up.

    Parameters
    ----------
    black_eval_fn : eval function for the BLACK agent
    white_eval_fn : eval function for the WHITE agent
    black_depth   : search depth for BLACK (default 7)
    white_depth   : search depth for WHITE (default 7)
    verbose       : if True, print the board after each move

    Returns
    -------
    winner        : BLACK, WHITE, or EMPTY (draw)
    black_score   : number of black discs at game end
    white_score   : number of white discs at game end
    """
    from othello import get_initial_board, get_score, display_board

    board          = get_initial_board()
    current_player = BLACK   # Black always moves first in Othello

    if verbose:
        print("=== Game start ===")
        display_board(board)

    while not is_game_over(board):
        legal_moves = get_legal_moves(board, current_player)

        if legal_moves:
            if current_player == BLACK:
                move = get_best_move(board, BLACK, black_eval_fn, black_depth)
            else:
                move = get_best_move(board, WHITE, white_eval_fn, white_depth)

            if move:
                board = apply_move(board, move, current_player)
                if verbose:
                    player_str = "Black" if current_player == BLACK else "White"
                    print(f"{player_str} plays {move}")
                    display_board(board)

        # Advance to the next player (handles forced passes automatically)
        next_p = get_next_player(board, current_player)
        if next_p is None:
            break
        current_player = next_p

    winner = get_winner(board)
    black_score, white_score = get_score(board)

    if verbose:
        result = {BLACK: "Black wins", WHITE: "White wins", EMPTY: "Draw"}
        print(f"=== Game over: {result[winner]} ===")
        print(f"Final score — Black: {black_score}  White: {white_score}")

    return winner, black_score, white_score


# ---------------------------------------------------------------------------
# Quick sanity test  (run: python search.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from othello import get_initial_board

    # A minimal eval function: raw disc count difference (weakest possible)
    def piece_count_eval(board, player):
        return int((board == player).sum() - (board == -player).sum())

    print("Testing get_best_move at depth 3 (fast check)...")
    board = get_initial_board()
    move  = get_best_move(board, BLACK, piece_count_eval, depth=3)
    print(f"Best move for Black: {move}")
    assert move is not None, "Should always find a move from the start."

    print("\nSimulating a short game (depth 3 for speed)...")
    winner, b, w = play_game(
        piece_count_eval, piece_count_eval,
        black_depth=3, white_depth=3,
        verbose=True
    )
    result = {BLACK: "Black", WHITE: "White", EMPTY: "Draw"}
    print(f"\nResult: {result[winner]}  |  Black: {b}  White: {w}")
    print("All sanity checks passed.")