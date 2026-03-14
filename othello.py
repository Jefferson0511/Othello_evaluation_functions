import numpy as np

# ---------------------------------------------------------------------------
# Player and board constants
# ---------------------------------------------------------------------------
BLACK = 1       # Black moves first
WHITE = -1      # White is Black's opponent
EMPTY = 0       # Empty square
BOARD_SIZE = 8  # Standard 8x8 Othello board

# All 8 directions a line of pieces can run: (row_delta, col_delta)
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
]



# Board initialisation

def get_initial_board():
    """
    Returns an 8x8 numpy array representing the standard Othello starting
    position.

    Board encoding:
        BLACK ( 1) = black disc
        WHITE (-1) = white disc
        EMPTY ( 0) = empty square

    Starting layout (B = black, W = white):
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        . . . W B . . .
        . . . B W . . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
    """
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    mid = BOARD_SIZE // 2 
    board[mid - 1][mid - 1] = WHITE
    board[mid - 1][mid]     = BLACK
    board[mid][mid - 1]     = BLACK
    board[mid][mid]         = WHITE
    return board


# Move generation helpers
"""
    Given a candidate move at (row, col) for `player`, return a list of
    (r, c) positions of opponent pieces that would be flipped.

    A flip occurs in a direction when:
      1. The very next square holds an opponent piece, AND
      2. Continuing in that direction we eventually reach one of our own
         pieces (without hitting an empty square or the board edge first).

    Returns an empty list if the move is illegal (no flips at all).
    """
def _get_flipped_pieces(board, row, col, player):
    
    opponent = -player
    flipped = []

    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        candidates = []
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == opponent:
            candidates.append((r, c))
            r += dr
            c += dc
        if candidates and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
            flipped.extend(candidates)

    return flipped

"""
Return a list of all legal moves for `player` on the given board.

Each move is a tuple (row, col).  A square is a legal move if:
    - It is currently empty, AND
    - Placing a piece there would flip at least one opponent piece.

Returns an empty list if the player has no legal moves.
"""
def get_legal_moves(board, player):

    moves = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == EMPTY:
                if _get_flipped_pieces(board, r, c, player):
                    moves.append((r, c))
    return moves


# 
# Applying a move


"""
Apply `move` (row, col) for `player` and return the resulting board.

The original board is never modified; a new numpy array is returned.
Raises ValueError if the move is illegal.
"""
def apply_move(board, move, player):

    row, col = move
    flipped = _get_flipped_pieces(board, row, col, player)

    if not flipped:
        raise ValueError(f"Illegal move ({row}, {col}) for player {player}.")

    new_board = board.copy()
    new_board[row][col] = player      
    for r, c in flipped:
        new_board[r][c] = player  
    return new_board


# Game-state queries
"""
    Return (black_count, white_count): the number of discs each player owns.
    """
def get_score(board):
    
    black_count = int(np.sum(board == BLACK))
    white_count = int(np.sum(board == WHITE))
    return black_count, white_count

"""
    The game ends when neither player has a legal move.  This occurs when:
      - The board is full, OR
      - Both players are simultaneously without moves (rare but possible).
    """
def is_game_over(board):
    
    return not get_legal_moves(board, BLACK) and not get_legal_moves(board, WHITE)

"""
Return the winner of a finished game (BLACK or WHITE), or EMPTY for a draw.
Should only be called after is_game_over() returns True.
"""
def get_winner(board):

    black_count, white_count = get_score(board)
    if black_count > white_count:
        return BLACK
    elif white_count > black_count:
        return WHITE
    else:
        return EMPTY

"""
Return the next player to move.

If the opponent has legal moves, they go next.
If the opponent has no moves but the current player does, the current
player moves again (a pass is forced).
If neither has moves, the game is over.
"""
def get_next_player(board, current_player):

    opponent = -current_player
    if get_legal_moves(board, opponent):
        return opponent
    elif get_legal_moves(board, current_player):
        return current_player   # Opponent must pass
    else:
        return None             # Game over


# Display utility

"""
Print the board to stdout in a human-readable format.
"""
def display_board(board):

    symbols = {BLACK: 'B', WHITE: 'W', EMPTY: '.'}
    print("  " + " ".join(str(c) for c in range(BOARD_SIZE)))
    for r in range(BOARD_SIZE):
        row_str = " ".join(symbols[board[r][c]] for c in range(BOARD_SIZE))
        print(f"{r} {row_str}")
    black_count, white_count = get_score(board)
    print(f"Score — Black: {black_count}  White: {white_count}\n")


if __name__ == "__main__":
    board = get_initial_board()
    print("=== Initial board ===")
    display_board(board)

    # Verify starting legal moves for black (should be 4)
    moves = get_legal_moves(board, BLACK)
    print(f"Black's legal moves: {moves}")
    assert len(moves) == 4, "Black should have exactly 4 moves at the start."

    # Apply one move and display the result
    first_move = moves[0]
    board = apply_move(board, first_move, BLACK)
    print(f"\n=== After Black plays {first_move} ===")
    display_board(board)