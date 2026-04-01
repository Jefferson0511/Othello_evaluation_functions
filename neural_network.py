import torch
import torch.nn as nn
import numpy as np
from othello import BLACK, WHITE, EMPTY, BOARD_SIZE, is_game_over, get_score

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Input encoding
def board_to_input(board, player):
    """
    Convert a board state into a normalised input tensor for the network.

    Encoding (relative to the current player):
        +1  = current player's disc
        -1  = opponent's disc
         0  = empty square

    By encoding relative to the player rather than using absolute
    black/white values, the same network weights work regardless of
    which colour the agent is playing.  This halves the effective
    learning problem compared to a colour-absolute encoding.

    Parameters
    ----------
    board  : 8x8 numpy array  (BLACK=1, WHITE=-1, EMPTY=0)
    player : BLACK or WHITE

    Returns
    -------
    torch.Tensor of shape (64,) on DEVICE, dtype float32
    """
    relative_board = board if player == BLACK else -board
    return torch.tensor(
        relative_board.flatten(), dtype=torch.float32, device=DEVICE
    )


# Network definition
class OthelloNet(nn.Module):
    """
    Multilayer perceptron for Othello position evaluation.

    Architecture (inspired by Tesauro 1995 and Alliot & Durand 1996):
        Input  : 64 neurons  — flattened 8×8 board, player-relative encoding
        Hidden1: 128 neurons — ReLU activation
        Hidden2:  64 neurons — ReLU activation
        Output :   1 neuron  — Tanh activation → score in (-1, 1)

    The output is interpreted as:
        +1  ≈ certain win for the current player
        -1  ≈ certain loss for the current player
         0  ≈ roughly equal position

    Tanh is chosen over sigmoid for the output because it is zero-centred,
    making the training signal symmetric for wins and losses.  ReLU hidden
    activations provide faster training than the sigmoid units used in the
    original TD-Gammon — an update consistent with modern practice.
    """

    def __init__(self):
        super(OthelloNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),                        
            nn.ReLU(),
            nn.Linear(64, 1),                          
            nn.Tanh()                                  
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, 64) or (64,)

        Returns
        -------
        torch.Tensor of shape (batch_size, 1) or (1,)
        """
        return self.network(x)

    def evaluate(self, board, player):
        """
        Evaluate a single board position and return a scalar float.

        This is the low-level evaluation method called by make_nn_eval.
        Runs in inference mode (no gradient tracking) for efficiency.

        Parameters
        ----------
        board  : 8x8 numpy array
        player : BLACK or WHITE

        Returns
        -------
        float : evaluation score in (-1, 1), higher = better for player
        """
        self.eval()
        with torch.no_grad():
            x     = board_to_input(board, player).unsqueeze(0) 
            score = self.forward(x)                        
            return float(score.item())


# Eval function factory
def make_nn_eval(model):
    """
    Wrap a trained OthelloNet model in an evaluation function compatible
    with search.py's (board, player) → float interface.

    This is the neural network's drop-in replacement for hand_crafted_eval
    and make_weighted_eval — all three share the same signature so the
    same minimax search engine runs all three agents unchanged.

    Parameters
    ----------
    model : OthelloNet instance (trained or untrained)

    Returns
    -------
    callable(board, player) → float
    """
    def nn_eval(board, player):
        if is_game_over(board):
            black_count, white_count = get_score(board)
            player_count   = black_count if player == BLACK else white_count
            opponent_count = white_count if player == BLACK else black_count
            if player_count > opponent_count:
                return 1.0
            elif player_count < opponent_count:
                return -1.0
            else:
                return 0.0

        return model.evaluate(board, player)

    return nn_eval


# Model persistence
def save_model(model, path="othello_net.pth"):
    """
    Save the model's state dictionary to disk.

    Parameters
    ----------
    model : OthelloNet instance
    path  : file path to save to (default: othello_net.pth)
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path="othello_net.pth"):
    """
    Load a previously saved OthelloNet from disk.

    Parameters
    ----------
    path : file path to load from

    Returns
    -------
    OthelloNet instance with loaded weights, set to eval mode
    """
    model = OthelloNet().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {path}")
    return model


# sanity test
if __name__ == "__main__":
    from othello import get_initial_board, apply_move, get_legal_moves, display_board

    print(f"Using device: {DEVICE}")

    # --- Test 1: Model initialisation ---
    print("\n=== Test 1: Model initialisation ===")
    model = OthelloNet().to(DEVICE)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # --- Test 2: board_to_input encoding ---
    print("\n=== Test 2: Input encoding ===")
    board = get_initial_board()
    x_black = board_to_input(board, BLACK)
    x_white = board_to_input(board, WHITE)
    print(f"Input tensor shape : {x_black.shape}")
    print(f"Black encoding sum : {x_black.sum().item():.1f}  (expected 0.0 — symmetric)")
    print(f"White encoding sum : {x_white.sum().item():.1f}  (expected 0.0 — symmetric)")
    assert x_black.shape == torch.Size([64]), "Input must be shape (64,)"
    assert float(x_black.sum()) == 0.0, "Starting board is symmetric — sum should be 0."

    # --- Test 3: Forward pass ---
    print("\n=== Test 3: Forward pass ===")
    score_black = model.evaluate(board, BLACK)
    score_white = model.evaluate(board, WHITE)
    print(f"Untrained eval for Black : {score_black:.4f}  (expected near 0)")
    print(f"Untrained eval for White : {score_white:.4f}  (expected near 0)")
    assert -1.0 <= score_black <= 1.0, "Output must be in (-1, 1)"
    assert -1.0 <= score_white <= 1.0, "Output must be in (-1, 1)"

    # --- Test 4: make_nn_eval interface ---
    print("\n=== Test 4: make_nn_eval interface ===")
    nn_eval = make_nn_eval(model)
    score   = nn_eval(board, BLACK)
    print(f"nn_eval(board, BLACK) = {score:.4f}")
    assert isinstance(score, float), "nn_eval must return a Python float."

    # --- Test 5: Terminal state handling ---
    print("\n=== Test 5: Terminal state handling ===")
    from othello import is_game_over
    score_terminal = nn_eval(board, BLACK)
    print(f"Non-terminal score: {score_terminal:.4f}  (network output)")

    # --- Test 6: Save and load ---
    print("\n=== Test 6: Save and load ===")
    save_model(model, "test_model.pth")
    loaded_model = load_model("test_model.pth")
    score_loaded = make_nn_eval(loaded_model)(board, BLACK)
    print(f"Score before save : {score_black:.6f}")
    print(f"Score after load  : {score_loaded:.6f}")
    assert abs(score_black - score_loaded) < 1e-6, "Loaded model must produce identical output."

    import os
    os.remove("test_model.pth")

    print("\nAll checks passed.")