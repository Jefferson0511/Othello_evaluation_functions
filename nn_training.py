import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from othello import (
    BLACK, WHITE, EMPTY, get_initial_board, apply_move,
    get_legal_moves, get_next_player, is_game_over, get_winner, get_score
)
from neural_network import OthelloNet, board_to_input, make_nn_eval, save_model, DEVICE
from search import get_best_move

NUM_GAMES        = 500 
SEARCH_DEPTH     = 3 
LEARNING_RATE    = 0.001 
TD_LAMBDA        = 0.7

SAVE_INTERVAL    = 100      
MODEL_PATH       = "othello_net.pth"

OUTCOME_WIN  =  1.0
OUTCOME_DRAW =  0.0
OUTCOME_LOSS = -1.0


def td_update(model, optimizer, positions, outcome, player):

    if not positions:
        return

    model.train()
    n = len(positions)

    inputs = torch.stack([
        board_to_input(board, pl) for board, pl in positions
    ])

    with torch.no_grad():
        all_evals = model(inputs).squeeze(-1)

    targets = torch.zeros(n, device=DEVICE)
    for t in range(n):
        td_target = 0.0
        weight    = 1.0 - TD_LAMBDA
        for k in range(t + 1, n):
            td_target += weight * float(all_evals[k])
            weight    *= TD_LAMBDA
        td_target += weight * outcome
        targets[t] = td_target

    model.train()
    preds = model(inputs).squeeze(-1)
    loss  = nn.MSELoss()(preds, targets.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def self_play_game(model, depth=SEARCH_DEPTH):

    board          = get_initial_board()
    current_player = BLACK
    eval_fn        = make_nn_eval(model)

    positions_black = []
    positions_white = []

    while not is_game_over(board):
        legal_moves = get_legal_moves(board, current_player)

        if legal_moves:
            if current_player == BLACK:
                positions_black.append((board.copy(), BLACK))
            else:
                positions_white.append((board.copy(), WHITE))

            move  = get_best_move(board, current_player, eval_fn, depth)
            if move:
                board = apply_move(board, move, current_player)

        next_p = get_next_player(board, current_player)
        if next_p is None:
            break
        current_player = next_p

    winner = get_winner(board)
    return positions_black, positions_white, winner


def train(num_games=NUM_GAMES, depth=SEARCH_DEPTH, verbose=True):

    model     = OthelloNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    stats = {
        "black_wins" : 0,
        "white_wins" : 0,
        "draws"      : 0,
        "td_losses"  : [] 
    }

    if verbose:
        print(f"Training OthelloNet for {num_games} self-play games "
              f"at depth {depth}...")
        print(f"Device: {DEVICE} | LR: {LEARNING_RATE} | "
              f"TD_λ: {TD_LAMBDA}\n")

    for game_idx in range(1, num_games + 1):
        positions_black, positions_white, winner = self_play_game(model, depth)

        if winner == BLACK:
            outcome_black = OUTCOME_WIN
            outcome_white = OUTCOME_LOSS
            stats["black_wins"] += 1
        elif winner == WHITE:
            outcome_black = OUTCOME_LOSS
            outcome_white = OUTCOME_WIN
            stats["white_wins"] += 1
        else:
            outcome_black = OUTCOME_DRAW
            outcome_white = OUTCOME_DRAW
            stats["draws"] += 1

        loss_b = td_update(model, optimizer, positions_black,
                           outcome_black, BLACK)
        loss_w = td_update(model, optimizer, positions_white,
                           outcome_white, WHITE)

        avg_loss = ((loss_b or 0.0) + (loss_w or 0.0)) / 2
        stats["td_losses"].append(avg_loss)

        if verbose and game_idx % 50 == 0:
            total    = stats["black_wins"] + stats["white_wins"] + stats["draws"]
            bw_pct   = 100 * stats["black_wins"] / total
            ww_pct   = 100 * stats["white_wins"] / total
            d_pct    = 100 * stats["draws"]      / total
            avg_td   = float(np.mean(stats["td_losses"][-50:]))
            print(f"Game {game_idx:4d}/{num_games} | "
                  f"B wins: {bw_pct:.1f}%  W wins: {ww_pct:.1f}%  "
                  f"Draws: {d_pct:.1f}%  | Avg TD loss: {avg_td:.4f}")

        if game_idx % SAVE_INTERVAL == 0:
            ckpt_path = f"othello_net_game{game_idx}.pth"
            save_model(model, ckpt_path)

    save_model(model, MODEL_PATH)

    if verbose:
        total  = stats["black_wins"] + stats["white_wins"] + stats["draws"]
        print(f"\n=== Training complete ===")
        print(f"Black wins : {stats['black_wins']} ({100*stats['black_wins']/total:.1f}%)")
        print(f"White wins : {stats['white_wins']} ({100*stats['white_wins']/total:.1f}%)")
        print(f"Draws      : {stats['draws']}      ({100*stats['draws']/total:.1f}%)")
        print(f"Final TD loss (last 50 games): "
              f"{float(np.mean(stats['td_losses'][-50:])):.4f}")

    return model, stats


if __name__ == "__main__":
    import sys

    full_run = "--full" in sys.argv

    if full_run:
        print(f"Starting FULL NN training: {NUM_GAMES} self-play games "
              f"at depth {SEARCH_DEPTH}...")
        print("This may take 1–3 hours. Model will be saved to othello_net.pth\n")
        model, stats = train(num_games=NUM_GAMES, depth=SEARCH_DEPTH, verbose=True)
        print("\nFull NN training complete. Use othello_net.pth in tournament.py.")

    else:
        print("Running training sanity check (10 games at depth 1)...\n")
        model, stats = train(num_games=10, depth=1, verbose=True)

        assert model is not None,              "train() must return a model."
        assert len(stats["td_losses"]) == 10,  "Should have one loss entry per game."
        assert all(isinstance(l, float) for l in stats["td_losses"]), \
            "All TD losses must be floats."

        total = stats["black_wins"] + stats["white_wins"] + stats["draws"]
        assert total == 10, "Total games must equal num_games."

        from othello import get_initial_board
        board   = get_initial_board()
        nn_eval = make_nn_eval(model)
        score   = nn_eval(board, BLACK)
        assert -1.0 <= score <= 1.0, "Eval score must be in (-1, 1)."
        print(f"\nPost-training eval on start position: {score:.4f}")
        print("All sanity checks passed.")
        print("For the full run: python nn_training.py --full")