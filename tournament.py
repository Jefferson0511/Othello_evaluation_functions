import os
import numpy as np
import random
from othello import (
    BLACK, WHITE, EMPTY, get_initial_board, apply_move,
    get_legal_moves, get_score
)
from genetic_algorithm import load_weights
from neural_network import OthelloNet, make_nn_eval, load_model, DEVICE
from search import play_game, DEFAULT_DEPTH

GAMES_PER_MATCHUP = 20 
TOURNAMENT_DEPTH  = 5  

from heuristics import hand_crafted_eval, make_weighted_eval
GA_WEIGHTS_PATH   = "ga_weights.npy"
NN_MODEL_PATH     = "othello_net.pth"


def _random_start():
    """
    Generate a randomised mid-game starting position.

    Plays 8-18 random legal moves from the standard opening so that
    deterministic minimax agents face different positions each game —
    directly following Alliot & Durand's fitness evaluation methodology.

    Returns (board, current_player).
    """
    board  = get_initial_board()
    player = BLACK
    for _ in range(random.randint(8, 18)):
        moves = get_legal_moves(board, player)
        if not moves:
            player = -player
            moves  = get_legal_moves(board, player)
            if not moves:
                break
        board  = apply_move(board, random.choice(moves), player)
        player = -player
    return board, player


def run_matchup(name_a, eval_a, name_b, eval_b,
                num_games=GAMES_PER_MATCHUP, depth=TOURNAMENT_DEPTH,
                verbose=True):

    wins_a    = 0
    wins_b    = 0
    draws     = 0
    disc_diff = 0.0

    if verbose:
        print(f"\n  {name_a} (B) vs {name_b} (W) — {num_games} games")
        print(f"  {'Game':<6} {'Winner':<18} {'Score (B-W)'}")
        print(f"  {'-'*42}")

    for g in range(1, num_games + 1):
        start_board, start_player = _random_start()

        winner, b_score, w_score = play_game(
            eval_a, eval_b,
            black_depth=depth, white_depth=depth,
            start_board=start_board, start_player=start_player,
            verbose=False
        )

        if winner == BLACK:
            wins_a += 1
            result  = name_a
        elif winner == WHITE:
            wins_b += 1
            result  = name_b
        else:
            draws  += 1
            result  = "Draw"

        disc_diff += b_score - w_score

        if verbose:
            print(f"  {g:<6} {result:<18} {b_score}-{w_score}")

    total      = wins_a + wins_b + draws
    win_rate_a = (wins_a + 0.5 * draws) / total 

    return {
        "wins_a"       : wins_a,
        "wins_b"       : wins_b,
        "draws"        : draws,
        "win_rate_a"   : win_rate_a,
        "avg_disc_diff": disc_diff / total
    }


def run_tournament(agents, num_games=GAMES_PER_MATCHUP,
                   depth=TOURNAMENT_DEPTH, verbose=True):
    """
    Run a full round-robin tournament between all agents.

    Every ordered pair (A, B) plays num_games games with A as BLACK and
    B as WHITE, then the roles are swapped for another num_games games.
    This controls for any first-mover advantage.

    Parameters
    ----------
    agents    : list of (name, eval_fn) tuples
    num_games : games per ordered pair
    depth     : minimax search depth

    Returns
    -------
    results : dict mapping agent name → dict of tournament statistics
    """
    names     = [a[0] for a in agents]
    eval_fns  = {a[0]: a[1] for a in agents}
    n         = len(agents)

    results = {
        name: {
            "wins"      : 0,
            "losses"    : 0,
            "draws"     : 0,
            "points"    : 0.0, 
            "disc_diff" : 0.0,
            "games"     : 0
        }
        for name in names
    }

    matchup_log = []

    if verbose:
        print("=" * 60)
        print("  OTHELLO EVALUATION FUNCTION TOURNAMENT")
        print(f"  Agents: {', '.join(names)}")
        print(f"  Games per ordered pair: {num_games}  |  Depth: {depth}")
        print("=" * 60)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            name_a  = names[i]
            name_b  = names[j]
            eval_a  = eval_fns[name_a]
            eval_b  = eval_fns[name_b]

            if verbose:
                print(f"\nMatchup: {name_a} vs {name_b}")

            res = run_matchup(name_a, eval_a, name_b, eval_b,
                              num_games, depth, verbose)
            matchup_log.append((name_a, name_b, res))

            results[name_a]["wins"]       += res["wins_a"]
            results[name_a]["losses"]     += res["wins_b"]
            results[name_a]["draws"]      += res["draws"]
            results[name_a]["points"]     += res["wins_a"] + 0.5 * res["draws"]
            results[name_a]["disc_diff"]  += res["avg_disc_diff"] * num_games
            results[name_a]["games"]      += num_games

            results[name_b]["wins"]       += res["wins_b"]
            results[name_b]["losses"]     += res["wins_a"]
            results[name_b]["draws"]      += res["draws"]
            results[name_b]["points"]     += res["wins_b"] + 0.5 * res["draws"]
            results[name_b]["disc_diff"]  -= res["avg_disc_diff"] * num_games
            results[name_b]["games"]      += num_games

    if verbose:
        print("\n" + "=" * 60)
        print("  FINAL STANDINGS")
        print("=" * 60)
        print(f"  {'Agent':<20} {'W':>4} {'L':>4} {'D':>4} "
              f"{'Points':>8} {'Win%':>7} {'AvgDisc':>9}")
        print(f"  {'-'*58}")

        sorted_agents = sorted(
            results.items(),
            key=lambda x: x[1]["points"],
            reverse=True
        )

        for name, r in sorted_agents:
            total    = r["games"]
            win_pct  = 100 * r["points"] / total if total > 0 else 0
            avg_disc = r["disc_diff"] / total if total > 0 else 0
            print(f"  {name:<20} {r['wins']:>4} {r['losses']:>4} "
                  f"{r['draws']:>4} {r['points']:>8.1f} "
                  f"{win_pct:>6.1f}%  {avg_disc:>+8.2f}")

        print("=" * 60)

    return results, matchup_log


def load_agents(ga_weights=None, nn_path=NN_MODEL_PATH):
    agents = []

    agents.append(("Hand-Crafted", hand_crafted_eval))

    if ga_weights is not None:
        ga_eval = make_weighted_eval(ga_weights)
        agents.append(("GA-Evolved", ga_eval))
    elif os.path.exists(GA_WEIGHTS_PATH):
        ga_weights = load_weights(GA_WEIGHTS_PATH)
        agents.append(("GA-Evolved", make_weighted_eval(ga_weights)))
    else:
        print("[Warning] No GA weights found — using stand-in weights.")
        print("          Run: python genetic_algorithm.py --full")
        agents.append(("GA-Evolved", make_weighted_eval([128.0, 17.0, 20.0, 1.4])))

    if os.path.exists(nn_path):
        nn_model = load_model(nn_path)
        agents.append(("Neural-Net", make_nn_eval(nn_model)))
    else:
        print(f"[Warning] No model found at {nn_path}. "
              f"Run nn_training.py first.")

    return agents

if __name__ == "__main__":
    import sys

    full_run = "--full" in sys.argv

    if full_run:
        print("Starting FULL tournament...")
        print(f"Games per matchup: {GAMES_PER_MATCHUP}  |  Depth: {TOURNAMENT_DEPTH}\n")
        agents = load_agents(ga_weights=None, nn_path=NN_MODEL_PATH)
        results, log = run_tournament(
            agents,
            num_games=GAMES_PER_MATCHUP,
            depth=TOURNAMENT_DEPTH,
            verbose=True
        )
        print("\nFull tournament complete.")

    else:
        print("Running tournament sanity check (3 games per matchup, depth 2)...\n")

        agents = load_agents(ga_weights=None, nn_path=NN_MODEL_PATH)

        results, log = run_tournament(
            agents,
            num_games=3,
            depth=2,
            verbose=True
        )

        for name, r in results.items():
            assert r["games"] > 0, f"{name} must have played at least one game."
            assert r["wins"] + r["losses"] + r["draws"] == r["games"], \
                f"{name} win/loss/draw counts must sum to total games."

        print("\nSanity checks passed — tournament is working correctly.")
        print("\nTo run the full tournament:")
        print("  1. Run: python genetic_algorithm.py --full")
        print("  2. Run: python nn_training.py --full")
        print("  3. Run: python tournament.py --full")