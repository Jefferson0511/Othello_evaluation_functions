import os
import numpy as np
import random
from othello import (
    BLACK, WHITE, EMPTY, get_initial_board, apply_move,
    get_legal_moves, get_score
)
from heuristics import hand_crafted_eval, make_weighted_eval
from neural_network import OthelloNet, make_nn_eval, load_model, DEVICE
from search import play_game, DEFAULT_DEPTH

# Tournament settings
GAMES_PER_MATCHUP = 20      # Games played per ordered pair (A vs B)
                            # Each matchup is played from randomised starting
                            # positions so deterministic agents produce
                            # statistically meaningful win rates
TOURNAMENT_DEPTH  = 5       # Search depth for all agents during tournament
                            # (deeper than training for stronger play)

GA_WEIGHTS_PATH   = None
NN_MODEL_PATH     = "othello_net.pth"


# Starting position randomiser
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


# Single matchup
def run_matchup(name_a, eval_a, name_b, eval_b,
                num_games=GAMES_PER_MATCHUP, depth=TOURNAMENT_DEPTH,
                verbose=True):
    """
    Play num_games games between agent A (BLACK) and agent B (WHITE)
    from randomised starting positions.

    Parameters
    ----------
    name_a, name_b   : display names for the two agents
    eval_a, eval_b   : evaluation functions for A and B
    num_games        : number of games to play
    depth            : minimax search depth for both agents
    verbose          : print per-game results if True

    Returns
    -------
    dict with keys:
        wins_a, wins_b, draws : int counts
        win_rate_a            : float in [0, 1]
        avg_disc_diff         : float — average final disc difference (A - B)
    """
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


# Round-robin tournament
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

    # Initialise result table
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

            # Update standings
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


# Load agents
def load_agents(ga_weights=None, nn_path=NN_MODEL_PATH):
    """
    Load all three evaluation agents.

    Parameters
    ----------
    ga_weights : list of 4 floats (evolved weights), or None to use a
                 stand-in (hand-crafted weights scaled slightly differently)
    nn_path    : path to a saved OthelloNet model file

    Returns
    -------
    list of (name, eval_fn) tuples ready for run_tournament
    """
    agents = []

    # Agent 1: Hand-crafted heuristic
    agents.append(("Hand-Crafted", hand_crafted_eval))

    # Agent 2: GA-evolved heuristic
    if ga_weights is not None:
        ga_eval = make_weighted_eval(ga_weights)
        agents.append(("GA-Evolved", ga_eval))
    else:
        print("[Warning] No GA weights provided — using stand-in weights.")
        stand_in = make_weighted_eval([128.0, 17.0, 20.0, 1.4])
        agents.append(("GA-Evolved", stand_in))

    # Agent 3: Neural network
    if os.path.exists(nn_path):
        nn_model = load_model(nn_path)
        agents.append(("Neural-Net", make_nn_eval(nn_model)))
    else:
        print(f"[Warning] No model found at {nn_path}. "
              f"Run nn_training.py first.")

    return agents


# ]sanity test]
if __name__ == "__main__":
    print("Running tournament sanity check (3 games per matchup, depth 2)...\n")

    agents = load_agents(ga_weights=None, nn_path=NN_MODEL_PATH)

    results, log = run_tournament(
        agents,
        num_games=3,       
        depth=2,            
        verbose=True
    )

    for name, r in results.items():
        assert r["games"] > 0,    f"{name} must have played at least one game."
        assert r["wins"] + r["losses"] + r["draws"] == r["games"], \
            f"{name} win/loss/draw counts must sum to total games."

    print("\nSanity checks passed — tournament is working correctly.")
    print("\nTo run the full tournament:")
    print("  1. Run genetic_algorithm.py and save evolved weights")
    print("  2. Run nn_training.py to train the neural network")
    print("  3. Call run_tournament() with GAMES_PER_MATCHUP=20, TOURNAMENT_DEPTH=5")