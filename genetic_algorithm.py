import numpy as np
import random
import copy
from othello import BLACK, WHITE, EMPTY, get_initial_board, apply_move, get_legal_moves
from heuristics import make_weighted_eval, hand_crafted_eval, CORNER_WEIGHT, STABILITY_WEIGHT, MOBILITY_WEIGHT, PIECE_WEIGHT
from search import play_game, DEFAULT_DEPTH

# GA Hyperparameters
POPULATION_SIZE  = 50       # Number of individuals per generation
NUM_GENERATIONS  = 100      # Number of generations to evolve
CROSSOVER_PROB   = 0.9      # Probability of applying crossover (Pc)
MUTATION_PROB    = 0.15     # Probability of mutating each gene (Pm)

# Gaussian mutation standard deviations per weight
MUTATION_STD = [50.0, 20.0, 10.0, 5.0]   # [corner, stability, mobility, piece]

# Fitness evaluation settings
NUM_POSITIONS    = 5        # Random starting positions per fitness evaluation
EVAL_DEPTH       = 3        # Search depth used during fitness games

# Simulated annealing: acceptance probability decays linearly across generations
SA_START_PROB    = 0.5
SA_END_PROB      = 0.01

# Weight bounds
WEIGHT_BOUNDS = [
    (1.0,   200.0),    # corner weight
    (0.0,   100.0),    # stability weight
    (0.0,    50.0),    # mobility weight
    (0.0,    20.0),    # piece count weight
]



# Population initialisation
def _random_individual():
    """Return one individual: 4 random weights within their bounds."""
    return [random.uniform(lo, hi) for lo, hi in WEIGHT_BOUNDS]


def initialise_population(size):
    """
    Create an initial population of `size` individuals.
    One slot is seeded with the hand-crafted baseline weights so the GA
    always has a known-good starting point in the gene pool.
    """
    population = [_random_individual() for _ in range(size - 1)]
    population.append([CORNER_WEIGHT, STABILITY_WEIGHT,
                        MOBILITY_WEIGHT, PIECE_WEIGHT])
    return population


# Random starting position generator
def _random_starting_position():
    """
    Generate a randomised mid-game board for fitness evaluation.

    Plays 8-18 random legal moves from the standard opening, following
    Alliot & Durand's methodology to prevent deterministic agents from
    always replaying the same game.

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


# Fitness evaluation
def evaluate_fitness(weights, reference_eval_fn, prev_fitness=None):
    """
    Evaluate fitness by playing NUM_POSITIONS × 2 games against the
    reference evaluation function from randomised starting positions.

    Fitness formula (Alliot & Durand 1996):
        F_current = (wins×2 + draws×1) / total_games
                    + disc_differential / (total_games × 1000)

    Smoothed with previous generation's fitness to reduce noise:
        F_real = (F_current + F_prev) / 2

    Parameters
    ----------
    weights           : list of 4 floats
    reference_eval_fn : fixed baseline evaluation function
    prev_fitness      : fitness from previous generation (None = first gen)

    Returns
    -------
    float : smoothed fitness value
    """
    candidate_eval = make_weighted_eval(weights)
    wins      = 0
    draws     = 0
    disc_diff = 0.0
    total     = NUM_POSITIONS * 2

    for _ in range(NUM_POSITIONS):
        for candidate_color in (BLACK, WHITE):
            if candidate_color == BLACK:
                winner, b, w = play_game(
                    candidate_eval, reference_eval_fn,
                    black_depth=EVAL_DEPTH, white_depth=EVAL_DEPTH
                )
            else:
                winner, b, w = play_game(
                    reference_eval_fn, candidate_eval,
                    black_depth=EVAL_DEPTH, white_depth=EVAL_DEPTH
                )

            c_score = b if candidate_color == BLACK else w
            o_score = w if candidate_color == BLACK else b

            if winner == candidate_color:
                wins += 1
            elif winner == EMPTY:
                draws += 1
            disc_diff += c_score - o_score

    f_current = (wins * 2 + draws) / total + disc_diff / (total * 1000)
    return (f_current + prev_fitness) / 2 if prev_fitness is not None else f_current


# Selection
def tournament_selection(population, fitnesses):
    """
    Select one individual via binary tournament selection.
    Two random individuals compete; the fitter one is returned.
    """
    i, j = random.sample(range(len(population)), 2)
    return copy.deepcopy(population[i if fitnesses[i] >= fitnesses[j] else j])


# Crossover
def barycentric_crossover(p1, p2):
    """
    Stochastic barycentric crossover (Alliot & Durand 1996).

    α ~ Uniform(-0.5, 1.5), crossover point k ~ Uniform(1, n-1).
    The k-th gene is blended:
        child1[k] = α×p1[k] + (1-α)×p2[k]
        child2[k] = (1-α)×p1[k] + α×p2[k]

    Returns two children with genes clipped to their bounds.
    """
    alpha  = random.uniform(-0.5, 1.5)
    k      = random.randint(1, len(p1) - 1)
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    c1[k]  = alpha * p1[k] + (1 - alpha) * p2[k]
    c2[k]  = (1 - alpha) * p1[k] + alpha * p2[k]
    for i in range(len(c1)):
        lo, hi = WEIGHT_BOUNDS[i]
        c1[i]  = max(lo, min(hi, c1[i]))
        c2[i]  = max(lo, min(hi, c2[i]))
    return c1, c2


# Mutation
def gaussian_mutation(individual):
    """
    Add zero-centred Gaussian noise to each gene with probability
    MUTATION_PROB, using per-gene standard deviations from MUTATION_STD.
    Results are clipped to WEIGHT_BOUNDS.
    """
    mutant = copy.deepcopy(individual)
    for i in range(len(mutant)):
        if random.random() < MUTATION_PROB:
            mutant[i] += random.gauss(0, MUTATION_STD[i])
            lo, hi     = WEIGHT_BOUNDS[i]
            mutant[i]  = max(lo, min(hi, mutant[i]))
    return mutant


# Simulated annealing acceptance
def _sa_accept(child_fitness, parent_fitness, generation):
    """
    Accept a child if it is better than its parent.
    Otherwise accept with a linearly decaying probability — allows early
    exploration and gradually tightens selection pressure (Alliot & Durand).
    """
    if child_fitness >= parent_fitness:
        return True
    t = generation / max(NUM_GENERATIONS - 1, 1)
    return random.random() < (SA_START_PROB + t * (SA_END_PROB - SA_START_PROB))


# Main GA loop
def run_ga(reference_eval_fn=None, verbose=True):
    """
    Run the genetic algorithm and return the best evolved weights.

    Each generation:
      1. Evaluate fitness of all parents (smoothed with previous gen)
      2. Select pairs via tournament selection
      3. Apply crossover and mutation to produce children
      4. Evaluate child fitness ONCE and reuse for SA acceptance
         (avoids the cost of double-evaluating each child)
      5. Accept or reject children via SA criterion
      6. Carry the best individual forward unchanged (elitism)

    Parameters
    ----------
    reference_eval_fn : baseline to compete against (default: hand_crafted_eval)
    verbose           : print per-generation statistics if True

    Returns
    -------
    best_weights : list of 4 floats
    best_fitness : float
    history      : list of (mean_fitness, best_fitness) per generation
    """
    if reference_eval_fn is None:
        reference_eval_fn = hand_crafted_eval

    population     = initialise_population(POPULATION_SIZE)
    prev_fitnesses = [None] * POPULATION_SIZE
    best_weights   = None
    best_fitness   = -float('inf')
    history        = []

    for generation in range(NUM_GENERATIONS):
        fitnesses = [
            evaluate_fitness(ind, reference_eval_fn, prev_fitnesses[i])
            for i, ind in enumerate(population)
        ]
        prev_fitnesses = fitnesses[:]

        # --- Track global best ---
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_weights = copy.deepcopy(population[gen_best_idx])

        mean_fit = float(np.mean(fitnesses))
        history.append((mean_fit, fitnesses[gen_best_idx]))

        if verbose:
            bw = [round(w, 2) for w in population[gen_best_idx]]
            print(f"Gen {generation + 1:3d}/{NUM_GENERATIONS} | "
                  f"Mean: {mean_fit:.4f} | "
                  f"Best: {fitnesses[gen_best_idx]:.4f} | "
                  f"Weights: {bw}")

        next_population  = [copy.deepcopy(population[gen_best_idx])]
        next_fitnesses   = [fitnesses[gen_best_idx]]

        while len(next_population) < POPULATION_SIZE:
            # Select parents
            p1_idx = random.randrange(POPULATION_SIZE)
            p2_idx = random.randrange(POPULATION_SIZE)
            p1 = copy.deepcopy(population[p1_idx if fitnesses[p1_idx] >= fitnesses[p2_idx] else p2_idx])

            p3_idx = random.randrange(POPULATION_SIZE)
            p4_idx = random.randrange(POPULATION_SIZE)
            p2 = copy.deepcopy(population[p3_idx if fitnesses[p3_idx] >= fitnesses[p4_idx] else p4_idx])

            p1_fit = fitnesses[p1_idx if fitnesses[p1_idx] >= fitnesses[p2_idx] else p2_idx]
            p2_fit = fitnesses[p3_idx if fitnesses[p3_idx] >= fitnesses[p4_idx] else p4_idx]

            # Crossover
            if random.random() < CROSSOVER_PROB:
                c1, c2 = barycentric_crossover(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

            # Mutation
            c1 = gaussian_mutation(c1)
            c2 = gaussian_mutation(c2)

            # Evaluate children ONCE — reuse for SA acceptance
            c1_fit = evaluate_fitness(c1, reference_eval_fn)
            c2_fit = evaluate_fitness(c2, reference_eval_fn)

            # SA acceptance
            accepted1 = c1 if _sa_accept(c1_fit, p1_fit, generation) else p1
            fit1      = c1_fit if _sa_accept(c1_fit, p1_fit, generation) else p1_fit
            accepted2 = c2 if _sa_accept(c2_fit, p2_fit, generation) else p2
            fit2      = c2_fit if _sa_accept(c2_fit, p2_fit, generation) else p2_fit

            next_population.append(accepted1)
            next_fitnesses.append(fit1)
            if len(next_population) < POPULATION_SIZE:
                next_population.append(accepted2)
                next_fitnesses.append(fit2)

        population     = next_population
        prev_fitnesses = next_fitnesses

    if verbose:
        print(f"\n=== GA complete ===")
        print(f"Best weights : {[round(w, 2) for w in best_weights]}")
        print(f"Best fitness : {best_fitness:.4f}")

    return best_weights, best_fitness, history


# sanity test
if __name__ == "__main__":
    POPULATION_SIZE = 4
    NUM_GENERATIONS = 2
    NUM_POSITIONS   = 2  
    EVAL_DEPTH      = 1    

    print("Running GA sanity check (4 individuals, 2 generations, depth 1)...")
    print("Full run uses the hyperparameters defined at the top of the file.\n")

    best_weights, best_fitness, history = run_ga(verbose=True)

    assert best_weights is not None, "GA must return a valid weight vector."
    assert len(best_weights) == 4,   "Weight vector must have exactly 4 elements."
    assert all(lo <= w <= hi for w, (lo, hi) in zip(best_weights, WEIGHT_BOUNDS)), \
        "All weights must be within their bounds."

    print(f"\nBest weights : {[round(w, 2) for w in best_weights]}")
    print(f"Best fitness : {best_fitness:.4f}")
    print("All checks passed.")