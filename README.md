# Comparing Evaluation Function Paradigms in Othello

**Foundations of Artificial Intelligence — Final Project**
Jefferson David Kingston

---

## Overview

This project compares three approaches to building evaluation functions for an Othello-playing agent:

1. **Hand-crafted heuristic** — weights designed using domain knowledge (corner control, stability, mobility, piece count)
2. **GA-evolved heuristic** — a genetic algorithm evolves optimal weights for the same features through tournament play
3. **Neural network heuristic** — a PyTorch MLP trained to evaluate board positions through self-play using TD(λ)

All three agents share the same minimax search engine with alpha-beta pruning. A round-robin tournament evaluates and compares their performance.

---

## Project Structure

```
othello-eval-comparison/
│
├── othello.py              # Game engine: board, legal moves, rules, win detection
├── search.py               # Minimax with alpha-beta pruning, game simulation
├── heuristics.py           # Hand-crafted evaluation function + GA weight interface
├── genetic_algorithm.py    # GA evolution of evaluation weights
├── neural_network.py       # PyTorch MLP board evaluator
├── nn_training.py          # Self-play TD(λ) training loop
├── tournament.py           # Round-robin tournament evaluation
├── generate_graphs.py      # Matplotlib graphs for paper figures
│
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Installation

**Requirements:** Python 3.8+

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Code

```bash
python othello.py
python search.py
python heuristics.py
python genetic_algorithm.py
python neural_network.py
python nn_training.py
python tournament.py
```

### Full experimental runs
```bash
# Step 1: Evolve GA weights (2-7 hours)
python genetic_algorithm.py --full

# Step 2: Train neural network (1-2 hours)
python nn_training.py --full

# Step 3: Run full tournament (6-10 hours)
python tournament.py --full

# Step 4: Generate paper figures
python generate_graphs.py
```

Figures are saved to the `graphs/` folder as PNG files.

---

## Current Status

| Component | Status |
|---|---|
| Game engine (`othello.py`) |
| Search (`search.py`) |
| Hand-crafted heuristic (`heuristics.py`) |
| Genetic algorithm (`genetic_algorithm.py`) |
| Neural network (`neural_network.py`) |
| NN training (`nn_training.py`) |
| Tournament (`tournament.py`) |
| Graph generation (`generate_graphs.py`) | 

---

## Results Summary

Full experimental results from the round-robin tournament at search depth 5:

| Agent | Win% | Avg Disc Diff |
|---|---|---|
| Neural-Net | 58.1% | -3.12 |
| GA-Evolved | 50.6% | +3.59 |
| Hand-Crafted | 41.2% | -0.46 |

---

## Design Decisions

**Board encoding:** 1 = black, -1 = white, 0 = empty. Using -1 for white means any piece can be flipped by multiplying by -1, and the neural network input encoding is naturally symmetric.

**Shared search interface:** All three evaluation functions share the same `(board, player) → float` signature, so the same minimax engine runs all three agents without modification.

**GA weight interface:** `heuristics.py` exposes a `make_weighted_eval(weights)` factory that converts a 4-element weight vector into a full evaluation function. The GA evolves these weights without needing to know anything about the feature implementations.

---

## References

Alliot, J.M., & Durand, N. (1995). A genetic algorithm to improve an Othello program. *Artificial Evolution (AE 1995), Lecture Notes in Computer Science, Vol. 1063*. Springer, pp. 307–319.

Tesauro, G. (1995). Temporal difference learning and TD-Gammon. *Communications of the ACM, 38*(3), 58–68.