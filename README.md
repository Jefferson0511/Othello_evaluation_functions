# Comparing Evaluation Function Paradigms in Othello

**Foundations of Artificial Intelligence — Final Project**
Jefferson David | Northeastern University | Spring 2026

---

## Overview

This project compares three approaches to building evaluation functions for an Othello-playing agent:

1. **Hand-crafted heuristic** — weights designed using domain knowledge (corner control, stability, mobility, piece count)
2. **GA-evolved heuristic** — a genetic algorithm evolves optimal weights for the same features through tournament play
3. **Neural network heuristic** — a PyTorch MLP trained to evaluate board positions through self-play

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
├── nn_training.py          # Self-play training loop             
├── tournament.py         
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

Each module can be run individually to verify it works correctly.

**Test the game engine:**
```bash
python othello.py
```
Expected output: starting board, black's 4 legal opening moves, board after first move.

**Test the search engine:**
```bash
python search.py
```
Expected output: a complete game simulated at depth 3, with move-by-move board display and final result.

**Test the hand-crafted evaluator:**
```bash
python heuristics.py
```
Expected output: starting position scores 0.00 (symmetric), custom weight factory verified, score after one move displayed.

---

## Current Status

| Component | Status |
|---|---|
| Game engine (`othello.py`) | ✅ Complete |
| Search (`search.py`) | ✅ Complete |
| Hand-crafted heuristic (`heuristics.py`) | ✅ Complete |
| Genetic algorithm (`genetic_algorithm.py`) | ✅ Complete |
| Neural network (`neural_network.py`) | ✅ Complete |
| NN training (`nn_training.py`) | ✅ Complete |
| Tournament (`tournament.py`) | ✅ Complete |

---

## Design Decisions

**Board encoding:** 1 = black, -1 = white, 0 = empty. Using -1 for white means any piece can be flipped by multiplying by -1, and the neural network input encoding is clean and symmetric.

**Shared search interface:** All three evaluation functions share the same `(board, player) → float` signature, so the same minimax engine runs all three agents without modification.

**GA weight interface:** `heuristics.py` exposes a `make_weighted_eval(weights)` factory that converts a 4-element weight vector into a full evaluation function. The GA evolves these weights without needing to know anything about the feature implementations.

---

## References

Alliot, J.M., & Durand, N. (1996). A genetic algorithm to improve an Othello program. *Artificial Evolution (AE 1995), LNCS Vol. 1063*. Springer, pp. 307–319.

Tesauro, G. (1995). Temporal difference learning and TD-Gammon. *Communications of the ACM, 38*(3), 58–68.