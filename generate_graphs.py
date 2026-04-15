import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

OUTPUT_DIR = "graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family"  : "serif",
    "font.size"    : 11,
    "axes.titlesize" : 12,
    "axes.labelsize" : 11,
    "legend.fontsize": 10,
    "figure.dpi"   : 150,
})

def plot_ga_fitness():
    """
    Line plot of mean and best fitness across GA generations.
    Data hardcoded from the completed GA run.
    """
    generations = list(range(1, 51))
    mean_fits = [
        0.8999, 1.2102, 1.4776, 1.6353, 1.7845,
        1.9571, 2.0242, 1.9636, 2.0503, 2.0084,
        1.9882, 2.0312, 2.0530, 2.0324, 2.0323,
        2.0062, 2.0533, 2.0010, 2.0326, 2.0525,
        1.9904, 2.0329, 2.0440, 2.0230, 2.0237,
        2.0437, 2.0337, 2.0332, 2.0434, 2.0430,
        2.0338, 2.0237, 2.0333, 2.0331, 2.0546,
        2.0235, 2.0440, 2.0339, 2.0546, 2.0336,
        2.0548, 2.0547, 2.0444, 2.0133, 2.0548,
        2.0547, 2.0550, 2.0550, 2.0550, 2.0550
    ]
    best_fits = [
        2.0410, 2.0440, 2.0490, 2.0530, 2.0530,
        2.0530, 2.0540, 2.0540, 2.0540, 2.0540,
        2.0550, 2.0550, 2.0550, 2.0550, 2.0550,
        2.0550, 2.0550, 2.0550, 2.0550, 2.0550,
        2.0550, 2.0550, 2.0550, 2.0550, 2.0550,
        2.0550, 2.0550, 2.0550, 2.0550, 2.0550,
        2.0550, 2.0550, 2.0550, 2.0550, 2.0550,
        2.0550, 2.0550, 2.0550, 2.0550, 2.0550,
        2.0550, 2.0550, 2.0550, 2.0550, 2.0550,
        2.0550, 2.0550, 2.0550, 2.0550, 2.0550
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(generations, mean_fits, label="Mean fitness",
            color="#4878CF", linewidth=1.8, linestyle="--")
    ax.plot(generations, best_fits, label="Best fitness",
            color="#D65F5F", linewidth=1.8)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("GA Fitness Progression Across 50 Generations")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure1_ga_fitness.png")
    plt.savefig(path)
    plt.close()
    print(f"Figure 1 saved to {path}")


def plot_nn_convergence():

    games      = [50,   100,  150,  200,  250,  300,  350,  400,  450,  500]
    black_wins = [66.0, 69.0, 60.7, 53.5, 49.2, 48.0, 47.7, 48.8, 48.9, 50.6]
    white_wins = [30.0, 27.0, 35.3, 43.0, 47.6, 48.7, 48.9, 47.8, 47.3, 45.6]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(games, black_wins, label="Black win %",
            color="#2C2C2C", linewidth=1.8, marker="o", markersize=4)
    ax.plot(games, white_wins, label="White win %",
            color="#999999", linewidth=1.8, marker="s", markersize=4,
            linestyle="--")

    ax.axhline(50, color="#AAAAAA", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Training games")
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Neural Network Win Rate Convergence During Self-Play Training")
    ax.set_ylim(20, 80)
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure2_nn_convergence.png")
    plt.savefig(path)
    plt.close()
    print(f"Figure 2 saved to {path}")

def plot_tournament_results():

    agents    = ["Hand-Crafted", "GA-Evolved", "Neural-Net"]
    win_rates = [41.2, 50.6, 58.1]

    colors = ["#4878CF", "#D65F5F", "#6ACC65"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(agents, win_rates, color=colors,
                   edgecolor="white", height=0.5)

    for bar, val in zip(bars, win_rates):
        ax.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)

    ax.axvline(50, color="#AAAAAA", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Win rate (%)")
    ax.set_title("Tournament Win Rates: Round-Robin at Depth 5")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure3_tournament.png")
    plt.savefig(path)
    plt.close()
    print(f"Figure 3 saved to {path}")

if __name__ == "__main__":
    print("Generating graphs...\n")

    plot_ga_fitness()

    plot_nn_convergence()

    try:
        plot_tournament_results()
    except NameError:
        print("Skipping Figure 3 — replace INSERT placeholders "
              "in plot_tournament_results() with actual win rates.")

    print("\nDone. Check the graphs/ folder.")