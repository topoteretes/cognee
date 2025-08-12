import json
import matplotlib.pyplot as plt
import numpy as np
import sys


def load_benchmark_data(filename):
    """Load benchmark data from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def visualize_benchmarks(benchmark_file, output_file=None):
    """Visualize benchmark results with error bars."""

    # Load data
    data = load_benchmark_data(benchmark_file)

    # Define metrics to plot
    metrics = ["Human-like Correctness", "DeepEval Correctness", "DeepEval EM", "DeepEval F1"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Setup plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get system names
    systems = [system["system"] for system in data]
    x_pos = np.arange(len(systems))

    # Plot each metric
    for i, metric in enumerate(metrics):
        means = []
        errors_lower = []
        errors_upper = []

        for system in data:
            if metric in system:
                means.append(system[metric])
                error_key = f"{metric} Error"
                if error_key in system:
                    errors_lower.append(system[metric] - system[error_key][0])
                    errors_upper.append(system[error_key][1] - system[metric])
                else:
                    errors_lower.append(0)
                    errors_upper.append(0)
            else:
                means.append(0)
                errors_lower.append(0)
                errors_upper.append(0)

        # Plot bars with error bars
        ax.bar(x_pos + i * 0.2, means, 0.2, label=metric, color=colors[i], alpha=0.8)

        # Add error bars
        for j, (mean, err_lower, err_upper) in enumerate(zip(means, errors_lower, errors_upper)):
            if mean > 0:  # Only show error bars for non-zero values
                ax.errorbar(
                    x_pos[j] + i * 0.2,
                    mean,
                    yerr=[[err_lower], [err_upper]],
                    fmt="none",
                    color="black",
                    capsize=3,
                    capthick=1,
                )

    # Customize plot
    ax.set_xlabel("Systems", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Benchmark Results", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + 0.3)  # Center the x-ticks
    ax.set_xticklabels(systems, rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved as {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    # Hardcoded benchmark files
    benchmark_file = "benchmark_summary_competition.json"
    # benchmark_file = "benchmark_summary_cognee.json"

    # Comment out which one you want to visualize
    # visualize_benchmarks(competition_file, competition_file.replace('.json', '.png'))
    visualize_benchmarks(benchmark_file, benchmark_file.replace(".json", ".png"))
