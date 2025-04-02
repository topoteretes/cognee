import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


def calculate_confidence_interval(accuracies, confidence=0.95):
    """Calculate mean and confidence interval for a list of accuracies."""
    if not accuracies:
        return 0, 0, 0

    mean = np.mean(accuracies)
    if len(accuracies) < 2:
        return mean, mean, mean

    ci = stats.t.interval(confidence, len(accuracies) - 1, loc=mean, scale=stats.sem(accuracies))
    return mean, ci[0], ci[1]


def load_human_eval_metrics(system_dir):
    """Load and calculate metrics from human evaluation JSON files."""
    human_eval_patterns = ["human_eval", "huma_eval"]
    metrics = {}

    for pattern in human_eval_patterns:
        for file in system_dir.glob(f"*{pattern}*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                scores = [item["metrics"]["humaneval"]["score"] for item in data]
                if scores:
                    mean, ci_low, ci_high = calculate_confidence_interval(scores)
                    metrics["Human-LLM Correctness"] = {
                        "mean": mean,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                    print(
                        f"Found human eval metrics in {file}: mean={mean:.4f}, CI=[{ci_low:.4f}, {ci_high:.4f}]"
                    )
                    break
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return metrics


def load_metrics(system_dir):
    """Load metrics from a system directory."""
    metrics = {}
    system_name = system_dir.name.split("_")[0].lower()

    # Load human evaluation metrics
    human_metrics = load_human_eval_metrics(system_dir)
    metrics.update(human_metrics)

    # Handle each system specifically to ensure we load from the correct files
    if system_name == "graphiti":
        # For Graphiti, load all metrics from the main metrics file
        main_metrics_file = system_dir / "aggregate_metrics_graphiti.json"

        if main_metrics_file.exists():
            try:
                with open(main_metrics_file) as f:
                    data = json.load(f)

                # Extract all metrics from main file
                standard_metrics = {"correctness": "Correctness", "f1": "F1", "EM": "EM"}

                for key, display_name in standard_metrics.items():
                    metric_key = f"DeepEval {display_name}"
                    if key in data and isinstance(data[key], dict) and "mean" in data[key]:
                        metrics[metric_key] = {
                            "mean": data[key]["mean"],
                            "ci_low": data[key].get("ci_lower", data[key]["mean"]),
                            "ci_high": data[key].get("ci_upper", data[key]["mean"]),
                        }
                        print(
                            f"Found DeepEval metrics in {main_metrics_file}: {key}={data[key]['mean']:.4f}"
                        )
            except Exception as e:
                print(f"Error loading {main_metrics_file}: {e}")

    elif system_name == "mem0":
        # For Mem0, check specific aggregate files
        metrics_file = system_dir / "aggregate_metrics_mem0.json"

        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    data = json.load(f)

                # Extract metrics with proper CI naming
                standard_metrics = {"correctness": "Correctness", "f1": "F1", "EM": "EM"}

                for key, display_name in standard_metrics.items():
                    metric_key = f"DeepEval {display_name}"
                    if key in data and isinstance(data[key], dict) and "mean" in data[key]:
                        metrics[metric_key] = {
                            "mean": data[key]["mean"],
                            "ci_low": data[key].get("ci_lower", data[key]["mean"]),
                            "ci_high": data[key].get("ci_upper", data[key]["mean"]),
                        }
                        print(
                            f"Found DeepEval metrics in {metrics_file}: {key}={data[key]['mean']:.4f}"
                        )
            except Exception as e:
                print(f"Error loading {metrics_file}: {e}")

    elif system_name == "falkor":
        # For Falkor, check specific aggregate files
        metrics_file = system_dir / "aggregate_metrics_falkor_graphrag_sdk.json"

        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    data = json.load(f)

                # Extract metrics with proper CI naming
                standard_metrics = {"correctness": "Correctness", "f1": "F1", "EM": "EM"}

                for key, display_name in standard_metrics.items():
                    metric_key = f"DeepEval {display_name}"
                    if key in data and isinstance(data[key], dict) and "mean" in data[key]:
                        metrics[metric_key] = {
                            "mean": data[key]["mean"],
                            "ci_low": data[key].get("ci_lower", data[key]["mean"]),
                            "ci_high": data[key].get("ci_upper", data[key]["mean"]),
                        }
                        print(
                            f"Found DeepEval metrics in {metrics_file}: {key}={data[key]['mean']:.4f}"
                        )
            except Exception as e:
                print(f"Error loading {metrics_file}: {e}")

    elif system_name == "cognee":
        # For Cognee, check specific aggregate files
        metrics_file = system_dir / "aggregate_metrics_v_deepeval.json"

        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    data = json.load(f)

                # Extract metrics with proper CI naming
                standard_metrics = {"correctness": "Correctness", "f1": "F1", "EM": "EM"}

                for key, display_name in standard_metrics.items():
                    metric_key = f"DeepEval {display_name}"
                    if key in data and isinstance(data[key], dict) and "mean" in data[key]:
                        metrics[metric_key] = {
                            "mean": data[key]["mean"],
                            "ci_low": data[key].get("ci_lower", data[key]["mean"]),
                            "ci_high": data[key].get("ci_upper", data[key]["mean"]),
                        }
                        print(
                            f"Found DeepEval metrics in {metrics_file}: {key}={data[key]['mean']:.4f}"
                        )
            except Exception as e:
                print(f"Error loading {metrics_file}: {e}")
    else:
        # Fallback for any other systems
        deepeval_patterns = [f"aggregate_metrics_{system_name}", "v_deepeval"]
        for pattern in deepeval_patterns:
            for file in system_dir.glob(f"*{pattern}*.json"):
                try:
                    with open(file) as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            # Extract metrics
                            standard_metrics = {
                                "correctness": "Correctness",
                                "f1": "F1",
                                "EM": "EM",
                            }

                            for key, display_name in standard_metrics.items():
                                metric_key = f"DeepEval {display_name}"
                                if (
                                    key in data
                                    and isinstance(data[key], dict)
                                    and "mean" in data[key]
                                ):
                                    metrics[metric_key] = {
                                        "mean": data[key]["mean"],
                                        "ci_low": data[key].get("ci_lower", data[key]["mean"]),
                                        "ci_high": data[key].get("ci_upper", data[key]["mean"]),
                                    }
                                    print(
                                        f"Found DeepEval metrics in {file}: {key}={data[key]['mean']:.4f}"
                                    )
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    # Make sure all standard metrics exist with defaults if missing
    standard_metrics = ["DeepEval Correctness", "DeepEval F1", "DeepEval EM"]
    for metric in standard_metrics:
        if metric not in metrics:
            metrics[metric] = {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            print(f"Added default metrics for missing {metric}")

    return metrics


def plot_metrics(all_systems_metrics):
    """Plot metrics comparison."""
    if not all_systems_metrics:
        print("No metrics found to plot")
        return

    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_theme(style="whitegrid")

    # Cognee brand colors
    brand_colors = {
        "data_dream_violet": "#6510F4",
        "data_flux_green": "#0DFF00",
        "secondary_purple": "#A550FF",
        "abyss_black": "#000000",
        "data_cloud_grey": "#F4F4F4",
        "dark_grey": "#323332",
    }

    # Color palette using Cognee brand colors
    colors = [
        brand_colors["data_flux_green"],
        brand_colors["data_dream_violet"],
        brand_colors["secondary_purple"],
        brand_colors["dark_grey"],
    ]

    # Prepare data with custom ordering (Cognee first, then Graphiti)
    preferred_order = ["Cognee", "Graphiti", "Mem0", "Falkor"]
    systems = [system for system in preferred_order if system in all_systems_metrics]

    # Add any systems not in preferred order at the end
    for system in all_systems_metrics.keys():
        if system not in systems:
            systems.append(system)

    metrics = set()
    for system_metrics in all_systems_metrics.values():
        metrics.update(system_metrics.keys())

    # Sort metrics by average score across systems (highest to lowest)
    def get_metric_avg_score(metric):
        scores = []
        for system in systems:
            if metric in all_systems_metrics[system]:
                scores.append(all_systems_metrics[system][metric]["mean"])
        return np.mean(scores) if scores else 0

    metrics = sorted(list(metrics), key=get_metric_avg_score, reverse=True)

    # Set up the plot with Cognee brand styling
    fig, ax = plt.subplots(figsize=(15, 8), facecolor=brand_colors["data_cloud_grey"])
    ax.set_facecolor(brand_colors["data_cloud_grey"])

    # Plot bars
    x = np.arange(len(systems))
    width = 0.8 / len(metrics)

    for i, metric in enumerate(metrics):
        means = []
        yerr_low = []
        yerr_high = []

        for system in systems:
            if metric in all_systems_metrics[system]:
                m = all_systems_metrics[system][metric]
                means.append(m["mean"])
                yerr_low.append(m["mean"] - m["ci_low"])
                yerr_high.append(m["ci_high"] - m["mean"])
            else:
                means.append(0)
                yerr_low.append(0)
                yerr_high.append(0)

        yerr = [yerr_low, yerr_high]
        ax.bar(
            x + i * width - (len(metrics) - 1) * width / 2,
            means,
            width,
            label=metric,
            color=colors[i % len(colors)],
            alpha=0.85,
            yerr=yerr,
            capsize=4,
            error_kw={
                "elinewidth": 1.5,
                "capthick": 1.5,
                "ecolor": brand_colors["dark_grey"],
                "alpha": 0.5,
            },
        )

    # Customize plot with Cognee styling
    ax.set_ylabel("Score", fontsize=14, fontweight="bold", color=brand_colors["abyss_black"])
    ax.set_title(
        "AI Memory - Benchmark Results",
        fontsize=18,
        pad=20,
        fontweight="bold",
        color=brand_colors["data_dream_violet"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        systems,
        rotation=45,
        ha="right",
        fontsize=12,
        fontweight="bold",
        color=brand_colors["abyss_black"],
    )
    ax.tick_params(axis="y", labelsize=11, colors=brand_colors["abyss_black"])

    # Set y-axis limits with some padding
    ax.set_ylim(0, 1.1)

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color=brand_colors["dark_grey"])
    ax.set_axisbelow(True)

    # Customize legend
    legend = ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Metrics",
        title_fontsize=14,
    )

    # Style the legend text with brand colors
    plt.setp(legend.get_title(), fontweight="bold", color=brand_colors["data_dream_violet"])

    # Add value labels on top of bars with improved visibility
    for i, metric in enumerate(metrics):
        for j, system in enumerate(systems):
            if metric in all_systems_metrics[system]:
                value = all_systems_metrics[system][metric]["mean"]
                if value > 0:  # Only show label if value is greater than 0
                    # Create a small white background for the text to improve legibility
                    ax.text(
                        j + i * width - (len(metrics) - 1) * width / 2,
                        value + 0.02,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                        color=brand_colors["data_dream_violet"],
                        bbox=dict(facecolor="white", alpha=0.7, pad=1, edgecolor="none"),
                    )

    # Add border to the plot
    for spine in ax.spines.values():
        spine.set_edgecolor(brand_colors["dark_grey"])
        spine.set_linewidth(1.5)

    # Adjust layout
    plt.tight_layout()

    # Save plot first without logo
    plt.savefig("metrics_comparison.png", bbox_inches="tight", dpi=300)

    # Now add logo and save again
    try:
        # Try to find the logo file
        logo_path = Path("../assets/cognee-logo-transparent.png")
        if not logo_path.exists():
            logo_path = Path("../assets/cognee_logo.png")

        if logo_path.exists():
            # Create a new figure with the same size
            height, width = fig.get_size_inches()
            fig_with_logo = plt.figure(
                figsize=(height, width), facecolor=brand_colors["data_cloud_grey"]
            )

            # First, plot the saved chart as a background
            chart_img = plt.imread("metrics_comparison.png")
            chart_ax = fig_with_logo.add_subplot(111)
            chart_ax.imshow(chart_img)
            chart_ax.axis("off")

            # Now overlay the logo with transparency
            logo_img = plt.imread(str(logo_path))

            # Position logo in the upper part of the chart with current horizontal position
            # Keep horizontal position (0.65) but move back to upper part of chart
            logo_ax = fig_with_logo.add_axes([0.65, 0.75, 0.085, 0.085], zorder=1)
            logo_ax.imshow(logo_img, alpha=0.45)  # Same opacity
            logo_ax.axis("off")  # Turn off axis

            # Save the combined image
            fig_with_logo.savefig("metrics_comparison_with_logo.png", dpi=300, bbox_inches="tight")
            plt.close(fig_with_logo)

            # Replace the original file with the logo version
            import os

            os.replace("metrics_comparison_with_logo.png", "metrics_comparison.png")

    except Exception as e:
        print(f"Warning: Could not add logo overlay - {e}")

    plt.close(fig)


def main():
    """Main function to process metrics and generate plot."""
    eval_dir = Path(".")
    all_systems_metrics = {}

    # Process each system directory
    for system_dir in eval_dir.glob("*_01032025"):
        print(f"\nChecking system directory: {system_dir}")
        system_name = system_dir.name.split("_")[0].capitalize()
        metrics = load_metrics(system_dir)
        if metrics:
            all_systems_metrics[system_name] = metrics
            print(f"Found metrics for {system_name}: {metrics}")

    print(f"\nAll systems metrics: {all_systems_metrics}")

    if not all_systems_metrics:
        print("No metrics data found!")
        return

    plot_metrics(all_systems_metrics)


if __name__ == "__main__":
    main()
