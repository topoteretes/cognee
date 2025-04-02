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
    
    system_name = system_dir.name.split("_")[0].lower()

    for pattern in human_eval_patterns:
        for file in system_dir.glob(f"*{pattern}*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                
                # Handle different JSON structures based on the system
                if system_name == "falkor":
                    # Falkor has metrics under 'metrics.correctness.score'
                    scores = [item["metrics"]["correctness"]["score"] for item in data]
                else:
                    # Other systems have metrics under 'metrics.humaneval.score'
                    scores = [item["metrics"]["humaneval"]["score"] for item in data]
                
                if scores:
                    mean, ci_low, ci_high = calculate_confidence_interval(scores)
                    metrics["Human-LLM Correctness"] = {
                        "mean": mean,
                        "ci_low": ci_low,
                        "ci_high": ci_high
                    }
                    print(f"Found human eval metrics in {file}: mean={mean:.4f}, CI=[{ci_low:.4f}, {ci_high:.4f}]")
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


def plot_metrics(all_metrics, output_file='metrics_comparison.png'):
    """Plot metrics comparison across systems."""
    # Define a color palette for a professional, consistent look
    colors = {
        "Human-LLM Correctness": "#2C7BB6",  # Blue
        "DeepEval Correctness": "#D7191C",   # Red
        "DeepEval F1": "#FDAE61",           # Orange
        "DeepEval EM": "#ABD9E9"            # Light blue
    }
    
    # Sort metrics by their average score across systems
    def get_metric_avg_score(metric_name):
        scores = []
        for system_metrics in all_metrics.values():
            if metric_name in system_metrics:
                scores.append(system_metrics[metric_name]["mean"])
        return sum(scores) / len(scores) if scores else 0
    
    # Sort system names by their Human-LLM Correctness scores in descending order
    systems = []
    for system_name, metrics in all_metrics.items():
        if "Human-LLM Correctness" in metrics:
            systems.append((system_name, metrics["Human-LLM Correctness"]["mean"]))
        else:
            systems.append((system_name, 0))
    systems.sort(key=lambda x: x[1], reverse=True)
    system_names = [system[0] for system in systems]
    
    # Get all unique metric types and sort them by average score
    all_metric_types = set()
    for metrics in all_metrics.values():
        all_metric_types.update(metrics.keys())
    all_metric_types = sorted(all_metric_types, key=get_metric_avg_score, reverse=True)
    
    # Set up the figure
    plt.figure(figsize=(15, 8))
    
    # Set width of bars and positions
    bar_width = 0.2
    num_metrics = len(all_metric_types)
    num_systems = len(all_metrics)
    
    # Create positions for groups of bars
    indices = np.arange(num_metrics)
    
    # Plot each system's metrics
    for i, system_name in enumerate(system_names):
        system_metrics = all_metrics[system_name]
        
        # Plot bars for each metric
        for j, metric_type in enumerate(all_metric_types):
            if metric_type in system_metrics:
                metric = system_metrics[metric_type]
                mean = metric["mean"]
                ci_low = metric.get("ci_low", mean)
                ci_high = metric.get("ci_high", mean)
                error = [[mean - ci_low], [ci_high - mean]]
                
                pos = indices[j] + (i - num_systems/2 + 0.5) * bar_width
                bar = plt.bar(pos, mean, bar_width, alpha=0.8, label=f"{system_name} {metric_type}" if j == 0 else "", 
                        color=colors.get(metric_type, f"C{j}"))
                
                # Add error bars
                plt.errorbar(pos, mean, yerr=error, fmt='none', ecolor='black', capsize=5, capthick=2, elinewidth=2)
                
                # Add value labels on top of bars
                plt.text(pos, mean + 0.03, f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Add labels, title and legend
    plt.xlabel('Metric Type', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Metrics Comparison Across Systems', fontsize=16, fontweight='bold')
    
    # Set x-ticks at the center of each group of bars
    plt.xticks(indices, all_metric_types, fontsize=10, fontweight='bold', rotation=0)
    
    # Set y-axis limit for better proportion
    plt.ylim(0, 1.1)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend with custom position and style
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=num_systems, 
                frameon=True, fontsize=10, title_fontsize=12)
    plt.setp(legend.get_title(), fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def main():
    # Get all system directories with pattern "*_01032025"
    base_dir = Path(".")
    system_dirs = [d for d in base_dir.glob("*_01032025") if d.is_dir()]
    
    if not system_dirs:
        print("No system directories found with pattern *_01032025")
        return
    
    # Load metrics for all systems
    all_systems_metrics = {}
    for system_dir in system_dirs:
        system_name = system_dir.name.split("_")[0].capitalize()
        print(f"\nChecking system directory: {system_dir}")
        
        # Load metrics from DeepEval JSON files
        metrics = load_metrics(system_dir)
        
        # Add human eval metrics if available
        human_eval_metrics = load_human_eval_metrics(system_dir)
        if human_eval_metrics:
            metrics.update(human_eval_metrics)
        
        if metrics:
            all_systems_metrics[system_name] = metrics
            print(f"Found metrics for {system_name}: {metrics}")
    
    # Print summary of all metrics
    print(f"\nAll systems metrics: {all_systems_metrics}")
    
    # Plot metrics if any were found
    if all_systems_metrics:
        plot_metrics(all_systems_metrics)
    else:
        print("No metrics found for plotting")


if __name__ == "__main__":
    main()
