import statistics
import time
import tracemalloc
from typing import Any, Callable, Dict

import psutil


def benchmark_function(func: Callable, *args, num_runs: int = 5) -> Dict[str, Any]:
    """
    Benchmark a function for memory usage and computational performance.

    Args:
        func: Function to benchmark
        *args: Arguments to pass to the function
        num_runs: Number of times to run the benchmark

    Returns:
        Dictionary containing benchmark metrics
    """
    execution_times = []
    peak_memory_usages = []
    cpu_percentages = []

    process = psutil.Process()

    for _ in range(num_runs):
        # Start memory tracking
        tracemalloc.start()
        initial_memory = process.memory_info().rss

        # Measure execution time and CPU usage
        start_time = time.perf_counter()
        start_cpu_time = process.cpu_times()

        result = func(*args)

        end_cpu_time = process.cpu_times()
        end_time = time.perf_counter()

        # Calculate metrics
        execution_time = end_time - start_time
        cpu_time = (end_cpu_time.user + end_cpu_time.system) - (
            start_cpu_time.user + start_cpu_time.system
        )
        current, peak = tracemalloc.get_traced_memory()
        final_memory = process.memory_info().rss
        memory_used = final_memory - initial_memory

        # Store results
        execution_times.append(execution_time)
        peak_memory_usages.append(peak / 1024 / 1024)  # Convert to MB
        cpu_percentages.append((cpu_time / execution_time) * 100)

        tracemalloc.stop()

    analysis = {
        "mean_execution_time": statistics.mean(execution_times),
        "mean_peak_memory_mb": statistics.mean(peak_memory_usages),
        "mean_cpu_percent": statistics.mean(cpu_percentages),
        "num_runs": num_runs,
    }

    if num_runs > 1:
        analysis["std_execution_time"] = statistics.stdev(execution_times)

    return analysis
