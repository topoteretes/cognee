import argparse
import asyncio

from .benchmark_function import benchmark_function

from cognee.modules.graph.utils import get_graph_from_model
from cognee.tests.unit.interfaces.graph.util import (
    PERSON_NAMES,
    create_organization_recursive,
)

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark graph model with configurable recursive depth"
    )
    parser.add_argument(
        "--recursive-depth",
        type=int,
        default=3,
        help="Recursive depth for graph generation (default: 3)",
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of benchmark runs (default: 5)"
    )
    args = parser.parse_args()

    society = create_organization_recursive(
        "society", "Society", PERSON_NAMES, args.recursive_depth
    )
    added_nodes = {}
    added_edges = {}
    visited_properties = {}
    nodes, edges = asyncio.run(get_graph_from_model(
        society,
        added_nodes = added_nodes,
        added_edges = added_edges,
        visited_properties = visited_properties,
    ))

    def get_graph_from_model_sync(model):
        added_nodes = {}
        added_edges = {}
        visited_properties = {}

        return asyncio.run(get_graph_from_model(
            model,
            added_nodes = added_nodes,
            added_edges = added_edges,
            visited_properties = visited_properties,
        ))

    results = benchmark_function(get_graph_from_model_sync, society, num_runs=args.runs)
    print("\nBenchmark Results:")
    print(
        f"N nodes: {len(nodes)}, N edges: {len(edges)}, Recursion depth: {args.recursive_depth}"
    )
    print(f"Mean Peak Memory: {results['mean_peak_memory_mb']:.2f} MB")
    print(f"Mean CPU Usage: {results['mean_cpu_percent']:.2f}%")
    print(f"Mean Execution Time: {results['mean_execution_time']:.4f} seconds")

    if "std_execution_time" in results:
        print(f"Execution Time Std: {results['std_execution_time']:.4f} seconds")
