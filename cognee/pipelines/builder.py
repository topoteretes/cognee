"""
Fluent Pipeline builder for complex, reusable pipelines.

Example:
    pipeline = (
        Pipeline("my-pipeline")
        .add_step(extract_people)
        .add_step(enrich, batch_size=10, graph_model=KnowledgeGraph)
        .add_step(store_results)
    )

    results = await pipeline.execute(input="text")
    results2 = await pipeline.execute(input="more text")  # Reuse!

    # Parallel per-item execution:
    results = await pipeline.execute(input=[doc1, doc2], parallel=True)
"""

import inspect
import warnings
from typing import Any, Callable

from cognee.pipelines.step import step as step_decorator


class Pipeline:
    """Fluent builder for composable, reusable pipelines.

    Example:
        pipeline = (
            Pipeline("analysis")
            .add_step(classify_docs)
            .add_step(extract_entities, batch_size=10, graph_model=KnowledgeGraph)
            .add_step(store_results, enriches=True)
        )

        results = await pipeline.execute(input=documents)
    """

    def __init__(self, name: str):
        self.name = name
        self._steps: list[tuple[Callable, dict[str, Any]]] = []

    def add_step(self, fn: Callable, **config) -> "Pipeline":
        """Add a step to the pipeline.

        Args:
            fn: The function to execute as a pipeline step.
            **config: Step configuration. Reserved keys:
                - batch_size: Number of items per batch (default 1)
                - cache: Enable caching (default False)
                - enriches: Step modifies data in place (default False)
              All other keys are stored as default params and injected
              into the function by parameter name.

        Returns:
            self, for method chaining.
        """
        self._steps.append((fn, config))
        return self

    def validate(self) -> list[str]:
        """Validate pipeline configuration.

        Returns:
            List of warning messages (empty if valid).
        """
        validation_warnings = _validate_steps(self._steps)
        for w in validation_warnings:
            warnings.warn(f"Pipeline '{self.name}': {w}", UserWarning, stacklevel=2)
        return validation_warnings

    async def execute(self, input=None, context: dict = None, parallel: bool = False, **kwargs):
        """Execute the pipeline.

        Args:
            input: Initial data to feed into the first step.
            context: Optional context dict.
            parallel: If True and input is a list, each item flows through
                      the full chain concurrently.

        Returns:
            The output of the last step.
        """
        from cognee.pipelines.flow import run_steps

        # Apply config from add_step() to functions that aren't already decorated
        decorated_steps = []
        for fn, config in self._steps:
            if config and not hasattr(fn, "_cognee_step_config"):
                fn = step_decorator(fn, **config)
            decorated_steps.append(fn)

        return await run_steps(
            *decorated_steps, input=input, context=context, parallel=parallel, **kwargs
        )

    @property
    def steps(self) -> list[str]:
        """Get list of step function names."""
        return [fn.__name__ for fn, _ in self._steps]

    def __repr__(self) -> str:
        step_names = " -> ".join(self.steps) if self._steps else "(empty)"
        return f"Pipeline({self.name!r}: {step_names})"


def _validate_steps(steps: list[tuple[Callable, dict]]) -> list[str]:
    """Validate step compatibility and return a list of warning messages."""
    validation_warnings = []

    for i, (fn, _config) in enumerate(steps):
        if not callable(fn):
            validation_warnings.append(f"Step {i + 1} is not callable: {fn!r}")

    for i in range(len(steps) - 1):
        current_fn, _ = steps[i]
        next_fn, _ = steps[i + 1]

        try:
            current_sig = inspect.signature(current_fn)
            next_sig = inspect.signature(next_fn)

            current_return = current_sig.return_annotation
            if current_return is inspect.Parameter.empty:
                continue

            next_params = list(next_sig.parameters.values())
            if not next_params:
                continue

            next_input = next_params[0]
            if next_input.annotation is inspect.Parameter.empty:
                continue

            if _is_obvious_mismatch(current_return, next_input.annotation):
                validation_warnings.append(
                    f"Step {i + 1} '{current_fn.__name__}' returns {current_return} "
                    f"but step {i + 2} '{next_fn.__name__}' expects {next_input.annotation}"
                )
        except (ValueError, TypeError):
            continue

    return validation_warnings


def _is_obvious_mismatch(return_type, input_type) -> bool:
    """Check for obvious type mismatches between steps.

    Flags cases where both are concrete types and neither is a subclass of the other.
    Returns False for generics, Annotated, or Any (to avoid false positives).
    """
    if not isinstance(return_type, type) or not isinstance(input_type, type):
        return False
    if issubclass(return_type, input_type) or issubclass(input_type, return_type):
        return False
    return True
