"""
Fluent Pipeline builder for complex, reusable pipelines.

Example:
    pipeline = (
        Pipeline("my-pipeline")
        .add_step(extract_people)
        .add_step(enrich, parallel=True)
        .add_step(store_results)
    )

    results = await pipeline.execute(input="text")
    results2 = await pipeline.execute(input="more text")  # Reuse!
"""

import inspect
import warnings
from typing import Any, Callable, Optional

from cognee.pipelines.flow import flow


class Pipeline:
    """Fluent builder for composable, reusable pipelines.

    Provides a builder-pattern API for constructing pipelines with
    explicit step configuration and optional pre-flight validation.

    Example:
        pipeline = (
            Pipeline("analysis")
            .add_step(classify_docs)
            .add_step(extract_entities, batch_size=10)
            .add_step(store_results)
            .validate()
        )

        results = await pipeline.execute(input=documents)
    """

    def __init__(self, name: str):
        self.name = name
        self._steps: list[tuple[Callable, dict[str, Any]]] = []
        self._validated = False

    def add_step(self, fn: Callable, **config) -> "Pipeline":
        """Add a step to the pipeline.

        Args:
            fn: The function to execute as a pipeline step.
            **config: Step configuration (batch_size, parallel, dataset, etc.)

        Returns:
            self, for method chaining.
        """
        self._steps.append((fn, config))
        self._validated = False  # Invalidate on modification
        return self

    def validate(self) -> "Pipeline":
        """Validate pipeline configuration before execution.

        Checks:
        - Each step is callable
        - Type annotations between adjacent steps are compatible (warnings only)

        Returns:
            self, for method chaining.
        """
        validation_warnings = _validate_steps(self._steps)
        for w in validation_warnings:
            warnings.warn(f"Pipeline '{self.name}' validation: {w}", UserWarning, stacklevel=2)
        self._validated = True
        return self

    async def execute(self, input=None, context: dict = None, **kwargs) -> Any:
        """Execute the pipeline.

        Args:
            input: Initial data to feed into the first step.
            context: Optional context dict.
            **kwargs: Additional keyword arguments.

        Returns:
            The output of the last step.
        """
        if not self._validated:
            self.validate()

        step_fns = [fn for fn, _ in self._steps]
        return await flow(*step_fns, input=input, context=context, **kwargs)

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

    # Type-compatibility checking between adjacent steps
    for i in range(len(steps) - 1):
        current_fn, _ = steps[i]
        next_fn, _ = steps[i + 1]

        try:
            current_sig = inspect.signature(current_fn)
            next_sig = inspect.signature(next_fn)

            current_return = current_sig.return_annotation
            if current_return is inspect.Parameter.empty:
                continue

            # Find first positional parameter of next step
            next_params = list(next_sig.parameters.values())
            if not next_params:
                continue

            next_input = next_params[0]
            if next_input.annotation is inspect.Parameter.empty:
                continue

            # Simple check: if both are annotated, warn on obvious mismatches
            # This is intentionally lenient — warn, don't error
            if (
                current_return is not inspect.Parameter.empty
                and next_input.annotation is not inspect.Parameter.empty
                and _is_obvious_mismatch(current_return, next_input.annotation)
            ):
                validation_warnings.append(
                    f"Step {i + 1} '{current_fn.__name__}' returns {current_return} "
                    f"but step {i + 2} '{next_fn.__name__}' expects {next_input.annotation}"
                )
        except (ValueError, TypeError):
            continue

    return validation_warnings


def _is_obvious_mismatch(return_type, input_type) -> bool:
    """Check for obvious type mismatches between steps.

    This is intentionally conservative — only flags clear problems.
    Returns False when uncertain (to avoid false positives).
    """
    # Both are concrete types (not generics or Annotated)
    if isinstance(return_type, type) and isinstance(input_type, type):
        # str -> int is obviously wrong, str -> str is fine, list -> list is fine
        if return_type is str and input_type is int:
            return True
        if return_type is int and input_type is str:
            return True
    return False
