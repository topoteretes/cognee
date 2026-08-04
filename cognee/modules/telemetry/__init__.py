"""Telemetry sinks and the local event model.

The model is deliberately NOT re-exported here: importing it from two paths gives
the declarative class two chances to register the same table on ``Base.metadata``.
Import it from ``cognee.modules.telemetry.models`` instead.
"""
