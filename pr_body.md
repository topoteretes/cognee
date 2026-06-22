## Summary

Implements frequency weight tracking for the Kuzu graph adapter and exposes it via a new session API, closing #1993.

### Changes

**GraphDBInterface**: Added 4 abstract methods:
- `get_node_frequency_weights(node_ids)` / `get_edge_frequency_weights(edge_ids)`
- `set_node_frequency_weights()` / `set_edge_frequency_weights()`

**KuzuAdapter**: Implemented the 4 methods following the existing feedback weights pattern:
- Frequency weights default to `1.0` (vs feedback weights' `0.5`)
- Each increment adds `1.0` to the stored value

**Session API** (new in this update):
- `add_frequency_weights(session_id, qa_id, node_ids, edge_ids, user)` - stores graph element IDs used in generating a QA entry and resets `frequency_weights_applied` flag
- `update_qa()` method in session manager to update QA entry metadata

**New files**:
- `cognee/tasks/memify/frequency_weights_constants.py` — metadata key constant
- `cognee/tasks/memify/apply_frequency_weights.py` — task that increments frequency by 1.0 per usage
- `cognee/tasks/memify/apply_frequency_weights_pipeline.py` — pipeline wrapper

### Usage

The `apply_frequency_weights_pipeline` reads QA entries from sessions and increments frequency weights on referenced graph nodes/edges each time they are used in retrieval.

Clients can call `cognee.api.v1.session.add_frequency_weights()` to record which graph elements were used for a given answer.

### Test results

```
38 passed, 8 warnings (all pre-existing deprecation warnings)
```

Closes #1993
