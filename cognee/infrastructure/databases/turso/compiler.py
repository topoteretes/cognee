"""SQL compiler for the Turso dialect: flattens nested joins.

The Turso engine rejects a parenthesized join inside a FROM clause::

    FROM a JOIN (b JOIN c ON b.id = c.id) ON c.id = a.c_id
    -- Parse error: Parenthesized FROM clause subqueries are not supported

SQLAlchemy 2.0 always renders that shape for a join whose right side is itself a
join, which cognee's ORM produces constantly: ``User``, ``Tenant`` and ``Role`` are
joined-table subclasses of ``Principal`` (their selectable *is* ``principals JOIN
tenants``), so every relationship load that targets them nests a join. The
``supports_right_nested_joins`` rewrite of SQLAlchemy 1.3 no longer exists.

This compiler renders such a tree as a left-deep chain instead::

    FROM a JOIN b ON 1 = 1 JOIN c ON b.id = c.id AND c.id = a.c_id

Every table of the tree is joined in left-to-right order, and each ON clause
receives the predicates whose tables are all available at that point. For inner
joins this is exact (a conjunction over the same cross product). A table that
arrives through an OUTER join keeps the outer join type, so
``a LEFT OUTER JOIN (b JOIN c ON bc) ON ab`` becomes
``a LEFT OUTER JOIN b ON ab LEFT OUTER JOIN c ON bc``, which is equivalent whenever
the inner join is total (every ``b`` has its ``c``), the case for joined-table
inheritance and for association tables under foreign-key integrity. If a ``b`` row
has no ``c`` (broken integrity), the original yields NULLs for both and the flattened
form yields ``b`` with NULL ``c`` columns; the ORM drops such rows because the
entity's primary key is NULL, so loaded collections are unaffected.

Only the shapes verified against cognee's ORM are rewritten; anything else raises
``CompileError`` instead of silently changing results:

* a nested group may contain inner joins only (the outer edge, if any, is the
  top-level join that introduces the group);
* every OUTER step must receive at least one predicate — an unconstrained
  ``LEFT OUTER JOIN ... ON 1 = 1`` would multiply rows.

FULL OUTER joins are left to the stock renderer (SQLite has none).
"""

from __future__ import annotations

import itertools
from typing import Any

from sqlalchemy.dialects.sqlite.base import SQLiteCompiler
from sqlalchemy.exc import CompileError
from sqlalchemy.sql import operators
from sqlalchemy.sql.elements import BooleanClauseList, ClauseElement
from sqlalchemy.sql.selectable import FromGrouping, Join


def _unwrap(node):
    while isinstance(node, FromGrouping):
        node = node.element
    return node


def _predicates(onclause: ClauseElement | None) -> list[ClauseElement]:
    """Split ``a AND b AND c`` into its conjuncts so each can join at its own step."""
    if onclause is None:
        return []
    if isinstance(onclause, BooleanClauseList) and onclause.operator is operators.and_:
        return [conjunct for clause in onclause.clauses for conjunct in _predicates(clause)]
    return [onclause]


class CogneeTursoCompiler(SQLiteCompiler):
    """SQLite compiler that never emits ``JOIN (x JOIN y ...)``."""

    def visit_join(self, join, asfrom=False, from_linter=None, **kwargs):
        if join.full or not isinstance(_unwrap(join.right), Join):
            return super().visit_join(join, asfrom=asfrom, from_linter=from_linter, **kwargs)

        steps: list[tuple[Any, bool]] = []  # (from object, arrives through an outer join)
        predicates: list[ClauseElement] = []

        def walk(node, outer: bool, nested: bool) -> None:
            node = _unwrap(node)
            if isinstance(node, Join) and not node.full:
                if nested and node.isouter:
                    raise CompileError(
                        "Turso: cannot flatten a nested OUTER join; the engine rejects "
                        "parenthesized joins and only nested INNER joins are rewritten."
                    )
                walk(node.left, outer, nested)
                walk(node.right, outer or node.isouter, nested)
                predicates.extend(_predicates(node.onclause))
            else:
                steps.append((node, outer))

        walk(join.left, False, nested=False)
        walk(join.right, join.isouter, nested=True)
        predicates.extend(_predicates(join.onclause))

        available: set[int] = set()
        remaining = list(predicates)

        def ready(clause: ClauseElement, extra: Any = None) -> bool:
            froms = clause._from_objects
            # A predicate with no FROM objects of its own (literal, correlated
            # column) can go anywhere; place it as soon as it is seen.
            return all(
                id(_unwrap(item)) in available or (extra is not None and _unwrap(item) is extra)
                for item in froms
            )

        def pick_next(pending: list[tuple[Any, bool]]) -> tuple[Any, bool]:
            # Prefer a table that some pending predicate can constrain right away,
            # so an OUTER step never degenerates into an unconstrained ``ON 1 = 1``
            # that multiplies rows. Only tables of the same join type as the next
            # one in tree order may be pulled forward: crossing an inner/outer
            # boundary would change the result.
            first_outer = pending[0][1]
            for candidate in pending:
                if candidate[1] != first_outer:
                    break
                if any(
                    ready(clause, extra=candidate[0])
                    and any(_unwrap(item) is candidate[0] for item in clause._from_objects)
                    for clause in remaining
                ):
                    return candidate
            return pending[0]

        parts: list[str] = []
        pending = list(steps)
        previous = None
        while pending:
            step = pending[0] if previous is None else pick_next(pending)
            pending.remove(step)
            from_object, outer = step
            available.add(id(from_object))
            rendered = from_object._compiler_dispatch(
                self, asfrom=True, from_linter=from_linter, **kwargs
            )
            if previous is None:
                parts.append(rendered)
                previous = from_object
                continue
            on_clauses = [clause for clause in remaining if ready(clause)]
            remaining = [clause for clause in remaining if clause not in on_clauses]
            if on_clauses:
                on_sql = " AND ".join(
                    clause._compiler_dispatch(self, from_linter=from_linter, **kwargs)
                    for clause in on_clauses
                )
            elif outer:
                raise CompileError(
                    "Turso: cannot flatten this join; an OUTER step would have no ON "
                    "predicate and would multiply rows."
                )
            else:
                on_sql = "1 = 1"  # exact for inner joins: the predicates land later
            if from_linter:
                from_linter.edges.update(
                    itertools.product(previous._from_objects, from_object._from_objects)
                )
            join_type = " LEFT OUTER JOIN " if outer else " JOIN "
            parts.append(f"{join_type}{rendered} ON {on_sql}")
            previous = from_object

        if remaining:  # predicates over tables outside the tree: append to the last step
            tail = " AND ".join(
                clause._compiler_dispatch(self, from_linter=from_linter, **kwargs)
                for clause in remaining
            )
            parts[-1] = f"{parts[-1]} AND {tail}"
        return "".join(parts)
