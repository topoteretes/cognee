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
joins this is exact (a conjunction over the same cross product).

A table that arrives through an OUTER join keeps the outer join type, so
``a LEFT OUTER JOIN (b JOIN c ON bc) ON ab`` becomes
``a LEFT OUTER JOIN b ON ab LEFT OUTER JOIN c ON bc``. That is equivalent only when
the step to ``c`` is *total*: every ``b`` row that matched has its ``c`` row, so the
second step can never leave ``b`` matched with ``c`` NULL where the original would
have dropped both. The compiler proves totality from the schema: every table after
the first in an outer-joined group must be reached by an equality
``b.col = c.col`` where ``b.col`` carries a foreign key to ``c.col``, and must
receive no other predicate. Joined-table inheritance satisfies this in one direction
only: ``users.id`` references ``principals.id``, so ``users`` then ``principals`` is
total, while ``principals`` then ``users`` is not (a tenant's principal has no
``users`` row). The step order follows the predicates (see ``pick_next``), so an
outer ON that names the subclass table puts it first. If the data breaks a declared
foreign key, the flattened form yields ``b`` with NULL ``c`` columns where the
original yields NULLs for both.

Only shapes the compiler can prove equivalent are rewritten; anything else raises
``CompileError`` instead of silently changing results:

* a nested group may contain inner joins only (the outer edge, if any, is the
  join that introduces the group);
* every OUTER step must receive at least one predicate — an unconstrained
  ``LEFT OUTER JOIN ... ON 1 = 1`` would multiply rows;
* every OUTER step after the first of its group must be total, as above. A filter
  (``u.active = 1``) or a predicate naming a table outside the group landing on
  such a step would turn "no match" into "half a match".

FULL OUTER joins are left to the stock renderer (SQLite has none).
"""

from __future__ import annotations

import itertools
from typing import Any

from sqlalchemy.dialects.sqlite.base import SQLiteCompiler
from sqlalchemy.exc import CompileError, NoReferenceError
from sqlalchemy.sql import operators
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList, ClauseElement
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


def _column_references(column, target) -> bool:
    """True when ``column`` carries a foreign key to ``target`` (aliases resolved)."""
    for foreign_key in getattr(column, "foreign_keys", ()):
        try:
            referenced = foreign_key.column
        except NoReferenceError:  # FK target not in this MetaData: no proof of totality
            continue
        if referenced in getattr(target, "proxy_set", ()):
            return True
    return False


def _total_step_predicate(clause: ClauseElement, step, earlier: set[int]) -> bool:
    """True for ``earlier.col = step.col`` where ``earlier.col`` references ``step.col``.

    Such an equality cannot fail for a matched earlier row while the foreign key
    holds, so the outer step to ``step`` adds columns without changing which rows
    match. ``earlier`` holds the ids of the group's tables already placed.
    """
    if not isinstance(clause, BinaryExpression) or clause.operator is not operators.eq:
        return False
    sides = (clause.left, clause.right)
    for source, target in (sides, sides[::-1]):
        source_tables = [_unwrap(item) for item in source._from_objects]
        target_tables = [_unwrap(item) for item in target._from_objects]
        if (
            len(source_tables) == 1
            and len(target_tables) == 1
            and id(source_tables[0]) in earlier
            and target_tables[0] is step
            and _column_references(source, target)
        ):
            return True
    return False


class CogneeTursoCompiler(SQLiteCompiler):
    """SQLite compiler that never emits ``JOIN (x JOIN y ...)``."""

    def visit_join(self, join, asfrom=False, from_linter=None, **kwargs):
        if join.full or not isinstance(_unwrap(join.right), Join):
            return super().visit_join(join, asfrom=asfrom, from_linter=from_linter, **kwargs)

        # (from object, arrives through an outer join, nested group id or None)
        steps: list[tuple[Any, bool, int | None]] = []
        predicates: list[ClauseElement] = []
        group_ids = itertools.count()

        def walk(node, outer: bool, group: int | None) -> None:
            node = _unwrap(node)
            if isinstance(node, Join) and not node.full:
                if group is not None and node.isouter:
                    raise CompileError(
                        "Turso: cannot flatten a nested OUTER join; the engine rejects "
                        "parenthesized joins and only nested INNER joins are rewritten."
                    )
                walk(node.left, outer, group)
                # A join on the right of a join is a parenthesized group.
                right_group = group
                if right_group is None and isinstance(_unwrap(node.right), Join):
                    right_group = next(group_ids)
                walk(node.right, outer or node.isouter, right_group)
                predicates.extend(_predicates(node.onclause))
            else:
                steps.append((node, outer, group))

        walk(join.left, False, None)
        walk(join.right, join.isouter, next(group_ids))
        predicates.extend(_predicates(join.onclause))

        # Tables of each outer-joined group already placed (ids): the first one
        # placed may take any predicate, every later one must be a total step.
        placed_in_group: dict[int, set[int]] = {}

        def check_outer_group_step(from_object, group, on_clauses) -> None:
            earlier = placed_in_group.setdefault(group, set())
            if earlier and not all(
                _total_step_predicate(clause, from_object, earlier) for clause in on_clauses
            ):
                raise CompileError(
                    "Turso: cannot flatten this join; a later step of an OUTER-joined "
                    "group is not reached through a foreign-key equality alone, so the "
                    "flattened chain could return a partial match where the nested "
                    "join returns none."
                )
            earlier.add(id(from_object))

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

        def pick_next(pending: list[tuple[Any, bool, int | None]]) -> tuple[Any, bool, int | None]:
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
        last_step_checked = False
        while pending:
            step = pending[0] if previous is None else pick_next(pending)
            pending.remove(step)
            from_object, outer, group = step
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
            last_step_checked = outer and group is not None
            if last_step_checked:  # on_clauses is non-empty here
                check_outer_group_step(from_object, group, on_clauses)
            if from_linter:
                from_linter.edges.update(
                    itertools.product(previous._from_objects, from_object._from_objects)
                )
            join_type = " LEFT OUTER JOIN " if outer else " JOIN "
            parts.append(f"{join_type}{rendered} ON {on_sql}")
            previous = from_object

        if remaining:  # predicates over tables outside the tree: append to the last step
            if last_step_checked and len(placed_in_group.get(group, ())) > 1:
                raise CompileError(
                    "Turso: cannot flatten this join; a predicate over a table outside "
                    "the join would land on a later step of an OUTER-joined group."
                )
            tail = " AND ".join(
                clause._compiler_dispatch(self, from_linter=from_linter, **kwargs)
                for clause in remaining
            )
            parts[-1] = f"{parts[-1]} AND {tail}"
        return "".join(parts)
