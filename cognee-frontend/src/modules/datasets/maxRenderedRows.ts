/**
 * Upper bound on file rows rendered at once.
 *
 * The file tables are plain `.map()` over their input with no virtualization,
 * so the row count is DOM built synchronously on the main thread. A dataset
 * with 171,828 documents froze the tab. Capping keeps the render bounded no
 * matter how many rows a caller hands over — including a caller that asked the
 * API for a large page, or an older server that still answered unbounded.
 *
 * One constant, so raising it (or replacing all of this with real windowing) is
 * a single edit rather than a hunt through the table components.
 */
export const MAX_RENDERED_ROWS = 500;

/** Rows to render, plus how many were held back. */
export function capRows<T>(rows: T[]): { visible: T[]; hidden: number } {
  return rows.length <= MAX_RENDERED_ROWS
    ? { visible: rows, hidden: 0 }
    : { visible: rows.slice(0, MAX_RENDERED_ROWS), hidden: rows.length - MAX_RENDERED_ROWS };
}
