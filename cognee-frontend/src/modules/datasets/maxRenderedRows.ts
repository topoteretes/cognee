/**
 * Upper bound on file rows rendered at once.
 *
 * The file tables are plain `.map()` over their input with no virtualization,
 * so the row count is DOM built synchronously on the main thread. A dataset
 * with 171,828 documents froze the tab.
 *
 * This is also the ceiling on infinite scroll: appending indefinitely would
 * walk the page straight back into that freeze, just more slowly. 2,000 rows is
 * roughly a dozen scroll steps — far enough to browse without hitting a wall,
 * and around 12k DOM nodes, which stays responsive. Reaching six figures needs
 * windowing or server-side search, not a bigger number here.
 *
 * One constant, so both the loader and the tables agree, and raising it (or
 * replacing all of this with real windowing) is a single edit.
 */
export const MAX_RENDERED_ROWS = 2000;

/** Rows to render, plus how many were held back. */
export function capRows<T>(rows: T[]): { visible: T[]; hidden: number } {
  return rows.length <= MAX_RENDERED_ROWS
    ? { visible: rows, hidden: 0 }
    : { visible: rows.slice(0, MAX_RENDERED_ROWS), hidden: rows.length - MAX_RENDERED_ROWS };
}
