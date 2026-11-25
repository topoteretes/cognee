/**
 * A plain integer with thousands separators, pinned to en-US.
 *
 * Bare `toLocaleString()` reads the ICU default, which Node resolves at startup
 * and no test can pin, so an Arabic or Persian locale renders Eastern
 * Arabic-Indic digits beside a Latin-digit axis and every assertion on the
 * figure fails. `jest.config.ts` states the rule and the pages were breaking it.
 */
export function fmtCount(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}
