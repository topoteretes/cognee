/**
 * Type scale and spacing rhythm for this page, following the dashboard restyle.
 *
 * Colour convention, applied everywhere on the page:
 *   T.faint  — column headers, eyebrows, the label above a value
 *   T.muted  — secondary numerics and metadata sitting beside primary text
 *   T.text   — primary text and any value a reader is meant to read off
 * Accent colours (green/amber/red) carry score meaning only, never decoration;
 * T.lavender is the brand accent for progress fills and active state.
 */

/** Square corners are the house style — only dots and status pills are round. */
export const RADIUS = 0;

export const SIZE = {
  /** Headline metric. */
  hero: 34,
  /** Page title — 20/300 to match the restyled page headers app-wide. */
  title: 20,
  /** Panel header — matches AsciiFrame's own 14/600 so panels agree app-wide. */
  panel: 14,
  /** Body copy, question text, values. */
  body: 14,
  /** Secondary numerics and metadata. */
  meta: 13,
  /** Column headers and eyebrows. */
  label: 12,
} as const;

export const SPACE = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 28,
} as const;
