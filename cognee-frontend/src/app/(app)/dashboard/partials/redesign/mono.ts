/**
 * Design tokens for the Overview redesign.
 *
 * The surface is the cognee brand style — sans typography, soft dark cards,
 * purple/lavender accents (see cognee.ai). `FONT` is the sans default every
 * panel spreads; `MONO_FONT` is opt-in for dense numerics. The module keeps its
 * historical name so the many `./mono` imports stay stable.
 */

export const SANS = '"TWKLausanne", ui-sans-serif, system-ui, -apple-system, sans-serif';
export const MONO = 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Roboto Mono", monospace';

export const FONT = { fontFamily: SANS } as const;
/** Opt-in monospace — used only where digits must not shift (charts, code). */
export const MONO_FONT = { fontFamily: MONO } as const;

export const T = {
  // ── Surfaces ──
  panel:        "#000000",   // card body
  chrome:       "rgba(255,255,255,0.06)",   // raised surface (memory core, chips)
  chromeAlt:    "rgba(255,255,255,0.04)",   // deepest inset
  frame:        "rgba(237,236,234,0.10)",   // hairline border
  frameStrong:  "rgba(237,236,234,0.18)",

  // ── Text ──
  text:         "#EDECEA",
  muted:        "rgba(237,236,234,0.58)",
  faint:        "rgba(237,236,234,0.36)",
  ghost:        "rgba(237,236,234,0.20)",

  // ── Accents ──
  green:        "#5CE39A",   // positive / live
  lavender:     "#BC9BFF",   // brand accent on dark (links, active)
  purple:       "#BC9BFF",   // brand purple = cognee lavender
  purpleSoft:   "rgba(188,155,255,0.14)",
  amber:        "#E6B24D",   // warning / reconnect
  red:          "#F0808A",   // error
  blue:         "#89B4FA",   // secondary
} as const;

/** Small uppercase brand eyebrow (cognee's section-label idiom). */
export const LABEL = {
  ...FONT,
  fontSize: 11,
  fontWeight: 600,
  letterSpacing: "0.14em",
  textTransform: "uppercase" as const,
  color: T.lavender,
};
