export const tokens = {
  // ── Brand accent ──
  purple:           "#6510F4",
  purpleHover:      "#5A0ED6",
  purplePressed:    "#4A0BAF",
  purpleLight:      "#6C5CE7",   // selected icon tint
  green:            "#0DFF00",

  // ── Status ──
  statusProcessing: "#FFD500",
  statusSuccess:    "#53FF24",
  statusError:      "#FF5024",

  // ── Surfaces ──
  bgPage:           "#F4F4F4",
  bgWhite:          "#FFFFFF",
  bgHover:          "#F4F4F5",   // zinc-100
  bgPressed:        "#E4E4E7",   // zinc-200
  bgSelected:       "#F0EDFF",   // light violet
  bgSelectedHover:  "#E8E2FD",
  bgDisabled:       "#FAFAFA",   // zinc-50

  // ── Borders ──
  border:           "#E4E4E7",   // zinc-200
  borderLight:      "#E5E7EB",   // gray-200
  borderSubtle:     "#F4F4F5",   // zinc-100 (within cards)
  borderFocus:      "#6510F4",

  // ── Text (Zinc scale) ──
  textDark:         "#18181B",   // zinc-900 — page titles
  textBody:         "#3F3F46",   // zinc-700 — body text
  textSecondary:    "#52525B",   // zinc-600 — sidebar text, labels
  textMuted:        "#71717A",   // zinc-500 — subtitles, icons
  textPlaceholder:  "#A1A1AA",   // zinc-400 — placeholders, state labels
  textDisabled:     "#D4D4D8",   // zinc-300

  // ── Hover / pressed text shifts ──
  textHover:        "#3F3F46",   // zinc-700
  textPressed:      "#27272A",   // zinc-800

  // ── Shadows ──
  shadowDropdown:   "0px 8px 30px #00000014",
  shadowDropdownHeavy: "0px 8px 30px #0000001F, 0px 0px 0px 1px #0000000F",
  shadowFocusRing:  "0px 0px 0px 3px #6510F41A",

  // ── Terminal / misc ──
  terminalDot:      "#351A4B",
  graphText:        "#333333",
} as const;
