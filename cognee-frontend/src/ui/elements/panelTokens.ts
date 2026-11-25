import type { CSSProperties } from "react";

export const TEXT = "#EDECEA";
export const TEXT_MUTED = "rgba(237,236,234,0.55)";
export const TEXT_FAINT = "rgba(237,236,234,0.35)";
export const ACCENT = "#6510F4";
export const BORDER = "rgba(255,255,255,0.1)";
export const BORDER_SOFT = "rgba(255,255,255,0.08)";
export const ROW_BORDER = "rgba(255,255,255,0.07)";
export const DANGER = "#EF4444";

/** TWKLausanne ships only 300 and 700 — any other weight is synthesised by the
 *  browser and snaps to one of these, so only these two are ever used. */
export const WEIGHT_LIGHT = 300;
export const WEIGHT_BOLD = 700;
export const FONT_SANS = "var(--font-sans)";
export const FONT_MONO = "var(--font-mono)";

/** Page title, matching the Sessions and Datasets headers. */
export const pageTitleStyle: CSSProperties = {
  fontSize: 20,
  fontWeight: WEIGHT_LIGHT,
  fontFamily: FONT_SANS,
  color: TEXT,
  margin: 0,
};

/** Large display numbers — the Sessions stat-card treatment. */
export const displayNumberStyle: CSSProperties = {
  fontWeight: WEIGHT_LIGHT,
  fontFamily: FONT_SANS,
  color: TEXT,
  fontVariantNumeric: "tabular-nums",
};

/** Small uppercase label above a value or a group of rows. */
export const eyebrowStyle: CSSProperties = {
  fontSize: 10,
  fontWeight: WEIGHT_BOLD,
  letterSpacing: "0.06em",
  textTransform: "uppercase",
};

/** The single surface the whole page lives on — tabs, toolbar and table all
 *  sit inside it so the tab strip reads as attached to its content. */
export const cardStyle: CSSProperties = {
  background: "rgba(255,255,255,0.06)",
  backdropFilter: "blur(12px)",
  border: `1px solid ${BORDER}`,
  borderRadius: 12,
  overflow: "hidden",
};

/** Title band: the live count on the left, the primary action on the right. */
export const bandStyle: CSSProperties = {
  padding: "20px 20px 16px",
};

/** Column headings above a table body. */
export const columnsStyle: CSSProperties = {
  height: 38,
  padding: "0 20px",
  borderTop: `1px solid ${BORDER_SOFT}`,
  borderBottom: `1px solid ${BORDER_SOFT}`,
  background: "rgba(255,255,255,0.03)",
  fontSize: 12,
  fontWeight: WEIGHT_BOLD,
  color: TEXT_MUTED,
};

/** Header for a section nested inside the page card (group detail). */
export const sectionHeaderStyle: CSSProperties = {
  height: 46,
  padding: "0 20px",
  borderTop: `1px solid ${BORDER_SOFT}`,
  borderBottom: `1px solid ${BORDER_SOFT}`,
  background: "rgba(255,255,255,0.03)",
};

export const rowStyle: CSSProperties = {
  height: 56,
  padding: "0 20px",
  borderBottom: `1px solid ${ROW_BORDER}`,
};

export const inputStyle: CSSProperties = {
  width: "100%",
  height: 36,
  padding: "0 12px",
  fontSize: 14,
  color: TEXT,
  background: "rgba(255,255,255,0.06)",
  border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 6,
  outline: "none",
};

export const selectStyle: CSSProperties = {
  height: 32,
  padding: "0 26px 0 10px",
  fontSize: 13,
  color: TEXT_MUTED,
  background: "#1a1a1a",
  border: `1px solid ${BORDER_SOFT}`,
  borderRadius: 6,
  outline: "none",
  appearance: "none",
  cursor: "pointer",
};

/** Dark pill, the Twingate primary action. */
export const primaryButtonStyle: CSSProperties = {
  height: 34,
  padding: "0 16px",
  background: ACCENT,
  color: "#fff",
  border: "none",
  borderRadius: 999,
  fontSize: 13,
  fontWeight: 500,
  cursor: "pointer",
  flexShrink: 0,
};

export const ghostButtonStyle: CSSProperties = {
  height: 30,
  padding: "0 12px",
  background: "rgba(255,255,255,0.06)",
  border: `1px solid ${BORDER}`,
  borderRadius: 6,
  fontSize: 12,
  color: TEXT_MUTED,
  cursor: "pointer",
  flexShrink: 0,
};

export const emptyStyle: CSSProperties = {
  padding: "36px 20px",
  textAlign: "center",
  color: TEXT_FAINT,
  fontSize: 13,
};

export const menuDropdownStyle = {
  background: "#1a1a1a",
  border: `1px solid ${BORDER_SOFT}`,
  backdropFilter: "blur(16px)",
  borderRadius: 8,
  boxShadow: "0 8px 24px rgba(0,0,0,0.4)",
  "--menu-item-hover": "rgba(255,255,255,0.08)",
} as const;

export const menuItemStyle = { fontSize: 13, color: TEXT } as const;
export const menuItemDangerStyle = { fontSize: 13, color: DANGER } as const;
