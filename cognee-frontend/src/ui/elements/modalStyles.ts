import type { CSSProperties } from "react";
import { DANGER, TEXT, TEXT_FAINT, TEXT_MUTED, WEIGHT_BOLD, primaryButtonStyle } from "@/ui/elements/panelTokens";

export const modalTitleStyle: CSSProperties = { fontSize: 18, fontWeight: WEIGHT_BOLD, color: TEXT, margin: 0 };

export const modalHintStyle: CSSProperties = { fontSize: 13, color: TEXT_MUTED, margin: "4px 0 0" };

export const fieldErrorStyle: CSSProperties = { fontSize: 12, color: DANGER, margin: "4px 0 0" };

/** The full-width confirm button at the bottom of a modal; greyed out until
 *  there is something to confirm. */
export function confirmButtonStyle(enabled: boolean): CSSProperties {
  return {
    ...primaryButtonStyle,
    height: 40,
    width: "100%",
    borderRadius: 8,
    background: enabled ? primaryButtonStyle.background : "rgba(255,255,255,0.08)",
    color: enabled ? "#fff" : TEXT_FAINT,
    cursor: enabled ? "pointer" : "default",
  };
}
