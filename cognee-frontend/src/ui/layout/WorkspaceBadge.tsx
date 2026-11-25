"use client";

import styles from "./WorkspaceBadge.module.css";

export type WorkspaceBadgeState = "active" | "asleep" | "waking" | "blocked";

const BLOCKED_FILL = "#555";
const FOREGROUND = "#EDECEA";

// The palette (FilterContext's TENANT_COLORS) is picked for a white initial on
// a solid fill, so several entries — the deep violet, blue and magenta — are
// far too dark to read as a hairline or a letter against the near-black chrome.
// Mixing toward the foreground keeps the workspace recognisable by its own
// colour in the hollow states without failing contrast on half the palette.
const HOLLOW_MIX_PERCENT = 55;

const STATE_LABEL: Record<WorkspaceBadgeState, string> = {
  active: "running",
  asleep: "asleep",
  waking: "waking up",
  blocked: "no active subscription",
};

function hollowColor(color: string): string {
  return `color-mix(in srgb, ${color} ${HOLLOW_MIX_PERCENT}%, ${FOREGROUND})`;
}

/**
 * The square initial that identifies a workspace, in the top bar trigger and in
 * every picker row.
 *
 * Solid when the workspace is up, hollow (dashed outline, no fill) when it is
 * asleep or being woken. The dashed outline says "not running" at a glance,
 * which the caption alone cannot do once the picker is closed — and closing the
 * picker is exactly what a user does right after clicking a sleeping workspace.
 */
export default function WorkspaceBadge({
  initial,
  color,
  size,
  state,
}: {
  initial: string;
  color: string;
  size: number;
  state: WorkspaceBadgeState;
}): React.ReactElement {
  const hollow = state === "asleep" || state === "waking";
  const outline = hollowColor(color);

  return (
    <div
      className={state === "waking" ? styles.waking : undefined}
      // The badge is the only place the state is shown once the picker closes,
      // and a dashed border says nothing to a screen reader.
      role="img"
      aria-label={`${initial} workspace, ${STATE_LABEL[state]}`}
      style={{
        width: size,
        height: size,
        borderRadius: size >= 20 ? 4 : 3,
        background: hollow ? "transparent" : state === "blocked" ? BLOCKED_FILL : color,
        // Longhands, not the `border` shorthand: a browser that cannot parse
        // color-mix drops the whole declaration it appears in, and with the
        // shorthand that took the dashed outline with it and left an invisible
        // square. Split like this only the colour is lost, and the letter falls
        // back to the wrapper's own colour instead of the UA default.
        borderWidth: hollow ? 1.5 : 0,
        borderStyle: hollow ? "dashed" : "none",
        borderColor: hollow ? outline : "transparent",
        color: FOREGROUND,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
      }}
    >
      <span
        style={{
          color: hollow ? outline : "#fff",
          fontSize: Math.round(size / 2),
          fontWeight: 700,
          lineHeight: 1,
        }}
      >
        {initial}
      </span>
    </div>
  );
}
