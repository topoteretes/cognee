"use client";

import { useState, type ReactElement } from "react";
import Image from "next/image";

// The terminal block and status-line callout the onboarding flow introduced.
// They live here rather than under onboarding/ because the dashboard's connect
// panel and the integrations wizard render the identical steps — the whole
// point of CLO-532's alignment pass is that all three read the same.

const MONO = 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace';

function CopyIcon(): ReactElement {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

/**
 * Terminal block with a line-number gutter and a single copy control for the
 * whole block — the user runs all of it, so there is nothing to copy piecemeal.
 */
export function TerminalBlock({ lines, loading, placeholder = "Preparing…", copyLabel = "Copy", onCopied }: {
  lines: string[];
  loading?: boolean;
  placeholder?: string;
  copyLabel?: string;
  /** Fired after a successful copy — callers use it to track the interaction. */
  onCopied?: () => void;
}): ReactElement {
  const [copied, setCopied] = useState(false);
  const shown = loading ? [placeholder] : lines;

  function copy(): void {
    if (loading) return;
    // writeText rejects in an insecure context or when permission is denied.
    // Flipping to "Copied" regardless would claim a copy that did not happen,
    // and onCopied is documented above as firing on success.
    navigator.clipboard.writeText(lines.join("\n")).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      onCopied?.();
    }).catch((err) => console.warn("Copy failed", err));
  }

  return (
    <div style={{ border: "1px solid rgba(255,255,255,0.08)", background: "#131316", width: "100%", boxSizing: "border-box" }}>
      <div style={{ padding: "10px 0", overflowX: "auto" }}>
        {shown.map((line, i) => (
          <div key={i} style={{ display: "flex", gap: 12, alignItems: "baseline", padding: "1px 12px 1px 0" }}>
            <span style={{ width: 28, textAlign: "right", flexShrink: 0, fontFamily: MONO, fontSize: 12, color: "rgba(237,236,234,0.25)", userSelect: "none" }}>
              {loading ? "" : i + 1}
            </span>
            <pre style={{ margin: 0, fontFamily: MONO, fontSize: 12.5, lineHeight: "21px", color: loading ? "#585B70" : "rgba(237,236,234,0.9)", whiteSpace: "pre" }}>
              <code>{line}</code>
            </pre>
          </div>
        ))}
      </div>
      {!loading && (
        <button
          onClick={(e) => { e.stopPropagation(); copy(); }}
          className="cursor-pointer"
          style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 7, width: "100%", background: "rgba(255,255,255,0.05)", border: "none", borderTop: "1px solid rgba(255,255,255,0.08)", padding: "9px 0", fontSize: 12.5, fontWeight: 500, color: copied ? "#22C55E" : "rgba(237,236,234,0.8)", fontFamily: "inherit" }}
        >
          <CopyIcon />
          {copied ? "Copied" : copyLabel}
        </button>
      )}
    </div>
  );
}

// Where the status line sits in claude-code-statusline.png, in the image's own
// pixel space. The overlay SVG shares that viewBox, so the callout scales with
// the image instead of drifting at other widths.
const SHOT_W = 920;
const SHOT_H = 723;
const BAR = { x: 64, y: 609, w: 250, h: 23 };
// The region the loupe magnifies: the FIRST status line only. The second line
// ("manual mode on …") is noise here and dilutes the one string the user is
// being asked to look for.
const LENS = { x: 58, y: 606, w: 268, h: 27 };

// Panel geometry, all in the image's own pixel space so the SVG connectors can
// be derived from the same numbers and never drift from the panel at other card
// widths. Left-aligned with the source rect, and parked just above it so the
// two read as one object rather than a pasted-on strip.
const LOUPE_W_PX = SHOT_W * 0.7;
const LOUPE_H_PX = LOUPE_W_PX * (LENS.h / LENS.w);
const LOUPE_LEFT_PX = BAR.x;
const LOUPE_BOTTOM_PX = BAR.y - 42;
const LOUPE_TOP_PX = LOUPE_BOTTOM_PX - LOUPE_H_PX;

// The SVG shares the image's viewBox, which is ~2.5x the rendered CSS size, so
// hairlines have to be scaled up to land on ~1 CSS px.
const HAIRLINE = 2.5;
const ACCENT = "#BC9BFF";

// The magnified crop. Re-renders the same asset oversized inside a clipped
// window rather than shipping a second image, so the two can never drift.
// Percentages work because the wrapper's aspect ratio is the crop's, which
// makes the horizontal and vertical scale factors identical.
function StatusLineLoupe(): ReactElement {
  return (
    <div
      style={{
        position: "absolute",
        left: `${(LOUPE_LEFT_PX / SHOT_W) * 100}%`,
        width: `${(LOUPE_W_PX / SHOT_W) * 100}%`,
        top: `${(LOUPE_TOP_PX / SHOT_H) * 100}%`,
        aspectRatio: `${LENS.w} / ${LENS.h}`,
        overflow: "hidden",
        borderRadius: 6,
        border: `1px solid rgba(188,155,255,0.6)`,
        boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
        boxSizing: "border-box",
      }}
      aria-hidden="true"
    >
      <Image
        src="/visuals/claude-code-statusline.png"
        alt=""
        width={SHOT_W}
        height={SHOT_H}
        style={{
          position: "absolute",
          width: `${(SHOT_W / LENS.w) * 100}%`,
          height: `${(SHOT_H / LENS.h) * 100}%`,
          left: `${-(LENS.x / LENS.w) * 100}%`,
          top: `${-(LENS.y / LENS.h) * 100}%`,
          maxWidth: "none",
        }}
      />
    </div>
  );
}

/**
 * A Claude Code window with the Cognee status line called out, plus the loupe
 * overlaid so it is actually readable — at this width the real thing renders
 * about four pixels tall.
 */
export function StatusLineScreenshot(): ReactElement {
  return (
    <div style={{ position: "relative", width: "100%", lineHeight: 0 }}>
      <Image
        src="/visuals/claude-code-statusline.png"
        alt="A Claude Code window with the Cognee status line in the bottom-left corner"
        width={SHOT_W}
        height={SHOT_H}
        style={{ width: "100%", height: "auto", display: "block" }}
      />
      <svg
        viewBox={`0 0 ${SHOT_W} ${SHOT_H}`}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
        aria-hidden="true"
      >
        {/* Source rect: a hairline, not an outline — it marks the area without
            competing with the panel above it. */}
        <rect
          x={BAR.x} y={BAR.y} width={BAR.w} height={BAR.h} rx="3"
          fill="none" stroke={ACCENT} strokeOpacity="0.55" strokeWidth={HAIRLINE}
        />
        {/* The two connectors: source rect's upper corners to the panel's lower
            corners. The trapezoid they form is what makes the panel read as an
            enlargement rather than a floating strip. They run to the panel's
            exact bottom edge, and the panel sits later in the DOM, so both ends
            finish behind it. */}
        <line
          x1={BAR.x} y1={BAR.y} x2={LOUPE_LEFT_PX} y2={LOUPE_BOTTOM_PX}
          stroke={ACCENT} strokeOpacity="0.32" strokeWidth={HAIRLINE}
        />
        <line
          x1={BAR.x + BAR.w} y1={BAR.y} x2={LOUPE_LEFT_PX + LOUPE_W_PX} y2={LOUPE_BOTTOM_PX}
          stroke={ACCENT} strokeOpacity="0.32" strokeWidth={HAIRLINE}
        />
      </svg>
      <StatusLineLoupe />
    </div>
  );
}
