"use client";

import type { ReactElement } from "react";
import classNames from "classnames";
import { connectorGlyphs } from "./connectorGlyphs";

interface ConnectorLogoProps {
  /** Glyph filename (no extension) under /visuals/logos/datasources. */
  logo?: string;
  /** Fallback shown when the connector has no glyph yet. */
  initials: string;
  /** Brand tile color. */
  color: string;
  size: number;
}

// Render vendored vector glyphs inline, with no network image, mask, or filter.
export default function ConnectorLogo({ logo, initials, color, size }: ConnectorLogoProps): ReactElement {
  const glyph = logo ? connectorGlyphs[logo] : undefined;
  const glyphSize = Math.round(size * 0.48);

  return (
    <div
      className="flex shrink-0 items-center justify-center rounded-[10px] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.08)]"
      style={{ width: size, height: size, background: color }}
    >
      {glyph ? (
        <svg aria-hidden="true" viewBox={glyph.viewBox} width={glyphSize} height={glyphSize}
          fill="currentColor" style={{ display: "block", color: "white", flexShrink: 0 }}>
          {glyph.shape}
        </svg>
      ) : logo ? (
        // A normal image avoids CSS-mask repaint loss after OAuth popup focus changes.
        // eslint-disable-next-line @next/next/no-img-element
        <img src={`/visuals/logos/datasources/${logo}.svg`} alt="" aria-hidden
          width={glyphSize} height={glyphSize}
          style={{ display: "block", objectFit: "contain",  }} />
      ) : (
        <span className={classNames("font-bold tracking-[-0.02em] text-white", size >= 36 ? "text-[13px]" : "text-[11px]")}>
          {initials}
        </span>
      )}
    </div>
  );
}
