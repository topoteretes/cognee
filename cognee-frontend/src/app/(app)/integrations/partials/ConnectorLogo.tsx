"use client";

import type { ReactElement } from "react";
import classNames from "classnames";

interface ConnectorLogoProps {
  /** Glyph filename (no extension) under /visuals/logos/datasources. */
  logo?: string;
  /** Fallback shown when the connector has no glyph yet. */
  initials: string;
  /** Brand tile color. */
  color: string;
  size: number;
}

// The glyph is painted as a CSS mask rather than an <img> so one monochrome
// source file renders white on any brand tile at any size, the same treatment as
// the "More data sources" grid, so a live connector and a false-door one don't
// read as two different kinds of thing.
export default function ConnectorLogo({ logo, initials, color, size }: ConnectorLogoProps): ReactElement {
  const glyphSize = Math.round(size * 0.48);
  const maskUrl = `url(/visuals/logos/datasources/${logo}.svg)`;

  return (
    <div
      className="flex shrink-0 items-center justify-center rounded-[10px] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.08)]"
      style={{ width: size, height: size, background: color }}
    >
      {logo ? (
        <span
          aria-hidden
          className="block bg-white"
          style={{
            width: glyphSize,
            height: glyphSize,
            maskImage: maskUrl,
            WebkitMaskImage: maskUrl,
            maskRepeat: "no-repeat",
            WebkitMaskRepeat: "no-repeat",
            maskPosition: "center",
            WebkitMaskPosition: "center",
            maskSize: "contain",
            WebkitMaskSize: "contain",
          }}
        />
      ) : (
        <span className={classNames("font-bold tracking-[-0.02em] text-white", size >= 36 ? "text-[13px]" : "text-[11px]")}>
          {initials}
        </span>
      )}
    </div>
  );
}
