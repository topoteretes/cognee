"use client";

import React, { useId, useState } from "react";
import { T } from "./mono";

/** Hovered point, reported in the SVG's own 0-100% coordinate space so a
 *  parent can position labels without knowing the chart's pixel size. */
export interface SparklineHoverPoint {
  index: number;
  xPct: number;
  /** Height of the `values` point — where to hang a label on that line. */
  yPct: number;
  /** Height of the `compare` point, when a compare series is drawn. */
  comparePct?: number;
  /** Pointer height, for a label that should follow the cursor rather than a line. */
  cursorYPct: number;
}

interface SparklineProps {
  /** Y values, oldest → newest. A flat/empty series renders a baseline. */
  values: number[];
  width?: number;
  height?: number;
  color?: string;
  /** Draw the soft area fill under the line. */
  fill?: boolean;
  /** Fires with the nearest point while hovering, and null on mouse-leave. */
  onHover?: (point: SparklineHoverPoint | null) => void;
  /**
   * Second series drawn as a reference line — must be index-aligned with
   * `values`. Both series share one y-scale, so the vertical gap between them is
   * readable as the real delta (that gap is the point of the comparison).
   */
  compare?: {
    values: number[];
    color: string;
    /** Fills the gap between the two lines — the delta is what the comparison is
     *  for, and at a large ratio the lower line alone is too flat to read it. */
    band?: string;
  };
  /**
   * Gridlines at the major ticks: verticals at these point indices and
   * horizontals at these heights (0 = plot floor, 1 = plot ceiling). Both are
   * expected to match the labelled ticks on the axes outside this component.
   */
  grid?: { xIndices: number[]; yFractions: number[]; color: string };
  /**
   * Top of the y-scale. Pass the axis maximum whenever labelled ticks sit beside
   * the plot: scaling to the data maximum instead puts the top gridline above
   * the highest point, and every label a little off its line.
   */
  yMax?: number;
}

/**
 * Minimal area sparkline for the Cost panel — replaces the ASCII `.-'` spend
 * curve in WO-0 with a real SVG plot while keeping the terminal restraint (one
 * hairline stroke, one faint fill). Scales to the container via a viewBox so it
 * stays crisp at any width.
 */
export function Sparkline({
  values,
  width = 320,
  height = 96,
  color = "#A6E3A1",
  fill = true,
  onHover,
  compare,
  grid,
  yMax,
}: SparklineProps): React.ReactElement {
  const gid = useId();
  const pad = 4;
  const w = width;
  const h = height;

  const compareValues = compare?.values ?? [];
  const scaled = [...values, ...compareValues];
  const max = yMax ?? Math.max(...scaled, 0.0001);
  const min = Math.min(...scaled, 0);
  const span = max - min || 1;

  const n = values.length;
  const project = (v: number, i: number, count: number): readonly [number, number] => {
    const x = count <= 1 ? pad : pad + (i / (count - 1)) * (w - pad * 2);
    const y = h - pad - ((v - min) / span) * (h - pad * 2);
    return [x, y] as const;
  };
  const pts = values.map((v, i) => project(v, i, n));
  const comparePts = compareValues.map((v, i) => project(v, i, compareValues.length));
  const compareLine = comparePts.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  // Closed region between the two lines: along the compare line, then back along
  // the value line in reverse.
  const bandPath = comparePts.length > 1 && pts.length === comparePts.length
    ? `${compareLine} ${[...pts].reverse().map(([x, y]) => `L${x.toFixed(1)},${y.toFixed(1)}`).join(" ")} Z`
    : "";

  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const line = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area = pts.length
    ? `${line} L${pts[pts.length - 1][0].toFixed(1)},${h - pad} L${pts[0][0].toFixed(1)},${h - pad} Z`
    : "";

  function handleMouseMove(e: React.MouseEvent<SVGSVGElement>): void {
    if (pts.length < 2) return;
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width === 0) return;
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const idx = Math.round(frac * (n - 1));
    setHoverIdx(idx);
    const [x, y] = pts[idx];
    const comparePoint = comparePts[idx];
    const cursorFrac = rect.height === 0 ? 0 : (e.clientY - rect.top) / rect.height;
    onHover?.({
      index: idx,
      xPct: (x / w) * 100,
      yPct: (y / h) * 100,
      comparePct: comparePoint ? (comparePoint[1] / h) * 100 : undefined,
      cursorYPct: Math.min(100, Math.max(0, cursorFrac * 100)),
    });
  }

  function handleMouseLeave(): void {
    setHoverIdx(null);
    onHover?.(null);
  }

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      style={{ display: "block", height }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {fill && pts.length > 1 && (
        <>
          <defs>
            <linearGradient id={`spark-${gid}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.22} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <path d={area} fill={`url(#spark-${gid})`} />
        </>
      )}
      {bandPath && compare?.band && <path d={bandPath} fill={compare.band} opacity={0.1} />}
      {/* Grid paints over the fills — under them the band swallowed every line
          that fell inside it — but stays under the series strokes. */}
      {grid && (
        <g stroke={grid.color} strokeWidth={1} shapeRendering="crispEdges">
          {grid.yFractions.map((f) => {
            const y = h - pad - f * (h - pad * 2);
            return <line key={`gy-${f}`} x1={pad} y1={y} x2={w - pad} y2={y} />;
          })}
          {grid.xIndices.map((i) => {
            const x = n <= 1 ? pad : pad + (i / (n - 1)) * (w - pad * 2);
            return <line key={`gx-${i}`} x1={x} y1={pad} x2={x} y2={h - pad} />;
          })}
        </g>
      )}
      {comparePts.length > 1 && (
        <path
          d={compareLine}
          fill="none"
          stroke={compare?.color}
          strokeWidth={1.5}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      )}
      {pts.length > 1 ? (
        <path d={line} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
      ) : (
        <line x1={pad} y1={h / 2} x2={w - pad} y2={h / 2} stroke={color} strokeWidth={1.5} strokeDasharray="3 3" opacity={0.5} />
      )}
      {hoverIdx !== null && pts[hoverIdx] && (
        <>
          <line
            x1={pts[hoverIdx][0]}
            y1={pad}
            x2={pts[hoverIdx][0]}
            y2={h - pad}
            stroke={color}
            strokeWidth={1}
            strokeDasharray="2 2"
            opacity={0.45}
          />
          <circle cx={pts[hoverIdx][0]} cy={pts[hoverIdx][1]} r={3} fill={color} stroke={T.chromeAlt} strokeWidth={1.5} />
          {/* The compare line gets a marker too, so its hovered value reads as
              belonging to that line rather than floating in the plot. */}
          {comparePts[hoverIdx] && (
            <circle
              cx={comparePts[hoverIdx][0]}
              cy={comparePts[hoverIdx][1]}
              r={3}
              fill={compare?.color}
              stroke={T.chromeAlt}
              strokeWidth={1.5}
            />
          )}
        </>
      )}
    </svg>
  );
}
