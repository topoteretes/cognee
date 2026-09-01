"use client";

import React, { useMemo, useState } from "react";
import {
  estimateCostUsd,
  estimateNoCogneeCostUsd,
  noCogneeCostFromSpendUsd,
  tokensAvoided,
  NO_COGNEE_TOKEN_MULTIPLIER,
  type SessionRow,
} from "@/modules/sessions/getSessions";
import type { TenantHourlyCosts } from "@/modules/billing/getTenantHourlyCosts";
import type { PipelineRun } from "@/ui/elements/AgentActivityTerminal";
import { AsciiFrame } from "./AsciiFrame";
import { Sparkline, type SparklineHoverPoint } from "./Sparkline";
import { FONT, T } from "./mono";
import { RangeToggle, type DashRange } from "./RangeToggle";

interface CostPanelProps {
  sessions: SessionRow[];
  /** The activity feed (pipeline runs + memory operations) — the only source
   *  for the real per-operation token counts behind the headline. */
  runs: PipelineRun[];
  balanceUsd: number | null;
  range: DashRange;
  onRangeChange: (range: DashRange) => void;
  hourlyCosts?: TenantHourlyCosts | null;
  onViewBreakdown?: () => void;
}

const DAYS_FOR_RANGE: Record<DashRange, number> = { "24h": 1, "7d": 7, "30d": 30 };
/** Left-hand width (px) of the y-axis label column and its x-axis spacer, kept in sync so ticks stay aligned under the chart. */
const AXIS_LABEL_WIDTH = 40;
/** Steps a reader recognises: 25 / 50 / 100 / 250 / 500 rather than whatever
 *  the data maximum divides into. Scaled by powers of ten as the axis grows. */
const STEP_MULTIPLES = [1, 2, 2.5, 5];
/** Ceiling on gridlines — the step is the smallest nice one that fits inside it. */
const MAX_Y_INTERVALS = 5;
/** Below this the axis keeps cents; above it whole dollars, since four ticks of
 *  cents on a three-figure axis is noise rather than precision. */
const AXIS_CENTS_BELOW_USD = 10;

interface Slot { key: string; label: string; value: number }

/** Both plotted lines plus their totals: what cognee cost, and what the same
 *  work would have cost with no cognee in the loop. */
interface CostSeries {
  series: number[];
  baseline: number[];
  pointLabels: string[];
  total: number;
  baselineTotal: number;
}

function fmtUsd(n: number): string {
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/** Whole dollars — the headline saving is an estimate, so cents would imply a
 *  precision it doesn't have. Per-point figures keep their cents on hover. */
function fmtUsdWhole(n: number): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

/** Axis tick label — precision follows the scale of the axis, not the tick. */
function fmtAxisUsd(value: number, axisMax: number): string {
  return axisMax < AXIS_CENTS_BELOW_USD ? fmtUsd(value) : fmtUsdWhole(value);
}

const TOKEN_UNITS: Array<{ limit: number; suffix: string }> = [
  { limit: 1_000_000_000, suffix: "B" },
  { limit: 1_000_000, suffix: "M" },
  { limit: 1_000, suffix: "K" },
];

/** Token counts at dashboard scale — abbreviated to one decimal (1.2B, 150.6M). */
export function fmtTokens(n: number): string {
  for (const { limit, suffix } of TOKEN_UNITS) {
    if (n >= limit) return `${(n / limit).toFixed(1)}${suffix}`;
  }
  return Math.round(n).toLocaleString();
}

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

function sum(values: number[]): number {
  return values.reduce((a, b) => a + b, 0);
}

/** Build the empty time slots for the range: hourly for 24h, else daily (capped). */
function buildSlots(range: DashRange, now: Date): Slot[] {
  const slots: Slot[] = [];
  if (range === "24h") {
    for (let i = 23; i >= 0; i--) {
      const d = new Date(now);
      d.setHours(now.getHours() - i, 0, 0, 0);
      slots.push({ key: `${d.getMonth()}-${d.getDate()}-${d.getHours()}`, label: `${pad2(d.getHours())}:00`, value: 0 });
    }
    return slots;
  }
  const days = DAYS_FOR_RANGE[range];
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(now.getDate() - i);
    slots.push({ key: `${d.getMonth()}-${d.getDate()}`, label: `${d.getMonth() + 1}/${d.getDate()}`, value: 0 });
  }
  return slots;
}

function slotKey(range: DashRange, d: Date): string {
  return range === "24h" ? `${d.getMonth()}-${d.getDate()}-${d.getHours()}` : `${d.getMonth()}-${d.getDate()}`;
}

export function summarizeHourlyCosts(range: DashRange, costs: TenantHourlyCosts, now: Date): CostSeries {
  const slots = buildSlots(range, now);
  const index = new Map(slots.map((s, i) => [s.key, i]));

  for (const point of costs.points) {
    const idx = index.get(slotKey(range, new Date(point.hour)));
    if (idx !== undefined) slots[idx].value += Number(point.spendUsd) || 0;
  }

  const series = slots.map((s) => s.value);
  const baseline = series.map(noCogneeCostFromSpendUsd);
  // Totals are summed from the plotted slots, not from every point the endpoint
  // returned: its window and this slot grid don't line up exactly at the edges,
  // and a headline that counts spend the chart doesn't show can't be checked
  // against the chart. Matters more since the headline is the saving, which
  // amplifies any stray point by the baseline multiple.
  const total = sum(series);
  return {
    series,
    baseline,
    pointLabels: slots.map((s) => s.label),
    total,
    baselineTotal: sum(baseline),
  };
}

/**
 * Real tokens routed over the plotted range, summed straight off the activity
 * feed (pipeline runs + memory operations) rather than inverted out of a
 * dollar figure — LiteLLM bills embedding calls this endpoint never counts,
 * so tokens and dollars are related but genuinely different numbers, and this
 * total must not pretend otherwise.
 *
 * tokens_in/tokens_out are independently nullable: null means "not measured",
 * 0 means "measured zero". A row contributes only what it actually measured
 * (a null component adds nothing, never a fabricated zero), and the total
 * itself is null only when nothing in range was measured at all — so a
 * genuine zero-usage range can be told apart from "we have no data yet",
 * mirroring how `balanceUsd === null` renders "—" below instead of "$0".
 */
export function sumMeasuredTokens(range: DashRange, runs: PipelineRun[], now: Date): number | null {
  const inRange = new Set(buildSlots(range, now).map((s) => s.key));
  let total = 0;
  let measured = false;
  for (const r of runs) {
    const at = r.started_at ?? r.created_at;
    if (!at) continue;
    if (!inRange.has(slotKey(range, new Date(at)))) continue;
    if (r.tokens_in === null && r.tokens_out === null) continue;
    measured = true;
    total += (r.tokens_in ?? 0) + (r.tokens_out ?? 0);
  }
  return measured ? total : null;
}

/** A y-axis rounded out to readable steps: the plot scales to `max`, not to the
 *  data maximum, so the labels sit exactly on the gridlines. */
export interface YAxis { max: number; step: number; ticks: number[] }

function buildYAxis(step: number, dataMax: number): YAxis {
  const intervals = Math.max(1, Math.ceil(dataMax / step));
  return {
    max: intervals * step,
    step,
    // Multiplied out rather than accumulated, so 0.1 steps don't drift.
    ticks: Array.from({ length: intervals + 1 }, (_, i) => i * step),
  };
}

/**
 * Rounds an axis up to the smallest "nice" step (1/2/2.5/5 × a power of ten)
 * that covers the data in at most MAX_Y_INTERVALS gridlines. Scaling the plot
 * to the data maximum instead gives ticks like $146 — arithmetically fine, but
 * nothing a reader can hold onto or compare between ranges.
 */
export function niceYAxis(dataMax: number): YAxis {
  if (!(dataMax > 0)) return { max: 1, step: 1, ticks: [0, 1] };
  // Start an order of magnitude below the ideal step and walk up, so the first
  // candidate that fits is also the tightest one.
  let magnitude = 10 ** (Math.floor(Math.log10(dataMax)) - 1);
  for (let decade = 0; decade < 40; decade++) {
    for (const multiple of STEP_MULTIPLES) {
      const step = multiple * magnitude;
      if (step > 0 && Math.ceil(dataMax / step) <= MAX_Y_INTERVALS) return buildYAxis(step, dataMax);
    }
    magnitude *= 10;
  }
  return buildYAxis(dataMax, dataMax);
}

interface Tick { index: number; label: string }

/**
 * Evenly-spaced x-axis ticks that always include the first and last point —
 * sampling on a fixed stride (e.g. every 6th of 24) can skip the true last
 * point, so a flex-spaced row would show a stale hour as if it were "now".
 * Each tick carries its real index so it can be positioned at its actual
 * proportional x, not just evenly spaced across the row.
 */
export function pickTicks(pointLabels: string[], maxTicks = 4): Tick[] {
  const n = pointLabels.length;
  if (n <= maxTicks) return pointLabels.map((label, index) => ({ index, label }));
  const indices = new Set<number>();
  for (let k = 0; k < maxTicks; k++) {
    indices.add(Math.round((k / (maxTicks - 1)) * (n - 1)));
  }
  return [...indices].sort((a, b) => a - b).map((index) => ({ index, label: pointLabels[index] }));
}

/** Inside these margins (share of plot width) a label would overflow the plot,
 *  so it flips to the crosshair's other side. */
const LABEL_FLIP_AT_PCT = 72;

interface PointLabelProps {
  xPct: number;
  yPct: number;
  text: string;
  color: string;
  /** Which side of the crosshair to sit on. The two line values take one side
   *  and the cursor value the other, so they can never stack on each other —
   *  the cursor sits between the lines, right where a shared side collides. */
  side: "left" | "right";
  /** The cursor label — bordered, since it floats free of any line. */
  emphasis?: boolean;
}

/** A hover value pinned beside its point on the chart, on an opaque plate so it
 *  stays readable over the lines, the gridlines and the savings band. */
function PointLabel({ xPct, yPct, text, color, side, emphasis = false }: PointLabelProps): React.ReactElement {
  const onLeft = side === "left" ? xPct > 100 - LABEL_FLIP_AT_PCT : xPct > LABEL_FLIP_AT_PCT;
  return (
    <span
      style={{
        ...FONT,
        position: "absolute",
        left: `${xPct}%`,
        top: `${yPct}%`,
        transform: `translate(${onLeft ? "calc(-100% - 9px)" : "9px"}, -50%)`,
        background: T.panel,
        border: emphasis ? `1px solid ${color}` : "none",
        padding: emphasis ? "2px 6px" : "1px 4px",
        fontSize: 11,
        fontWeight: 600,
        color,
        fontVariantNumeric: "tabular-nums",
        whiteSpace: "nowrap",
        pointerEvents: "none",
        zIndex: 2,
      }}
    >
      {text}
    </span>
  );
}

/** Chart key: a stroke swatch matching how its series is drawn, plus its label. */
function LegendKey({ color, label }: { color: string; label: string }): React.ReactElement {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <svg width={16} height={2} aria-hidden style={{ flexShrink: 0 }}>
        <line x1={0} y1={1} x2={16} y2={1} stroke={color} strokeWidth={2} />
      </svg>
      {label}
    </span>
  );
}

/**
 * SPEND panel — leads with what cognee saved over the selected range, not with
 * spend: the chart plots real spend against what the same work would have cost
 * with no cognee in the loop, and the gap between those two lines is the whole
 * story the panel is telling. Both lines share one y-scale so that gap is a
 * true delta; per-point spend stays readable on hover.
 *
 * The headline is that delta in TOKENS, with the dollar equivalent demoted to a
 * sub-line: tokens are the only saving that holds on every plan. Usage-based
 * billing converts them to cash, but a subscription refunds nothing — there the
 * saving is limit headroom left unspent, so a dollar headline would overclaim.
 *
 * The live account balance (not range-scoped) sits in the header badge rather
 * than the headline, so the range-scoped chart underneath can't read as a
 * balance trend. The no-cognee baseline is an estimate (see the helpers in
 * getSessions) — flagged as such in the UI, not fabricated as exact.
 */
export function CostPanel({ sessions, runs, balanceUsd, range, onRangeChange, hourlyCosts, onViewBreakdown }: CostPanelProps): React.ReactElement {
  const [hover, setHover] = useState<SparklineHoverPoint | null>(null);

  const { series, baseline, pointLabels, total, baselineTotal } = useMemo(() => {
    const now = new Date();

    // Billing's hourly costs are authoritative; session-priced spend is the
    // fallback for a range billing has no rows for yet. Either way this is
    // dollars only — the token headline below is sourced independently, off
    // the activity feed, and deliberately never derived from these figures.
    if (hourlyCosts) {
      const summary = summarizeHourlyCosts(range, hourlyCosts, now);
      if (summary.total > 0) return summary;
    } else {
      const slots = buildSlots(range, now);
      const baselineSlots = buildSlots(range, now);
      const index = new Map(slots.map((s, i) => [s.key, i]));

      for (const s of sessions) {
        if (!s.started_at) continue;
        const idx = index.get(slotKey(range, new Date(s.started_at)));
        if (idx === undefined) continue;
        const tokens = (s.tokens_in || 0) + (s.tokens_out || 0);
        // The pod always reports cost_usd as 0 (it can't price the LiteLLM alias),
        // so cost is derived from token counts — see estimateCostUsd in getSessions.
        slots[idx].value += estimateCostUsd(s.tokens_in || 0, s.tokens_out || 0);
        // Real token counts are in hand here, so the no-cognee baseline is priced
        // from them directly rather than inverted back out of a spend figure.
        baselineSlots[idx].value += estimateNoCogneeCostUsd(tokens);
      }
      const sessionSeries = slots.map((s) => s.value);
      const sessionBaseline = baselineSlots.map((s) => s.value);
      // Totals cover exactly the plotted slots. Sessions outside the range (or
      // with no start time) used to land in the headline but never in the chart,
      // which the saving multiplies into a visible discrepancy.
      const totalCost = sum(sessionSeries);
      if (totalCost > 0) {
        return {
          series: sessionSeries,
          baseline: sessionBaseline,
          pointLabels: slots.map((s) => s.label),
          total: totalCost,
          baselineTotal: sum(sessionBaseline),
        };
      }
    }

    // No spend recorded in the range: plot the empty grid rather than inventing
    // a curve. A tenant with no usage should read as no usage.
    const slots = buildSlots(range, now);
    const empty = slots.map(() => 0);
    return {
      series: empty,
      baseline: empty,
      pointLabels: slots.map((s) => s.label),
      total: 0,
      baselineTotal: 0,
    };
  }, [sessions, range, hourlyCosts]);

  const tokensTotal = useMemo(() => sumMeasuredTokens(range, runs, new Date()), [runs, range]);
  const savedTokens = tokensTotal === null ? null : tokensAvoided(tokensTotal);
  const savedUsd = Math.max(0, baselineTotal - total);
  // Both lines share one y-scale, so the axis has to cover the taller one.
  const yAxis = niceYAxis(Math.max(...series, ...baseline, 0));
  // Show at most ~4 x-axis ticks, always including the true last point.
  const ticks = pickTicks(pointLabels);
  // Reads "—" until billing answers, rather than claiming a $0 balance.
  const balanceLabel = balanceUsd === null ? "—" : `$${fmtUsd(balanceUsd)}`;

  return (
    <AsciiFrame
      label="Cost Savings"
      meta={
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ ...FONT, fontSize: 11, color: T.muted, display: "inline-flex", alignItems: "center", gap: 4 }}>
            Balance
            <span style={{ color: T.text, fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>{balanceLabel}</span>
          </span>
          <RangeToggle value={range} onChange={onRangeChange} />
        </div>
      }
      minHeight={260}
    >
      <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 10 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
            <span
              style={{
                ...FONT,
                fontSize: 26,
                fontWeight: 700,
                color: T.lavender,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {/* "—" until the activity feed has measured tokens for this range —
                  distinct from a genuine 0, which fmtTokens renders as "0". */}
              {savedTokens === null ? "—" : fmtTokens(savedTokens)}
            </span>
            <span style={{ ...FONT, fontSize: 12, color: T.muted }}>tokens saved</span>
          </div>
          {/* The estimate flag rides the dollar sub-line now, keeping the headline
              label to two words without dropping the disclosure. */}
          <div style={{ ...FONT, fontSize: 12, color: T.muted }}>
            ≈ <span style={{ color: T.text, fontVariantNumeric: "tabular-nums" }}>${fmtUsdWhole(savedUsd)}</span> at Opus 5 API rates
            <span
              style={{ color: T.faint }}
              title={`Estimated — assumes the same work costs ${NO_COGNEE_TOKEN_MULTIPLIER}× the tokens without cognee's recall. Tokens are the figure that holds on any plan: on usage-based billing they convert to cash, on a subscription they are limit headroom left unspent.`}
            > · estimated</span>
          </div>
        </div>

        <div style={{ flex: 1, minHeight: 84, display: "flex", gap: 8 }}>
          <div
            style={{
              ...FONT,
              width: AXIS_LABEL_WIDTH,
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              fontSize: 10,
              color: T.text,
              textAlign: "right",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {/* Top-down, so the ticks read in the same order as the plot. */}
            {[...yAxis.ticks].reverse().map((value) => (
              <span key={value}>${fmtAxisUsd(value, yAxis.max)}</span>
            ))}
          </div>

          <div style={{ flex: 1, position: "relative" }}>
            <Sparkline
              values={series}
              color={T.lavender}
              height={96}
              onHover={setHover}
              compare={{ values: baseline, color: T.muted, band: T.lavender }}
              yMax={yAxis.max}
              grid={{
                xIndices: ticks.map((t) => t.index),
                yFractions: yAxis.ticks.map((v) => v / yAxis.max),
                color: T.frameStrong,
              }}
            />

            {/* Hover reads off the chart itself instead of one stacked plate: each
                line carries its own value, the cursor carries the delta, and the
                date sits on the x-axis under the crosshair. */}
            {hover && pointLabels[hover.index] !== undefined && (
              <>
                <PointLabel
                  xPct={hover.xPct}
                  yPct={hover.comparePct ?? 0}
                  text={`$${fmtUsd(baseline[hover.index] ?? 0)}`}
                  color={T.text}
                  side="left"
                />
                <PointLabel
                  xPct={hover.xPct}
                  yPct={hover.yPct}
                  text={`$${fmtUsd(series[hover.index])}`}
                  color={T.lavender}
                  side="left"
                />
                <PointLabel
                  xPct={hover.xPct}
                  yPct={hover.cursorYPct}
                  text={`$${fmtUsd(Math.max(0, (baseline[hover.index] ?? 0) - series[hover.index]))} saved`}
                  color={T.lavender}
                  side="right"
                  emphasis
                />
              </>
            )}
          </div>
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <div style={{ width: AXIS_LABEL_WIDTH }} />
          <div style={{ flex: 1, position: "relative", height: 14, ...FONT, fontSize: 11, color: T.muted }}>
            {/* While hovering, the axis shows the hovered slot instead of its
                static ticks — printing both collides whenever the crosshair
                lands near a tick, which is exactly when it's being read. */}
            {hover && pointLabels[hover.index] !== undefined ? (
              <span
                style={{
                  position: "absolute",
                  left: `${hover.xPct}%`,
                  transform: `translateX(${hover.xPct > LABEL_FLIP_AT_PCT ? "-100%" : hover.xPct < 100 - LABEL_FLIP_AT_PCT ? "0" : "-50%"})`,
                  whiteSpace: "nowrap",
                  color: T.text,
                  fontWeight: 600,
                }}
              >
                {pointLabels[hover.index]}
              </span>
            ) : ticks.map(({ index, label }) => {
              const pct = series.length > 1 ? (index / (series.length - 1)) * 100 : 50;
              const isFirst = index === 0;
              const isLast = index === series.length - 1;
              return (
                <span
                  key={index}
                  style={{
                    position: "absolute",
                    left: `${pct}%`,
                    transform: isFirst ? "translateX(0)" : isLast ? "translateX(-100%)" : "translateX(-50%)",
                    whiteSpace: "nowrap",
                  }}
                >
                  {label}
                </span>
              );
            })
            }
          </div>
        </div>

        {/* Key shares the footer line with the breakdown link — it needs no row
            of its own, and the height goes to the plot instead. */}
        <div
          style={{
            borderTop: `1px solid ${T.frame}`,
            paddingTop: 10,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <button
            type="button"
            onClick={onViewBreakdown}
            style={{ ...FONT, fontSize: 12, color: T.lavender, background: "none", border: "none", cursor: "pointer", padding: 0 }}
          >
            View breakdown →
          </button>
          <div style={{ display: "flex", alignItems: "center", gap: 12, ...FONT, fontSize: 11, color: T.muted }}>
            <LegendKey color={T.lavender} label="with cognee" />
            <LegendKey color={T.muted} label="without cognee" />
          </div>
        </div>
      </div>
    </AsciiFrame>
  );
}
