"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { TrackPageView } from "@/modules/analytics";
import { useFilter } from "@/ui/layout/FilterContext";
import { channelForSessionId, datasetNameById, estimateCostUsd, runStatus, sessionBrainLabel } from "@/modules/sessions/getSessions";
import { actorColor, ownerDisplayName } from "@/ui/elements/AgentActivityTerminal";
import { actionForPipeline, ACTION_COLOR, formatCostUsd, sessionStatus, StatusPill, STATUS_META, type Action, type Status } from "@/app/(app)/dashboard/partials/redesign/ActivityPanel";
import { AsciiFrame } from "@/app/(app)/dashboard/partials/redesign/AsciiFrame";
import { FONT, MONO_FONT, T } from "@/app/(app)/dashboard/partials/redesign/mono";
import { formatDateTime, parseServerIso } from "@/utils/formatDate";
import { useActivityFeed } from "./hooks/useActivityFeed";
import type { TimeRange } from "@/modules/sessions/getSessions";

const RANGES: TimeRange[] = ["24h", "7d", "30d", "all"];

interface Row {
  key: string;
  ts: number;
  timestampIso: string | null;
  runId: string;
  status: Status;
  user: string;
  /** How the session reached Cognee — UI, API, or a named agent/IDE integration. */
  via: string;
  dataset: string;
  action: Action;
  tokens: number | null;
  cost: number | null;
}

function tsOf(iso: string | null): number {
  if (!iso) return 0;
  const t = new Date(iso).getTime();
  return Number.isNaN(t) ? 0 : t;
}

/** Range picker matching the Balance card's RangeToggle, generalized to the
 *  4-way TimeRange ("24h"/"7d"/"30d"/"all") a full activity table needs —
 *  the dashboard's RangeToggle only supports 3 (no "all"). */
function ActivityRangeToggle({ value, onChange }: { value: TimeRange; onChange: (r: TimeRange) => void }): React.ReactElement {
  return (
    <div style={{ ...FONT, display: "inline-flex", alignItems: "center", gap: 2, padding: 3, background: "rgba(255,255,255,0.03)", border: `1px solid ${T.frame}`, borderRadius: 0 }}>
      {RANGES.map((r) => {
        const active = r === value;
        return (
          <button
            key={r}
            type="button"
            onClick={() => onChange(r)}
            aria-pressed={active}
            style={{
              ...FONT,
              fontSize: 12,
              fontWeight: active ? 600 : 500,
              cursor: "pointer",
              border: "none",
              background: active ? T.purpleSoft : "transparent",
              color: active ? T.lavender : T.muted,
              borderRadius: 0,
              padding: "4px 11px",
              transition: "color 120ms, background 120ms",
            }}
          >
            {r}
          </button>
        );
      })}
    </div>
  );
}

const ACTION_OPTIONS = Object.keys(ACTION_COLOR) as Action[];
const STATUS_OPTIONS = Object.keys(STATUS_META) as Status[];

function ChevronIcon({ open }: { open: boolean }): React.ReactElement {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style={{ flexShrink: 0, transform: open ? "rotate(180deg)" : undefined, transition: "transform 120ms ease" }}>
      <path d="M2 3.5L5 6.5L8 3.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function CheckIcon(): React.ReactElement {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ flexShrink: 0 }}>
      <path d="M2.5 6.2L4.7 8.4L9.2 3.4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function titleCase(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

const CLEAR_BUTTON_STYLE: React.CSSProperties = { ...FONT, fontSize: 12, color: T.lavender, background: "none", border: "none", cursor: "pointer", padding: "6px 8px 2px" };
const FILTER_INPUT_STYLE: React.CSSProperties = { ...FONT, fontSize: 13, color: T.text, background: "rgba(255,255,255,0.04)", border: `1px solid ${T.frame}`, borderRadius: 0, padding: "6px 8px", width: "100%", colorScheme: "dark" };

// ── Categorical multi-select filter (Action, Status, User, Via, Dataset) ──

type FilterOperator = "in" | "notIn";
interface FilterState { operator: FilterOperator; values: string[] }
const EMPTY_FILTER: FilterState = { operator: "in", values: [] };
const CATEGORICAL_OPERATOR_LABELS: Record<FilterOperator, string> = { in: "is any of", notIn: "is none of" };
const CATEGORICAL_OPERATOR_ORDER: FilterOperator[] = ["in", "notIn"];

/** values.length === 0 means "no constraint" regardless of operator — an
 *  empty "is none of" would otherwise vacuously exclude nothing, which reads
 *  as "match everything" anyway, so both operators agree when unset. */
function matchesFilter(filter: FilterState, fieldValue: string): boolean {
  if (filter.values.length === 0) return true;
  const included = filter.values.includes(fieldValue);
  return filter.operator === "in" ? included : !included;
}

// ── Date filter (Time) ──
//
// Mirrors the Qonto reference: a "Date" header with a small "is ▾" mode
// dropdown. The default mode ("is") shows one-click relative presets
// (Today, This week, …); switching the mode to a specific comparison
// ("is on"/"is between"/…) swaps the body to real date input(s). `uiMode`
// is purely presentational — `operator`/`date`/`date2` are what filtering
// actually reads, so a preset just resolves straight into those.
type TimeOperator = "on" | "before" | "after" | "onOrBefore" | "onOrAfter" | "between";
type TimeUiMode = "is" | TimeOperator;
interface TimeFilterState { uiMode: TimeUiMode; operator: TimeOperator; date: string; date2: string }
const EMPTY_TIME_FILTER: TimeFilterState = { uiMode: "is", operator: "on", date: "", date2: "" };

const TIME_MODE_LABELS: Record<TimeUiMode, string> = {
  is: "is",
  between: "is between",
  on: "is on",
  before: "is before",
  after: "is after",
  onOrBefore: "is on or before",
  onOrAfter: "is on or after",
};
const TIME_MODE_ORDER: TimeUiMode[] = ["is", "between", "on", "before", "after", "onOrBefore", "onOrAfter"];

function timeOperatorForMode(mode: TimeUiMode): TimeOperator {
  return mode === "is" ? "on" : mode;
}

function toDateInputStr(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
function addDaysStr(delta: number): string {
  const d = new Date();
  d.setDate(d.getDate() + delta);
  return toDateInputStr(d);
}
function todayStr(): string { return toDateInputStr(new Date()); }
function startOfWeekStr(): string {
  const d = new Date();
  d.setDate(d.getDate() - ((d.getDay() + 6) % 7)); // Monday start
  return toDateInputStr(d);
}
function startOfMonthStr(): string { const d = new Date(); d.setDate(1); return toDateInputStr(d); }
function startOfYearStr(): string { const d = new Date(); d.setMonth(0, 1); return toDateInputStr(d); }

interface DatePreset { label: string; operator: TimeOperator; date: string; date2: string }
function datePresets(): DatePreset[] {
  const today = todayStr();
  return [
    { label: "Today", operator: "on", date: today, date2: "" },
    { label: "Yesterday", operator: "on", date: addDaysStr(-1), date2: "" },
    { label: "This week", operator: "between", date: startOfWeekStr(), date2: today },
    { label: "Past week", operator: "between", date: addDaysStr(-7), date2: today },
    { label: "This month", operator: "between", date: startOfMonthStr(), date2: today },
    { label: "Past month", operator: "between", date: addDaysStr(-30), date2: today },
    { label: "This year", operator: "between", date: startOfYearStr(), date2: today },
    { label: "Past year", operator: "between", date: addDaysStr(-365), date2: today },
  ];
}

function matchesTime(filter: TimeFilterState, iso: string | null): boolean {
  if (!filter.date) return true;
  const rowDate = iso ? parseServerIso(iso) : null;
  if (!rowDate || Number.isNaN(rowDate.getTime())) return false;
  const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const rowDay = startOfDay(rowDate);
  const target = startOfDay(new Date(`${filter.date}T00:00:00`));
  switch (filter.operator) {
    case "on": return rowDay === target;
    case "before": return rowDay < target;
    case "after": return rowDay > target;
    case "onOrBefore": return rowDay <= target;
    case "onOrAfter": return rowDay >= target;
    case "between": {
      if (!filter.date2) return rowDay >= target;
      const target2 = startOfDay(new Date(`${filter.date2}T00:00:00`));
      return rowDay >= Math.min(target, target2) && rowDay <= Math.max(target, target2);
    }
    default: return true;
  }
}

// ── Text filter (Run ID) ──

type TextOperator = "contains" | "notContains";
interface TextFilterState { operator: TextOperator; query: string }
const EMPTY_TEXT_FILTER: TextFilterState = { operator: "contains", query: "" };
const TEXT_OPERATOR_LABELS: Record<TextOperator, string> = { contains: "contains", notContains: "does not contain" };
const TEXT_OPERATOR_ORDER: TextOperator[] = ["contains", "notContains"];

function matchesText(filter: TextFilterState, value: string): boolean {
  const q = filter.query.trim().toLowerCase();
  if (q === "") return true;
  const included = value.toLowerCase().includes(q);
  return filter.operator === "contains" ? included : !included;
}

// ── Numeric range filter (Tokens, Cost) ──

interface RangeFilterState { min: string; max: string }
const EMPTY_RANGE_FILTER: RangeFilterState = { min: "", max: "" };

function matchesRange(filter: RangeFilterState, value: number | null): boolean {
  const min = filter.min === "" ? null : Number(filter.min);
  const max = filter.max === "" ? null : Number(filter.max);
  if (min === null && max === null) return true;
  if (value === null) return false;
  if (min !== null && value < min) return false;
  if (max !== null && value > max) return false;
  return true;
}

function CheckboxRow({ label, checked, onToggle }: { label: string; checked: boolean; onToggle: () => void }): React.ReactElement {
  return (
    <button
      type="button"
      onClick={onToggle}
      style={{ ...FONT, fontSize: 13, display: "flex", alignItems: "center", gap: 8, width: "100%", textAlign: "left", padding: "6px 8px", borderRadius: 0, border: "none", background: "transparent", color: T.text, cursor: "pointer" }}
      onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
    >
      <span style={{ width: 15, height: 15, borderRadius: 0, flexShrink: 0, border: `1.5px solid ${checked ? T.lavender : T.frameStrong}`, background: checked ? T.lavender : "transparent", display: "flex", alignItems: "center", justifyContent: "center" }}>
        {checked && <span style={{ color: "#1e1e1c" }}><CheckIcon /></span>}
      </span>
      {label}
    </button>
  );
}

function RadioRow({ label, selected, onClick }: { label: string; selected: boolean; onClick: () => void }): React.ReactElement {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{ ...FONT, fontSize: 13, display: "flex", alignItems: "center", gap: 8, width: "100%", textAlign: "left", padding: "6px 8px", borderRadius: 0, border: "none", background: selected ? T.purpleSoft : "transparent", color: selected ? T.lavender : T.text, cursor: "pointer" }}
      onMouseEnter={(e) => { if (!selected) e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
      onMouseLeave={(e) => { if (!selected) e.currentTarget.style.background = "transparent"; }}
    >
      <span style={{ width: 14, height: 14, borderRadius: "50%", flexShrink: 0, border: `1.5px solid ${selected ? T.lavender : T.frameStrong}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
        {selected && <span style={{ width: 6, height: 6, borderRadius: "50%", background: T.lavender }} />}
      </span>
      {label}
    </button>
  );
}

/** Shared pill-button + floating-panel chrome every filter kind renders its
 *  own body into — owns the open/close state wiring (outside-click, Escape)
 *  so each filter type below only has to describe its own controls. Pills
 *  use the "raised surface" token (T.chrome) instead of pure black so they
 *  actually read as distinct, clickable tiles against the dark page. */
function FilterPopover({ label, active, summary, isOpen, onToggle, panelWidth = 230, children }: { label: string; active: boolean; summary: string | null; isOpen: boolean; onToggle: () => void; panelWidth?: number; children: React.ReactNode }): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) return;
    const handleClick = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) onToggle(); };
    const handleKey = (e: KeyboardEvent) => { if (e.key === "Escape") onToggle(); };
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [isOpen, onToggle]);

  const baseBg = active ? T.purpleSoft : T.chrome;

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        type="button"
        onClick={onToggle}
        style={{
          ...FONT, fontSize: 13, fontWeight: 500, display: "inline-flex", alignItems: "center", gap: 6,
          color: active ? T.lavender : T.text, background: isOpen ? T.purpleSoft : baseBg,
          border: `1px solid ${active || isOpen ? T.purple : T.frameStrong}`, borderRadius: 0, padding: "7px 14px", cursor: "pointer",
          transition: "background 120ms ease, border-color 120ms ease",
        }}
        onMouseEnter={(e) => { if (!active && !isOpen) e.currentTarget.style.background = "rgba(255,255,255,0.08)"; }}
        onMouseLeave={(e) => { if (!active && !isOpen) e.currentTarget.style.background = baseBg; }}
      >
        {titleCase(label)}
        {summary && <span style={{ color: active ? T.lavender : T.muted }}>: {summary}</span>}
        <ChevronIcon open={isOpen} />
      </button>
      {isOpen && (
        <div
          style={{
            position: "absolute", top: "calc(100% + 6px)", left: 0, zIndex: 20, minWidth: panelWidth,
            background: T.panel, border: `1px solid ${T.frameStrong}`, borderRadius: 0,
            boxShadow: "0 16px 40px rgba(0,0,0,0.6)", padding: 8,
          }}
        >
          {children}
        </div>
      )}
    </div>
  );
}

/** Compact "is ▾" nested dropdown for the Time filter's comparison mode —
 *  pre-selected value always visible, full option list only on click. */
function MiniDropdown({ value, options, onSelect }: { value: string; options: string[]; onSelect: (label: string) => void }): React.ReactElement {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{ ...FONT, fontSize: 12, fontWeight: 600, color: T.lavender, background: T.purpleSoft, border: "none", borderRadius: 0, padding: "3px 8px", cursor: "pointer", display: "inline-flex", alignItems: "center", gap: 4 }}
      >
        {value}
        <ChevronIcon open={open} />
      </button>
      {open && (
        <div style={{ position: "absolute", top: "calc(100% + 4px)", right: 0, zIndex: 30, minWidth: 170, background: T.panel, border: `1px solid ${T.frameStrong}`, borderRadius: 0, boxShadow: "0 12px 32px rgba(0,0,0,0.6)", padding: 4 }}>
          {options.map((opt) => {
            const selected = opt === value;
            return (
              <button
                key={opt}
                type="button"
                onClick={() => { onSelect(opt); setOpen(false); }}
                style={{ ...FONT, fontSize: 13, display: "block", width: "100%", textAlign: "left", padding: "6px 8px", borderRadius: 0, border: "none", background: selected ? "rgba(255,255,255,0.08)" : "transparent", color: T.text, cursor: "pointer" }}
                onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.08)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = selected ? "rgba(255,255,255,0.08)" : "transparent"; }}
              >
                {opt}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function PanelHeader({ children }: { children: React.ReactNode }): React.ReactElement {
  return <div style={{ ...FONT, fontSize: 13, fontWeight: 600, color: T.text, paddingBottom: 6, marginBottom: 6, borderBottom: `1px solid ${T.frame}` }}>{children}</div>;
}

/** "{Field} {operator ▾}" panel header, tight-packed so it reads as one
 *  phrase (e.g. "Date is ▾", "Action is any of ▾") instead of two elements
 *  pushed to opposite ends of the panel. */
function FieldOperatorHeader({ field, operatorLabel, operatorOptions, onSelectOperator }: { field: string; operatorLabel: string; operatorOptions: string[]; onSelectOperator: (label: string) => void }): React.ReactElement {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, paddingBottom: 8, marginBottom: 6, borderBottom: `1px solid ${T.frame}` }}>
      <span style={{ ...FONT, fontSize: 13, fontWeight: 600, color: T.text }}>{field}</span>
      <MiniDropdown value={operatorLabel} options={operatorOptions} onSelect={onSelectOperator} />
    </div>
  );
}

/** "Action is any of Recall, Improve" — categorical multi-select with
 *  inclusion/exclusion, used for Action/Status/User/Via/Dataset. */
function CategoricalFilterDropdown({ label, value, options, onChange, isOpen, onToggle }: { label: string; value: FilterState; options: string[]; onChange: (v: FilterState) => void; isOpen: boolean; onToggle: () => void }): React.ReactElement {
  const active = value.values.length > 0;
  const toggleValue = (opt: string): void => {
    const values = value.values.includes(opt) ? value.values.filter((v) => v !== opt) : [...value.values, opt];
    onChange({ ...value, values });
  };
  const summary = active
    ? `${value.operator === "notIn" ? "not " : ""}${value.values.length <= 2 ? value.values.map(titleCase).join(", ") : `${value.values.length} selected`}`
    : null;
  const selectOperator = (opLabel: string): void => {
    const operator = CATEGORICAL_OPERATOR_ORDER.find((o) => CATEGORICAL_OPERATOR_LABELS[o] === opLabel) ?? "in";
    onChange({ ...value, operator });
  };

  return (
    <FilterPopover label={label} active={active} summary={summary} isOpen={isOpen} onToggle={onToggle}>
      <FieldOperatorHeader
        field={titleCase(label)}
        operatorLabel={CATEGORICAL_OPERATOR_LABELS[value.operator]}
        operatorOptions={CATEGORICAL_OPERATOR_ORDER.map((o) => CATEGORICAL_OPERATOR_LABELS[o])}
        onSelectOperator={selectOperator}
      />
      <div style={{ maxHeight: 240, overflowY: "auto" }}>
        {options.map((opt) => (
          <CheckboxRow key={opt} label={titleCase(opt)} checked={value.values.includes(opt)} onToggle={() => toggleValue(opt)} />
        ))}
      </div>
      {active && <button type="button" onClick={() => onChange(EMPTY_FILTER)} style={CLEAR_BUTTON_STYLE}>Clear</button>}
    </FilterPopover>
  );
}

/** "Date is Today" / "Date is between 2026-08-06 and 2026-08-10" — condition-
 *  first filtering: pick the comparison mode via the small "is ▾" dropdown,
 *  then either a one-click relative preset (default "is" mode) or a real
 *  date input for the specific comparisons. */
function TimeFilterDropdown({ value, onChange, isOpen, onToggle }: { value: TimeFilterState; onChange: (v: TimeFilterState) => void; isOpen: boolean; onToggle: () => void }): React.ReactElement {
  const active = value.date !== "";
  const modeLabel = TIME_MODE_LABELS[value.uiMode];
  const summary = active ? `${modeLabel} ${value.date}${value.operator === "between" && value.date2 ? ` – ${value.date2}` : ""}` : null;

  const selectMode = (label: string): void => {
    const mode = TIME_MODE_ORDER.find((m) => TIME_MODE_LABELS[m] === label) ?? "is";
    if (mode === value.uiMode) return;
    onChange({ uiMode: mode, operator: timeOperatorForMode(mode), date: "", date2: "" });
  };

  return (
    <FilterPopover label="date" active={active} summary={summary} isOpen={isOpen} onToggle={onToggle} panelWidth={220}>
      <FieldOperatorHeader field="Date" operatorLabel={modeLabel} operatorOptions={TIME_MODE_ORDER.map((m) => TIME_MODE_LABELS[m])} onSelectOperator={selectMode} />

      {value.uiMode === "is" ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 1, maxHeight: 260, overflowY: "auto" }}>
          {datePresets().map((p) => {
            const selected = value.date === p.date && value.operator === p.operator && value.date2 === p.date2;
            return <RadioRow key={p.label} label={p.label} selected={selected} onClick={() => onChange({ ...value, operator: p.operator, date: p.date, date2: p.date2 })} />;
          })}
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <input type="date" value={value.date} onChange={(e) => onChange({ ...value, date: e.target.value })} style={FILTER_INPUT_STYLE} />
          {value.uiMode === "between" && (
            <input type="date" value={value.date2} onChange={(e) => onChange({ ...value, date2: e.target.value })} style={FILTER_INPUT_STYLE} />
          )}
        </div>
      )}

      {active && <button type="button" onClick={() => onChange(EMPTY_TIME_FILTER)} style={CLEAR_BUTTON_STYLE}>Clear</button>}
    </FilterPopover>
  );
}

/** "Run ID contains sess-2018" — substring match; a checkbox list would be
 *  useless here since every run/session id is effectively unique. */
function RunIdFilterDropdown({ value, onChange, isOpen, onToggle }: { value: TextFilterState; onChange: (v: TextFilterState) => void; isOpen: boolean; onToggle: () => void }): React.ReactElement {
  const active = value.query.trim() !== "";
  const summary = active ? `${TEXT_OPERATOR_LABELS[value.operator]} "${value.query.trim()}"` : null;
  const selectOperator = (opLabel: string): void => {
    const operator = TEXT_OPERATOR_ORDER.find((o) => TEXT_OPERATOR_LABELS[o] === opLabel) ?? "contains";
    onChange({ ...value, operator });
  };

  return (
    <FilterPopover label="run id" active={active} summary={summary} isOpen={isOpen} onToggle={onToggle} panelWidth={220}>
      <FieldOperatorHeader field="Run ID" operatorLabel={TEXT_OPERATOR_LABELS[value.operator]} operatorOptions={TEXT_OPERATOR_ORDER.map((o) => TEXT_OPERATOR_LABELS[o])} onSelectOperator={selectOperator} />
      <input
        type="text"
        value={value.query}
        onChange={(e) => onChange({ ...value, query: e.target.value })}
        placeholder="e.g. sess-2018"
        style={FILTER_INPUT_STYLE}
      />
      {active && <button type="button" onClick={() => onChange(EMPTY_TEXT_FILTER)} style={CLEAR_BUTTON_STYLE}>Clear</button>}
    </FilterPopover>
  );
}

/** Min/max numeric range — used for Tokens and Cost. */
function RangeFilterDropdown({ label, value, onChange, isOpen, onToggle }: { label: string; value: RangeFilterState; onChange: (v: RangeFilterState) => void; isOpen: boolean; onToggle: () => void }): React.ReactElement {
  const active = value.min !== "" || value.max !== "";
  const summary = active ? `${value.min || "…"} – ${value.max || "…"}` : null;

  return (
    <FilterPopover label={label} active={active} summary={summary} isOpen={isOpen} onToggle={onToggle} panelWidth={200}>
      <PanelHeader>{titleCase(label)} range</PanelHeader>
      <div style={{ display: "flex", gap: 8 }}>
        <div style={{ flex: 1 }}>
          <label style={{ ...FONT, fontSize: 11, color: T.muted, display: "block", marginBottom: 3 }}>Min</label>
          <input type="number" value={value.min} onChange={(e) => onChange({ ...value, min: e.target.value })} placeholder="0" style={FILTER_INPUT_STYLE} />
        </div>
        <div style={{ flex: 1 }}>
          <label style={{ ...FONT, fontSize: 11, color: T.muted, display: "block", marginBottom: 3 }}>Max</label>
          <input type="number" value={value.max} onChange={(e) => onChange({ ...value, max: e.target.value })} placeholder="∞" style={FILTER_INPUT_STYLE} />
        </div>
      </div>
      {active && <button type="button" onClick={() => onChange(EMPTY_RANGE_FILTER)} style={CLEAR_BUTTON_STYLE}>Clear</button>}
    </FilterPopover>
  );
}

export default function ActivityPage(): React.ReactElement {
  const router = useRouter();
  const { agents, datasets } = useFilter();
  const [range, setRange] = useState<TimeRange>("7d");
  const [actionFilter, setActionFilter] = useState<FilterState>(EMPTY_FILTER);
  const [statusFilter, setStatusFilter] = useState<FilterState>(EMPTY_FILTER);
  const [userFilter, setUserFilter] = useState<FilterState>(EMPTY_FILTER);
  const [timeFilter, setTimeFilter] = useState<TimeFilterState>(EMPTY_TIME_FILTER);
  const [viaFilter, setViaFilter] = useState<FilterState>(EMPTY_FILTER);
  const [datasetFilter, setDatasetFilter] = useState<FilterState>(EMPTY_FILTER);
  const [runIdFilter, setRunIdFilter] = useState<TextFilterState>(EMPTY_TEXT_FILTER);
  const [tokensFilter, setTokensFilter] = useState<RangeFilterState>(EMPTY_RANGE_FILTER);
  const [costFilter, setCostFilter] = useState<RangeFilterState>(EMPTY_RANGE_FILTER);
  const [openFilter, setOpenFilter] = useState<string | null>(null);
  const [showMoreFilters, setShowMoreFilters] = useState(false);
  const { runs, sessions, loading, error, refetch } = useActivityFeed(range);

  const rows = useMemo(() => {
    const agentNameById = new Map(agents.map((a) => [a.id, ownerDisplayName(a.email)]));
    const nameById = datasetNameById(datasets);
    const cutoffMs = rangeCutoffMs(range);

    const list: Row[] = [];

    for (const r of runs) {
      // Operation rows (kind: "operation") are recall/search/remember/etc. —
      // already represented below via `sessions` at the session-lifecycle
      // granularity. Counting them again here would double the tokens/cost
      // totals, so this table only takes named pipelines (cognify, add,
      // memify, indexing), the same kind-based split AgentActivityTerminal
      // uses for its own log.
      if (r.kind !== "pipeline") continue;
      const ts = tsOf(r.created_at);
      if (cutoffMs !== null && ts < cutoffMs) continue; // the runs endpoint isn't range-scoped server-side
      // SDK-399 tokens are independently nullable: null means "not measured".
      const tokens = r.tokens_in === null && r.tokens_out === null ? null : (r.tokens_in ?? 0) + (r.tokens_out ?? 0);
      list.push({
        key: `run-${r.pipeline_run_id || r.id}`,
        ts,
        timestampIso: r.created_at,
        runId: r.pipeline_run_id || r.id,
        status: runStatus(r.status),
        user: ownerDisplayName(r.owner_email),
        via: "—", // pipeline runs carry no access-channel signal
        dataset: r.dataset_name || (r.dataset_id ? nameById.get(r.dataset_id) : undefined) || "—",
        action: actionForPipeline(r.pipeline_name || ""),
        tokens,
        // Estimated the same way as everywhere else in the dashboard — see
        // CostPanel — never the authoritative per-operation dollar figure,
        // which the backend doesn't expose.
        cost: tokens === null ? null : estimateCostUsd(r.tokens_in ?? 0, r.tokens_out ?? 0),
      });
    }

    for (const s of sessions) {
      const iso = s.started_at || s.last_activity_at;
      list.push({
        key: `ses-${s.session_id}`,
        ts: tsOf(iso),
        timestampIso: iso,
        runId: s.session_id,
        status: sessionStatus(s.effective_status || s.status || ""),
        user: agentNameById.get(s.user_id) ?? (s.user_id.includes("@") ? ownerDisplayName(s.user_id) : "agent"),
        via: channelForSessionId(s.session_id),
        dataset: sessionBrainLabel(s.dataset_id, nameById),
        action: "recall",
        tokens: (s.tokens_in || 0) + (s.tokens_out || 0),
        cost: estimateCostUsd(s.tokens_in || 0, s.tokens_out || 0),
      });
    }

    return list.sort((a, b) => b.ts - a.ts);
  }, [runs, sessions, agents, datasets, range]);

  // Filter option lists are derived from the current rows (not fixed enums)
  // so they only ever offer values that actually appear in this range.
  const userOptions = useMemo(() => [...new Set(rows.map((r) => r.user))].sort(), [rows]);
  const viaOptions = useMemo(() => [...new Set(rows.map((r) => r.via).filter((v) => v !== "—"))].sort(), [rows]);
  const datasetOptions = useMemo(() => [...new Set(rows.map((r) => r.dataset).filter((d) => d !== "—"))].sort(), [rows]);

  const filteredRows = useMemo(() => rows.filter((r) =>
    matchesFilter(actionFilter, r.action) &&
    matchesFilter(statusFilter, r.status) &&
    matchesFilter(userFilter, r.user) &&
    matchesTime(timeFilter, r.timestampIso) &&
    matchesFilter(viaFilter, r.via) &&
    matchesFilter(datasetFilter, r.dataset) &&
    matchesText(runIdFilter, r.runId) &&
    matchesRange(tokensFilter, r.tokens) &&
    matchesRange(costFilter, r.cost)
  ), [rows, actionFilter, statusFilter, userFilter, timeFilter, viaFilter, datasetFilter, runIdFilter, tokensFilter, costFilter]);

  const extraFilterActive = [viaFilter, datasetFilter].some((f) => f.values.length > 0) || runIdFilter.query.trim() !== "" || tokensFilter.min !== "" || tokensFilter.max !== "" || costFilter.min !== "" || costFilter.max !== "";
  const filtersActive = actionFilter.values.length > 0 || statusFilter.values.length > 0 || userFilter.values.length > 0 || timeFilter.date !== "" || extraFilterActive;
  const clearFilters = (): void => {
    setActionFilter(EMPTY_FILTER);
    setStatusFilter(EMPTY_FILTER);
    setUserFilter(EMPTY_FILTER);
    setTimeFilter(EMPTY_TIME_FILTER);
    setViaFilter(EMPTY_FILTER);
    setDatasetFilter(EMPTY_FILTER);
    setRunIdFilter(EMPTY_TEXT_FILTER);
    setTokensFilter(EMPTY_RANGE_FILTER);
    setCostFilter(EMPTY_RANGE_FILTER);
  };

  const totals = useMemo(() => filteredRows.reduce(
    (acc, r) => ({ tokens: acc.tokens + (r.tokens ?? 0), cost: acc.cost + (r.cost ?? 0) }),
    { tokens: 0, cost: 0 },
  ), [filteredRows]);

  const toggle = (key: string) => () => setOpenFilter(openFilter === key ? null : key);

  return (
    <div style={{ minHeight: "100%", padding: "clamp(16px, 3vw, 32px)", display: "flex", flexDirection: "column", gap: 20 }}>
      <TrackPageView page="Activity" />

      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ ...FONT, margin: 0, fontSize: 20, fontWeight: 300, color: T.text }}>Activity</h1>
          <p style={{ ...FONT, margin: "5px 0 0", fontSize: 13, color: T.muted }}>
            Every pipeline run and session across your workspace. Tokens are measured; cost is estimated.
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <button
            type="button"
            onClick={() => refetch()}
            style={{ ...FONT, fontSize: 12, color: T.muted, background: "none", border: `1px solid ${T.frame}`, borderRadius: 0, padding: "6px 12px", cursor: "pointer" }}
          >
            Refresh
          </button>
          <ActivityRangeToggle value={range} onChange={setRange} />
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", background: T.panel, border: `1px solid ${T.frame}`, borderRadius: 0, padding: 10 }}>
        <TimeFilterDropdown value={timeFilter} onChange={setTimeFilter} isOpen={openFilter === "time"} onToggle={toggle("time")} />
        <CategoricalFilterDropdown label="action" value={actionFilter} options={ACTION_OPTIONS} onChange={setActionFilter} isOpen={openFilter === "action"} onToggle={toggle("action")} />
        <CategoricalFilterDropdown label="status" value={statusFilter} options={STATUS_OPTIONS} onChange={setStatusFilter} isOpen={openFilter === "status"} onToggle={toggle("status")} />
        <CategoricalFilterDropdown label="user" value={userFilter} options={userOptions} onChange={setUserFilter} isOpen={openFilter === "user"} onToggle={toggle("user")} />

        {(showMoreFilters || extraFilterActive) && (
          <>
            <CategoricalFilterDropdown label="via" value={viaFilter} options={viaOptions} onChange={setViaFilter} isOpen={openFilter === "via"} onToggle={toggle("via")} />
            <CategoricalFilterDropdown label="dataset" value={datasetFilter} options={datasetOptions} onChange={setDatasetFilter} isOpen={openFilter === "dataset"} onToggle={toggle("dataset")} />
            <RunIdFilterDropdown value={runIdFilter} onChange={setRunIdFilter} isOpen={openFilter === "runid"} onToggle={toggle("runid")} />
            <RangeFilterDropdown label="tokens" value={tokensFilter} onChange={setTokensFilter} isOpen={openFilter === "tokens"} onToggle={toggle("tokens")} />
            <RangeFilterDropdown label="cost (est.)" value={costFilter} onChange={setCostFilter} isOpen={openFilter === "cost"} onToggle={toggle("cost")} />
          </>
        )}

        <button
          type="button"
          onClick={() => setShowMoreFilters((v) => !v)}
          style={{ ...FONT, fontSize: 13, color: T.muted, background: "transparent", border: `1px dashed ${T.frameStrong}`, borderRadius: 0, padding: "7px 14px", cursor: "pointer" }}
          onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
        >
          {showMoreFilters ? "− Fewer filters" : "+ More filters"}
        </button>

        {filtersActive && (
          <button
            type="button"
            onClick={clearFilters}
            style={{ ...FONT, fontSize: 13, color: T.lavender, background: "none", border: "none", cursor: "pointer", padding: "7px 8px" }}
          >
            Clear filters
          </button>
        )}
      </div>

      <AsciiFrame label={null}>
        {error && (
          <div style={{ ...FONT, fontSize: 13, color: T.red, padding: "4px 0 16px" }}>
            Couldn&apos;t load activity — the workspace may still be starting up.
          </div>
        )}

        {!error && loading && (
          <div style={{ ...FONT, fontSize: 13, color: T.muted, padding: "4px 0 16px" }}>Loading…</div>
        )}

        {!error && !loading && rows.length === 0 && (
          <div style={{ ...FONT, fontSize: 13, color: T.faint, padding: "4px 0 16px" }}>No activity in this range yet.</div>
        )}

        {!error && !loading && rows.length > 0 && filteredRows.length === 0 && (
          <div style={{ ...FONT, fontSize: 13, color: T.faint, padding: "4px 0 16px" }}>No activity matches these filters.</div>
        )}

        {!error && filteredRows.length > 0 && (
          <div style={{ overflowX: "auto" }}>
            <table style={{ ...FONT, width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr>
                  <Th align="left">Timestamp</Th>
                  <Th align="left">Action</Th>
                  <Th align="left">Status</Th>
                  <Th align="left">User</Th>
                  <Th align="left">Via</Th>
                  <Th align="left">Dataset</Th>
                  <Th align="left">Run ID</Th>
                  <Th align="right">Tokens</Th>
                  <Th align="right">Cost (est.)</Th>
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((row) => (
                  <tr
                    key={row.key}
                    onClick={() => row.key.startsWith("ses-") && router.push(`/sessions?session=${encodeURIComponent(row.runId)}`)}
                    style={{ borderTop: `1px solid ${T.frame}`, cursor: row.key.startsWith("ses-") ? "pointer" : "default", background: "transparent", transition: "background 150ms ease" }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                  >
                    <Td>
                      <span style={{ color: T.muted, fontVariantNumeric: "tabular-nums" }}>{formatDateTime(row.timestampIso)}</span>
                    </Td>
                    <Td><span style={{ color: ACTION_COLOR[row.action] }}>{row.action}</span></Td>
                    <Td><StatusPill status={row.status} /></Td>
                    <Td>
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                        <span style={{ width: 6, height: 6, borderRadius: "50%", background: actorColor(row.user), flexShrink: 0 }} />
                        <span style={{ color: T.text }}>{row.user}</span>
                      </span>
                    </Td>
                    <Td><span style={{ color: T.muted }}>{row.via}</span></Td>
                    <Td><span style={{ color: T.muted }}>{row.dataset}</span></Td>
                    <Td>
                      <span style={{ ...MONO_FONT, fontSize: 12, color: T.muted }} title={row.runId}>{row.runId}</span>
                    </Td>
                    <Td align="right"><span style={{ ...MONO_FONT, fontSize: 12, color: row.tokens === null ? T.faint : T.muted, fontVariantNumeric: "tabular-nums" }}>{row.tokens === null ? "—" : row.tokens.toLocaleString()}</span></Td>
                    <Td align="right"><span style={{ ...MONO_FONT, fontSize: 12, color: row.cost === null ? T.faint : T.muted, fontVariantNumeric: "tabular-nums" }}>{formatCostUsd(row.cost)}</span></Td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr style={{ background: T.chrome }}>
                  <td colSpan={7} style={{ padding: "9px 12px", fontSize: 12, fontWeight: 700, color: T.text, borderTop: `1px solid ${T.frameStrong}` }}>
                    Total · {filteredRows.length} {filteredRows.length === 1 ? "event" : "events"}
                  </td>
                  <td style={{ textAlign: "right", padding: "9px 12px", fontSize: 12, fontWeight: 700, color: T.text, borderTop: `1px solid ${T.frameStrong}`, fontVariantNumeric: "tabular-nums" }}>
                    {totals.tokens.toLocaleString()}
                  </td>
                  <td style={{ textAlign: "right", padding: "9px 12px", fontSize: 12, fontWeight: 700, color: T.text, borderTop: `1px solid ${T.frameStrong}`, fontVariantNumeric: "tabular-nums" }}>
                    {formatCostUsd(totals.cost)}
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        )}
      </AsciiFrame>
    </div>
  );
}

function rangeCutoffMs(range: TimeRange): number | null {
  if (range === "all") return null;
  const days = range === "24h" ? 1 : range === "7d" ? 7 : 30;
  return Date.now() - days * 24 * 60 * 60 * 1000;
}

function Th({ children, align = "left" }: { children: React.ReactNode; align?: "left" | "right" }): React.ReactElement {
  return (
    <th style={{ textAlign: align, fontWeight: 600, fontSize: 12, color: T.muted, padding: "0 12px 8px", borderBottom: `1px solid ${T.frame}`, whiteSpace: "nowrap" }}>
      {children}
    </th>
  );
}

function Td({ children, align = "left" }: { children: React.ReactNode; align?: "left" | "right" }): React.ReactElement {
  return (
    <td style={{ textAlign: align, padding: "10px 12px", verticalAlign: "middle" }}>
      {children}
    </td>
  );
}
