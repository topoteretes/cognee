"use client";

import { useCallback, useEffect, useRef, useState, type UIEvent } from "react";
import { Menu } from "@mantine/core";
import type { BrainsSummaryPayload } from "../types";
import type { GovernanceIndex, ReadableGovernanceIndex, ReadableListState } from "../useGovernanceIndex";
import { accessibleDatasetIds } from "../useGovernanceIndex";
import { sourceLabel } from "../computeBrainState";

interface BrainSwitcherProps {
  brains: BrainsSummaryPayload | null;
  // Narrowed to what this caller may read. The branded type is what enforces
  // that: the tenant-wide index is structurally identical and would otherwise
  // compile here.
  index: ReadableGovernanceIndex;
  // How the read-scoped list (`/v1/datasets/`) is doing. The index alone
  // cannot say why it holds no datasets: "still loading", "the list failed"
  // and "you may read nothing" all arrive here as an empty list, and they are
  // three different things to tell the user.
  listState: ReadableListState;
  activeDatasetId: string | null;
  onSelect: (datasetId: string) => void;
  hoveredPrincipalId: string | null;
}

// Fixed row and label heights make the cap below land half way through a row:
// a list that keeps going shows its next row cut off, which says "scroll"
// better than any cap on whole rows (the cut would otherwise fall at an
// arbitrary point, sometimes exactly between two rows, hiding the hint).
const ROW_HEIGHT = 48;
const SECTION_LABEL_HEIGHT = 22;
const VISIBLE_ROWS = 6.5;
// Dropdown padding (4 + 4) plus its 1px border on both sides.
const DROPDOWN_CHROME = 10;
const SCROLL_END_TOLERANCE = 1;

// Counts the section labels that sit above the cut, since each one pushes the
// half row down by its own height.
function dropdownMaxHeight(sectionSizes: number[]): number {
  let rowsLeft = VISIBLE_ROWS;
  let labels = 0;
  for (const size of sectionSizes) {
    if (rowsLeft <= 0) break;
    labels += 1;
    rowsLeft -= size;
  }
  return DROPDOWN_CHROME + labels * SECTION_LABEL_HEIGHT + VISIBLE_ROWS * ROW_HEIGHT;
}

function hasMoreBelow(element: HTMLElement | null): boolean {
  return element !== null &&
    element.scrollHeight - element.scrollTop - element.clientHeight > SCROLL_END_TOLERANCE;
}

const SECTION_LABEL_STYLE = {
  color: "#7E8CA6", fontSize: 9, letterSpacing: "0.05em",
  height: SECTION_LABEL_HEIGHT, display: "flex", alignItems: "center", padding: "0 10px",
} as const;

function holderCount(index: GovernanceIndex, datasetId: string): number {
  return index.users.filter((u) => (index.access[u.id] || {})[datasetId]).length;
}

// Replaces the always-expanded knowledge rail with a compact switcher —
// reviewer feedback: a brain is picked once and rarely revisited, so a
// permanent sidebar spent screen space (competing with SourcesRail right
// above it) on a choice that's mostly made once per session.
export default function BrainSwitcher({
  brains, index, listState, activeDatasetId, onSelect, hoveredPrincipalId,
}: BrainSwitcherProps) {
  // Controlled instead of relying on Mantine's own outside-click close:
  // d3-zoom's mousedown handler on the graph canvas stops propagation, so
  // Mantine's bubble-phase document listener never fires for clicks on the
  // graph and the dropdown stayed open. A capture-phase listener runs
  // before d3 can swallow the event.
  const [opened, setOpened] = useState(false);
  useEffect(() => {
    if (!opened) return;
    function closeOnOutsidePointer(e: PointerEvent) {
      const target = e.target instanceof Element ? e.target : null;
      if (target?.closest(".bv-brain-switcher-dropdown, .bv-brain-switcher-target")) return;
      setOpened(false);
    }
    document.addEventListener("pointerdown", closeOnOutsidePointer, true);
    return () => document.removeEventListener("pointerdown", closeOnOutsidePointer, true);
  }, [opened]);

  // Whether the list continues below the visible part, which drives the fade.
  const [moreBelow, setMoreBelow] = useState(false);
  const dropdownRef = useRef<HTMLDivElement | null>(null);
  const attachDropdown = useCallback((element: HTMLDivElement | null) => {
    dropdownRef.current = element;
    setMoreBelow(hasMoreBelow(element));
  }, []);
  const updateMoreBelow = useCallback((event: UIEvent<HTMLDivElement>) => {
    setMoreBelow(hasMoreBelow(event.currentTarget));
  }, []);

  const reachable = hoveredPrincipalId ? accessibleDatasetIds(index, hoveredPrincipalId) : null;
  const datasets = index.datasets;
  const team = datasets.filter((d) => holderCount(index, d.id) > 1);
  const personal = datasets.filter((d) => holderCount(index, d.id) <= 1);
  useEffect(() => { setMoreBelow(hasMoreBelow(dropdownRef.current)); }, [datasets.length]);
  const activeDataset = datasets.find((d) => d.id === activeDatasetId);
  // Four cases, not two. A dataset can be on screen while this component is
  // not allowed to name it: opening from last visit's remembered id while the
  // readable list is still answering, or seeded off the governance graph
  // because that list failed outright. Labelling either one "select a dataset"
  // contradicts the graph rendering behind the chip — the complaint b9cbef4f
  // fixed by handing this component the tenant list instead, which is what
  // leaked. So the chip says what is happening and names nothing; the name
  // appears the moment the readable list vouches for it.
  const activeName = activeDataset
    ? String(activeDataset.name || "dataset")
    : !activeDatasetId
      ? "select a dataset"
      : listState === "loading"
        ? "opening…"
        : "current dataset";
  // Said in the dropdown rather than on the chip: the chip already reads
  // "select a dataset", and the reason belongs where the empty list is.
  const emptyNote = listState === "loading"
    ? "loading your datasets…"
    : listState === "unavailable"
      ? "couldn't load your datasets — reload to try again"
      : "no datasets you can read";

  const row = (datasetId: string, name: string, isTeam: boolean) => {
    const dimmed = reachable ? !reachable.has(datasetId) : false;
    // brains-summary hands these over directly; the previous /brains
    // payload only carried them implicitly, inside each dataset's full
    // node array, so this line used to mean deriving labels from a graph
    // fetch that cost seconds (COG-6258).
    const summary = brains?.[datasetId];
    const sourceNames = summary?.source_names ?? [];
    // brains-summary counts the whole graph as of the dataset's latest
    // cognify run, so 0 means "never cognified" rather than "empty result".
    // Worth saying out loud: a dataset with sources attached but no run looks
    // identical to a ready one in this list, and picking it opens an empty
    // Mindmap with nothing explaining why.
    const notCognified = summary !== undefined && summary.node_count === 0;
    return (
      <Menu.Item
        key={datasetId}
        onClick={() => onSelect(datasetId)}
        style={{ padding: "8px 10px", height: ROW_HEIGHT, boxSizing: "border-box" }}
      >
        <div className="flex min-w-0 items-baseline gap-1.5">
          <span className={`truncate text-xs font-semibold ${datasetId === activeDatasetId ? "text-[#43D9E8]" : "text-[#E9EEF6]"} ${dimmed ? "opacity-25" : ""}`}>
            {name}
          </span>
          <span className="shrink-0 rounded border border-[#2A3652] px-1 text-[8.5px] uppercase text-[#7E8CA6]">
            {isTeam ? "team" : "personal"}
          </span>
        </div>
        {notCognified ? (
          <div className={`truncate text-[10.5px] text-[#7E8CA6] ${dimmed ? "opacity-25" : ""}`}>not cognified yet</div>
        ) : (
          sourceNames.length > 0 && (
            <div className={`truncate text-[10.5px] text-[#7E8CA6] ${dimmed ? "opacity-25" : ""}`}>{sourceNames.map(sourceLabel).join(" · ")}</div>
          )
        )}
      </Menu.Item>
    );
  };

  return (
    <Menu shadow="md" width={220} position="bottom-start" withinPortal opened={opened} onChange={setOpened}>
      <Menu.Target>
        <button
          type="button"
          className="bv-brain-switcher-target flex max-w-[196px] items-center gap-1.5 rounded-[10px] border border-[#2A3652] bg-[#1A2438] px-2.5 py-1.5 text-left text-xs text-[#E9EEF6] hover:-translate-y-0.5 bv-motion transition-[transform,opacity,border-color]"
        >
          <span className="truncate">{activeName}</span>
          <span className="ml-auto shrink-0 text-[#7E8CA6]">▾</span>
        </button>
      </Menu.Target>
      {/* Capped and scrollable: Mantine's dropdown grows with its items, so a
          workspace with many datasets pushed the list past the viewport
          and the rows at the bottom could not be reached. */}
      <Menu.Dropdown
        ref={attachDropdown}
        onScroll={updateMoreBelow}
        className="bv-brain-switcher-dropdown"
        style={{ background: "#1A2438", border: "1px solid #2A3652", padding: 4, maxHeight: dropdownMaxHeight([team.length, personal.length].filter(Boolean)), overflowY: "auto" }}
      >
        {team.length > 0 && (
          <>
            <Menu.Label style={SECTION_LABEL_STYLE}>team</Menu.Label>
            {team.map((d) => row(d.id, String(d.name || "dataset"), true))}
          </>
        )}
        {personal.length > 0 && (
          <>
            <Menu.Label style={SECTION_LABEL_STYLE}>personal</Menu.Label>
            {personal.map((d) => row(d.id, String(d.name || "dataset"), false))}
          </>
        )}
        {datasets.length === 0 && (
          <Menu.Label style={{ color: "#7E8CA6", fontSize: 10, letterSpacing: "0.02em" }}>{emptyNote}</Menu.Label>
        )}
        <div aria-hidden className="bv-brain-switcher-fade" data-visible={moreBelow} />
      </Menu.Dropdown>
    </Menu>
  );
}
