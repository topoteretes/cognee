"use client";

import { useEffect, useState } from "react";
import { Menu } from "@mantine/core";
import type { BrainsPayload } from "../types";
import type { GovernanceIndex } from "../useGovernanceIndex";
import { accessibleDatasetIds } from "../useGovernanceIndex";
import { setsOf, sourceLabel } from "../computeBrainState";

interface BrainSwitcherProps {
  brains: BrainsPayload | null;
  index: GovernanceIndex;
  activeDatasetId: string | null;
  onSelect: (datasetId: string) => void;
  hoveredPrincipalId: string | null;
}

function holderCount(index: GovernanceIndex, datasetId: string): number {
  return index.users.filter((u) => (index.access[u.id] || {})[datasetId]).length;
}

// Replaces the always-expanded knowledge rail with a compact switcher —
// reviewer feedback: a brain is picked once and rarely revisited, so a
// permanent sidebar spent screen space (competing with SourcesRail right
// above it) on a choice that's mostly made once per session.
export default function BrainSwitcher({ brains, index, activeDatasetId, onSelect, hoveredPrincipalId }: BrainSwitcherProps) {
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

  const reachable = hoveredPrincipalId ? accessibleDatasetIds(index, hoveredPrincipalId) : null;
  const datasets = index.datasets;
  const team = datasets.filter((d) => holderCount(index, d.id) > 1);
  const personal = datasets.filter((d) => holderCount(index, d.id) <= 1);
  const activeDataset = datasets.find((d) => d.id === activeDatasetId);
  const activeName = activeDataset ? String(activeDataset.name || "brain") : "select a brain";

  const row = (datasetId: string, name: string, isTeam: boolean) => {
    const dimmed = reachable ? !reachable.has(datasetId) : false;
    const preview = brains?.[datasetId];
    const entities = preview?.nodes.filter((n) => n.stage === "entity") ?? [];
    const sourceNames = [...new Set(entities.flatMap(setsOf))];
    return (
      <Menu.Item
        key={datasetId}
        onClick={() => onSelect(datasetId)}
        style={{ padding: "8px 10px" }}
      >
        <div className="flex min-w-0 items-baseline gap-1.5">
          <span className={`truncate text-xs font-semibold ${datasetId === activeDatasetId ? "text-[#43D9E8]" : "text-[#E9EEF6]"} ${dimmed ? "opacity-25" : ""}`}>
            {name}
          </span>
          <span className="shrink-0 rounded border border-[#2A3652] px-1 text-[8.5px] uppercase text-[#7E8CA6]">
            {isTeam ? "team" : "personal"}
          </span>
        </div>
        {sourceNames.length > 0 && (
          <div className={`truncate text-[10.5px] text-[#7E8CA6] ${dimmed ? "opacity-25" : ""}`}>{sourceNames.map(sourceLabel).join(" · ")}</div>
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
      <Menu.Dropdown className="bv-brain-switcher-dropdown" style={{ background: "#1A2438", border: "1px solid #2A3652", padding: 4 }}>
        {team.length > 0 && (
          <>
            <Menu.Label style={{ color: "#7E8CA6", fontSize: 9, letterSpacing: "0.05em" }}>team</Menu.Label>
            {team.map((d) => row(d.id, String(d.name || "brain"), true))}
          </>
        )}
        {personal.length > 0 && (
          <>
            <Menu.Label style={{ color: "#7E8CA6", fontSize: 9, letterSpacing: "0.05em" }}>personal</Menu.Label>
            {personal.map((d) => row(d.id, String(d.name || "brain"), false))}
          </>
        )}
      </Menu.Dropdown>
    </Menu>
  );
}
