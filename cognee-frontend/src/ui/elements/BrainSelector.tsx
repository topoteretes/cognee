"use client";

import { useCallback } from "react";
import { useFilter } from "@/ui/layout/FilterContext";
import useBoolean from "@/utils/useBoolean";
import useOutsideClick from "@/utils/useOutsideClick";

function DatabaseIcon({ color = "#52525B" }: { color?: string }) {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
    </svg>
  );
}

function Chevron() {
  return (
    <svg width="10" height="10" viewBox="0 0 12 12" fill="none" style={{ flexShrink: 0 }}>
      <path d="M3 4.5L6 7.5L9 4.5" stroke="rgba(255,255,255,0.4)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Check() {
  return (
    <svg width="11" height="11" viewBox="0 0 12 12" fill="none" style={{ marginLeft: "auto", flexShrink: 0 }}>
      <path d="M2.5 6L5 8.5L9.5 3.5" stroke="rgba(188,155,255,0.60)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

interface BrainSelectorProps {
  /** When false, hides the "All brains" option and defaults display to the first dataset. Default: true */
  allowAll?: boolean;
  /** Which side the dropdown opens toward. Default: "left" */
  align?: "left" | "right";
  /** Use "up" when the selector sits near the bottom of the viewport. Default: "down" */
  direction?: "down" | "up";
}

export default function BrainSelector({ allowAll = true, align = "left", direction = "down" }: BrainSelectorProps) {
  const { selectedDataset, setSelectedDataset, datasets } = useFilter();
  const { value: isOpen, toggle, setFalse: close } = useBoolean(false);
  const closeCallback = useCallback(() => close(), [close]);
  const ref = useOutsideClick<HTMLDivElement>(closeCallback, isOpen);

  // When "All brains" is not allowed, fall back to the first dataset for display
  const displayDataset = allowAll ? selectedDataset : (selectedDataset ?? datasets[0] ?? null);
  const isAllSelected = !selectedDataset;

  const dropdownPosition: React.CSSProperties = {
    position: "absolute",
    minWidth: 220,
    ...(align === "right" ? { right: 0 } : { left: 0 }),
    ...(direction === "up" ? { bottom: "calc(100% + 4px)" } : { top: "calc(100% + 4px)" }),
  };

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        onClick={toggle}
        className="cursor-pointer"
        style={{
          display: "flex", alignItems: "center", gap: 6,
          background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 8, padding: "6px 10px",
          fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.85)",
          fontFamily: "inherit",
        }}
      >
        <DatabaseIcon color={displayDataset ? "rgba(188,155,255,0.60)" : "rgba(255,255,255,0.45)"} />
        <span style={{ maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {displayDataset ? displayDataset.name : "All brains"}
        </span>
        <Chevron />
      </button>

      {isOpen && (
        <div
          onClick={close}
          style={{
            ...dropdownPosition,
            background: "rgba(10,10,10,0.92)", backdropFilter: "blur(16px)",
            borderRadius: 10,
            boxShadow: "0px 8px 30px rgba(0,0,0,0.5), 0px 0px 0px 1px rgba(255,255,255,0.1)",
            padding: 6, zIndex: 200,
            maxHeight: 320, overflowY: "auto",
          }}
        >
          {allowAll && (
            <>
              <div
                onClick={() => setSelectedDataset(null)}
                className="cursor-pointer"
                style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "8px 10px", borderRadius: 6,
                  background: isAllSelected ? "rgba(188,155,255,0.20)" : "transparent",
                }}
                onMouseEnter={e => { if (!isAllSelected) (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.06)"; }}
                onMouseLeave={e => { if (!isAllSelected) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
              >
                <span style={{ fontSize: 13, fontWeight: isAllSelected ? 500 : 400, color: isAllSelected ? "rgba(188,155,255,0.60)" : "rgba(237,236,234,0.7)", flex: 1 }}>
                  All brains
                </span>
                {isAllSelected && <Check />}
              </div>
              {datasets.length > 0 && <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "4px 0" }} />}
            </>
          )}
          {datasets.map((d) => {
            const isSelected = selectedDataset?.id === d.id;
            return (
            <div
              key={d.id}
              onClick={() => setSelectedDataset(d)}
              className="cursor-pointer"
              style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "8px 10px", borderRadius: 6,
                background: isSelected ? "rgba(188,155,255,0.20)" : "transparent",
              }}
              onMouseEnter={e => { if (!isSelected) (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.06)"; }}
              onMouseLeave={e => { if (!isSelected) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
            >
              <DatabaseIcon color={isSelected ? "rgba(188,155,255,0.60)" : "rgba(255,255,255,0.4)"} />
              <span style={{ fontSize: 13, fontWeight: isSelected ? 500 : 400, color: isSelected ? "rgba(188,155,255,0.60)" : "rgba(237,236,234,0.7)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {d.name}
              </span>
              {isSelected && <Check />}
            </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
