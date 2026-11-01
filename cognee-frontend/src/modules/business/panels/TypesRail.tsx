"use client";

import { Tooltip } from "@mantine/core";
import type { TypeNode } from "../sceneTypes";
import { UNTYPED_BUCKET, type ColorEncoding } from "../colorEncoding";

// Above this many types, a full card per type stops fitting the panel — see
// SourcesRail's TOP_SOURCE_CARD_LIMIT for the same reasoning (chips carry the
// long tail, ranked by member count).
const TOP_TYPE_CARD_LIMIT = 8;

interface TypesRailProps {
  typeNodes: TypeNode[] | null;
  encoding: ColorEncoding;
  focusTypes: Set<string> | null;
  onToggleFocus: (name: string) => void;
}

interface TypeCardProps {
  name: string;
  count: number;
  color: string;
  focused: boolean;
  dimmed: boolean;
  onToggleFocus: (name: string) => void;
}

function TypeCard({ name, count, color, focused, dimmed, onToggleFocus }: TypeCardProps) {
  return (
    <button
      type="button"
      onClick={() => onToggleFocus(name)}
      className={`bv-motion relative w-full min-w-0 cursor-pointer overflow-hidden rounded-[10px] border p-2.5 pl-3.5 text-left text-xs transition-[transform,opacity,border-color] duration-500 hover:-translate-y-0.5 ${
        focused ? "border-[#E9EEF6]" : "border-[#2A3652]"
      } bg-[#1A2438] ${dimmed ? "opacity-25" : ""}`}
    >
      <span className="absolute left-0 top-2 bottom-2 w-[3px] rounded" style={{ background: color }} />
      <Tooltip label={name} openDelay={200} withinPortal>
        <div className="truncate font-semibold text-[#E9EEF6]">{name}</div>
      </Tooltip>
      <div className="truncate text-[11px] text-[#7E8CA6]">{count} entities</div>
    </button>
  );
}

interface TypeChipProps {
  name: string;
  count: number;
  color: string;
  focused: boolean;
  dimmed: boolean;
  onToggleFocus: (name: string) => void;
}

function TypeChip({ name, count, color, focused, dimmed, onToggleFocus }: TypeChipProps) {
  return (
    <Tooltip label={name} openDelay={200} withinPortal>
      <button
        type="button"
        onClick={() => onToggleFocus(name)}
        aria-label={name}
        className={`bv-motion flex h-6 min-w-[24px] items-center justify-center gap-1 rounded-full border px-1.5 text-[10px] text-[#E9EEF6] transition-[transform,opacity,border-color] duration-500 hover:-translate-y-0.5 ${
          focused ? "border-[#E9EEF6]" : "border-[#2A3652]"
        } bg-[#1A2438] ${dimmed ? "opacity-25" : ""}`}
      >
        <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
        {count}
      </button>
    </Tooltip>
  );
}

// The node_set filter's sibling — same click-to-isolate mechanism
// (SourcesRail), keyed by business-model type (BrainState.entityType)
// instead of source. Composes with it: an entity has to pass whichever of
// the two lenses are active (businessEntityLayer.inLens).
export default function TypesRail({ typeNodes, encoding, focusTypes, onToggleFocus }: TypesRailProps) {
  const nodes = typeNodes ?? [];
  if (!nodes.length) return null;

  // The untyped bucket goes last however big it is: it has no color, so left
  // in rank it would sit gray above the colored types.
  const ranked = [...nodes].sort(
    (a, b) => Number(a.name === UNTYPED_BUCKET) - Number(b.name === UNTYPED_BUCKET) || b.members.length - a.members.length,
  );
  const overflowing = ranked.length > TOP_TYPE_CARD_LIMIT;
  const cardNodes = overflowing ? ranked.slice(0, TOP_TYPE_CARD_LIMIT) : ranked;
  const chipNodes = overflowing ? ranked.slice(TOP_TYPE_CARD_LIMIT) : [];

  return (
    <div className="p-2.5">
      <div className="px-1 text-[10px] uppercase tracking-widest text-[#7E8CA6]">types</div>
      <div className="mt-2 flex flex-col gap-2">
        {cardNodes.map((tn) => (
          <TypeCard
            key={tn.name}
            name={tn.name}
            count={tn.members.length}
            color={encoding.typeFill(tn.sets, null, tn.name)}
            focused={focusTypes?.has(tn.name) ?? false}
            dimmed={focusTypes ? !focusTypes.has(tn.name) : false}
            onToggleFocus={onToggleFocus}
          />
        ))}
      </div>
      {chipNodes.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5 px-1">
          {chipNodes.map((tn) => (
            <TypeChip
              key={tn.name}
              name={tn.name}
              count={tn.members.length}
              color={encoding.typeFill(tn.sets, null, tn.name)}
              focused={focusTypes?.has(tn.name) ?? false}
              dimmed={focusTypes ? !focusTypes.has(tn.name) : false}
              onToggleFocus={onToggleFocus}
            />
          ))}
        </div>
      )}
    </div>
  );
}
