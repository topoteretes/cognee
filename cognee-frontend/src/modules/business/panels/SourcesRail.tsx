"use client";

import { Tooltip } from "@mantine/core";
import { sourceLabel, sourceTooltipLabel } from "../computeBrainState";
import type { BrainState } from "../sceneTypes";

type SourcesBrainState = Pick<
  BrainState,
  "sourceNames" | "setEntityCount" | "setDocCount" | "setMemberCount" | "setColor"
>;

// Above this many sources, a full card per source (name, color bar,
// entities/items line) stops fitting the panel without endless scrolling —
// and every card gets a permanent filament thread to the canvas (see
// businessFilaments.ts), so a long tail of low-content sources means a long
// tail of threads crossing the scene for content nobody's looking at. Past
// the limit, only the most substantial sources (by entity count) keep a
// full card; the rest collapse into small color+count chips with no card
// ref (drawFilaments only knows about registered cards, so an unregistered
// chip simply never threads a line) and no name shown at rest — the color
// already matches the same source's territory in the graph, and the full
// name is one hover away via the same Tooltip the cards use.
const TOP_SOURCE_CARD_LIMIT = 8;

interface SourcesRailProps {
  brainState: SourcesBrainState | null;
  focusSets: Set<string> | null;
  onToggleFocus: (name: string) => void;
  // Lets the canvas draw the drifting-particle filament from this card to
  // its territory (customer_tutorial.html:6640-6663) — the card is real DOM,
  // the filament is canvas, so the canvas needs the card's screen position.
  registerCardRef: (name: string, el: HTMLElement | null) => void;
  // Ports the "cognify complete" growth flash (customer_tutorial.html
  // ~7218: card.classList.add('flash')) — briefly highlights whichever
  // source most of the newly-arrived entities came from.
  flashSourceName?: string | null;
  // Ports wireUserHover's "the rendered dataset's sources dim too when this
  // user can't read it" (customer_tutorial.html ~6085-6088) — a hovered
  // operator dims every source card at once (not per-source) when they lack
  // access to the currently active dataset. Undefined/true means unaffected.
  reachableByHoveredPrincipal?: boolean;
}

interface SourceCardProps {
  name: string;
  brainState: SourcesBrainState | null;
  focused: boolean;
  dimmed: boolean;
  flashSourceName?: string | null;
  registerCardRef: (name: string, el: HTMLElement | null) => void;
  onToggleFocus: (name: string) => void;
}

function SourceCard({
  name, brainState, focused, dimmed, flashSourceName, registerCardRef, onToggleFocus,
}: SourceCardProps) {
  const entities = brainState?.setEntityCount[name] ?? 0;
  const docs = brainState?.setDocCount[name] ?? 0;
  const members = brainState?.setMemberCount[name] ?? 0;
  return (
    <button
      ref={(el) => registerCardRef(name, el)}
      type="button"
      onClick={() => onToggleFocus(name)}
      className={`bv-motion relative w-full min-w-0 cursor-pointer overflow-hidden rounded-[10px] border p-2.5 pl-3.5 text-left text-xs transition-[transform,opacity,border-color] duration-500 hover:-translate-y-0.5 ${
        name === flashSourceName
          ? "border-[#43D9E8]"
          : focused ? "border-[#E9EEF6]" : "border-[#2A3652]"
      } bg-[#1A2438] ${dimmed ? "opacity-25" : ""}`}
    >
      <span
        className="absolute left-0 top-2 bottom-2 w-[3px] rounded"
        style={{ background: brainState?.setColor[name] || "#7E8CA6" }}
      />
      {/* A native `title` tooltip is slow to appear, inconsistent across
          browsers, and never fires on touch — the only way to see a
          truncated name reported as "not working" on staging. Mantine's
          Tooltip (already used for BrainSwitcher's Menu right above this
          rail) is instant, consistent, and has a visible affordance instead
          of relying on an invisible browser default. Its label is
          sourceTooltipLabel, not sourceLabel: labelling it with the same
          string the card already displays made the hover a no-op and left
          the raw source name unreachable. */}
      <Tooltip label={sourceTooltipLabel(name)} openDelay={200} withinPortal>
        <div className="truncate font-semibold text-[#E9EEF6]">{sourceLabel(name)}</div>
      </Tooltip>
      <div
        className="truncate text-[11px] text-[#7E8CA6]"
        title="entities = concepts extracted from this source; items = source documents/records ingested"
      >
        {entities
          ? `${entities} entities · ${docs} item${docs === 1 ? "" : "s"}`
          : members
            ? `${members} item${members === 1 ? "" : "s"}`
            : "weaving…"}
      </div>
    </button>
  );
}

interface SourceChipProps {
  name: string;
  count: number;
  color: string;
  focused: boolean;
  dimmed: boolean;
  onToggleFocus: (name: string) => void;
}

// The long-tail counterpart to SourceCard — color and count only, name on
// hover. Deliberately never calls registerCardRef: the filament layer
// (businessFilaments.ts) only threads a line to sources it has a card ref
// for, so a chip's source simply never gets one, not just a hidden one.
function SourceChip({ name, count, color, focused, dimmed, onToggleFocus }: SourceChipProps) {
  return (
    <Tooltip label={sourceTooltipLabel(name)} openDelay={200} withinPortal>
      <button
        type="button"
        onClick={() => onToggleFocus(name)}
        /* The chip shows a colour dot and, for a source with no entities yet,
           nothing else — Mantine's Tooltip renders into a portal and supplies
           no accessible name, so without this the control is announced as an
           unlabelled button. */
        aria-label={sourceLabel(name)}
        className={`bv-motion flex h-6 min-w-[24px] items-center justify-center gap-1 rounded-full border px-1.5 text-[10px] text-[#E9EEF6] transition-[transform,opacity,border-color] duration-500 hover:-translate-y-0.5 ${
          focused ? "border-[#E9EEF6]" : "border-[#2A3652]"
        } bg-[#1A2438] ${dimmed ? "opacity-25" : ""}`}
      >
        <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
        {count > 0 && count}
      </button>
    </Tooltip>
  );
}

// Ports the sources rail (customer_tutorial.html:5983-6001) — click a source
// to see only its memory (setFocusSets); click again for everything. This
// is what wires up the `focusSets` plumbing already threaded through
// BusinessCanvas/businessDraw (hulls, entity dimming) to an actual control.
export default function SourcesRail({
  brainState, focusSets, onToggleFocus, registerCardRef, flashSourceName, reachableByHoveredPrincipal = true,
}: SourcesRailProps) {
  const sourceNames = brainState?.sourceNames ?? [];
  if (!sourceNames.length) return null;

  const overflowing = sourceNames.length > TOP_SOURCE_CARD_LIMIT;
  const ranked = overflowing
    ? [...sourceNames].sort((a, b) => (brainState?.setEntityCount[b] ?? 0) - (brainState?.setEntityCount[a] ?? 0))
    : sourceNames;
  const topNames = overflowing ? ranked.slice(0, TOP_SOURCE_CARD_LIMIT) : ranked;
  const tailNames = overflowing ? ranked.slice(TOP_SOURCE_CARD_LIMIT) : [];
  // The flash target is chosen by NEWBORN count (BusinessView's handleGrowth)
  // while this list ranks by CUMULATIVE count — so a freshly connected source,
  // the one whose growth is most worth flashing, is exactly the one likeliest
  // to sit below the limit with no card to flash. Promote it for the ~2s the
  // flash lasts, appended rather than swapped in so nothing else reorders
  // (and no card loses the filament it had a frame ago).
  const promoteFlashed = flashSourceName != null && tailNames.includes(flashSourceName);
  const cardNames = promoteFlashed && flashSourceName ? [...topNames, flashSourceName] : topNames;
  const chipNames = promoteFlashed ? tailNames.filter((n) => n !== flashSourceName) : tailNames;

  return (
    <div className="p-2.5">
      <div className="px-1 text-[10px] uppercase tracking-widest text-[#7E8CA6]">sources</div>
      <div className="mt-2 flex flex-col gap-2">
        {cardNames.map((name) => (
          <SourceCard
            key={name}
            name={name}
            brainState={brainState}
            focused={focusSets?.has(name) ?? false}
            dimmed={(focusSets ? !focusSets.has(name) : false) || !reachableByHoveredPrincipal}
            flashSourceName={flashSourceName}
            registerCardRef={registerCardRef}
            onToggleFocus={onToggleFocus}
          />
        ))}
        {/* Hidden for now — re-enable once the connect-a-source flow ships. */}
      </div>
      {chipNames.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5 px-1">
          {chipNames.map((name) => (
            <SourceChip
              key={name}
              name={name}
              count={brainState?.setEntityCount[name] ?? 0}
              color={brainState?.setColor[name] || "#7E8CA6"}
              focused={focusSets?.has(name) ?? false}
              dimmed={(focusSets ? !focusSets.has(name) : false) || !reachableByHoveredPrincipal}
              onToggleFocus={onToggleFocus}
            />
          ))}
        </div>
      )}
    </div>
  );
}
