"use client";

import { useEffect } from "react";
import type { GovernanceIndex } from "../useGovernanceIndex";
import { accessibleDatasetIds, userLabel } from "../useGovernanceIndex";
import AccessChipList from "./AccessChipList";

interface OperatorsRailProps {
  index: GovernanceIndex;
  onHoverPrincipal: (principalId: string | null) => void;
  // The dataset currently being viewed — ports focusKnowledge's "light up
  // who this brain applies to; dim everyone else" (customer_tutorial.html
  // ~6255), the mirror image of KnowledgeRail's hover-driven dim.
  focusedDatasetId: string | null;
  // Ports .bv-org-node.asking — the agent a live search event is currently
  // attributed to gets a 9s amber glow (customer_tutorial.html ~7015).
  askingPrincipalId: string | null;
  // Session memory is the USER's own conversation history, not a property
  // of any one agent (see SessionMemoryCard) — this opens it from the user
  // card.
  onOpenSessionMemory: (principalId: string, principalName: string) => void;
}

function initials(name: string): string {
  return name.replace(/@.*/, "").split(/[._\- ]/).map((w) => w[0] || "").join("").slice(0, 2).toUpperCase();
}

const memBadgeBase = "rounded px-1.5 py-0.5";

// Ports the Operators tree (customer_tutorial.html:6003-6172): tenant → users
// → their agents, each with access spelled out. Hovering used to dim/light
// DOM rows directly (wireUserHover); here it just reports the hovered
// principal up, and the knowledge rail decides its own highlight state —
// same "access is a highlight on hover, never a drawn edge" rule, React-native.
export default function OperatorsRail({
  index, onHoverPrincipal, focusedDatasetId, askingPrincipalId, onOpenSessionMemory,
}: OperatorsRailProps) {
  // Leaving a hovered row (unmount mid-hover: a dataset switch, a governance
  // refetch) fires no mouseleave, so the last reported principal stuck — and
  // with it the sources rail's ACL dim, permanently, with nothing on screen
  // still hovered to explain it.
  useEffect(() => () => onHoverPrincipal(null), [onHoverPrincipal]);

  // The tenant card names the organization and counts its members: with one
  // member it says nothing the user card below it doesn't. This is the only
  // thing "how many users are there" should ever gate — the previous
  // version used it to hide the whole rail and every user's access block
  // too, which took the sole entry point to SessionMemoryCard (the user
  // card's own click) out of the app entirely for a solo workspace.
  const showTenantCard = index.users.length > 1;
  // Access chips are noise only when they all say the same thing. A
  // principal who owns every dataset gets a row of identical "owner" chips
  // (confirmed live: an owner of 10/11 datasets produced a chip block tall
  // enough to collide with the cards below it) — but one who does NOT own
  // everything has the most informative row on the panel. Keying on the
  // principal's own access, rather than on how many users exist, is also
  // what makes user rows and agent rows follow one rule: agent rows never
  // had the user-count guard, so a solo user's card showed no access at all
  // beside their own agent's full detail.
  const ownsEverything = (principalId: string): boolean =>
    index.datasets.length > 0
    && index.datasets.every((d) => (index.access[principalId] || {})[d.id]?.owns);

  const agentsByOwner: Record<string, typeof index.agents> = {};
  index.agents.forEach((a) => {
    const ownerId = index.agentOwnerId[a.id] || index.users[0]?.id;
    if (!ownerId) return;
    (agentsByOwner[ownerId] = agentsByOwner[ownerId] || []).push(a);
  });
  const holdsFocused = (principalId: string): boolean =>
    !focusedDatasetId || Boolean((index.access[principalId] || {})[focusedDatasetId]);
  // Only worth a glance when access is actually restricted — an owner/
  // full-access principal's chip row already reads as "everything", so a
  // "7/7" badge next to every single card would be noise, not signal.
  const restrictedReach = (principalId: string): string | null => {
    const reach = accessibleDatasetIds(index, principalId).size;
    return reach < index.datasets.length ? `${reach}/${index.datasets.length} datasets` : null;
  };

  return (
    // top-0 + own p-2.5 padding puts the "operators" label at 10px from the
    // top, level with BusinessView's "brain" label; leading-[15px] and the
    // workspace card's mt-1 land the card's top edge at 10+15+4 = 29px, the
    // same line as the BrainSwitcher chip and the search bar.
    <div className="absolute right-0 top-0 bottom-24 w-[196px] overflow-y-auto p-2.5 text-xs text-[#E9EEF6]">
      <div className="px-1 text-[10px] leading-[15px] uppercase tracking-widest text-[#7E8CA6]">operators</div>
      {showTenantCard && index.tenants[0] && (
        <div className="mt-1 rounded-[10px] border border-dashed border-[#2A3652] p-2.5 text-[#7E8CA6]">
          <div>⌂ {String(index.tenants[0].name || "organization")}</div>
          <div className="text-[10.5px]">{index.users.length} member{index.users.length === 1 ? "" : "s"}</div>
        </div>
      )}
      {index.users.map((u) => {
        const uLabel = userLabel(index, u);
        return (
          <div key={u.id} className="ml-4 mt-2">
            <div
              className={`cursor-pointer rounded-[10px] border border-[#2A3652] bg-[#1A2438] p-2.5 transition-opacity ${holdsFocused(u.id) ? "" : "opacity-25"}`}
              onMouseEnter={() => onHoverPrincipal(u.id)}
              onMouseLeave={() => onHoverPrincipal(null)}
              onClick={() => onOpenSessionMemory(u.id, uLabel)}
            >
              <div className="flex min-w-0 items-center gap-1.5 font-semibold">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-[#2A3652] bg-[#141D33] text-[9px]">
                  {initials(String(u.name || "?"))}
                </span>
                <span className="min-w-0 truncate" title={uLabel}>{uLabel}</span>
                <span
                  className={`${memBadgeBase} ml-auto shrink-0 border border-[rgba(245,168,60,.45)] text-[#F5A83C]`}
                  title="session memory — this user's own conversation history; distilled traces land as session_learnings"
                >
                  session
                </span>
              </div>
              {!ownsEverything(u.id) && restrictedReach(u.id) && (
                <div className="ml-[26px] mt-0.5 text-[10.5px] text-[#7E8CA6]">{restrictedReach(u.id)}</div>
              )}
              {!ownsEverything(u.id) && (
                <AccessChipList
                  access={index.access[u.id] || {}}
                  datasets={index.datasets}
                  principalName={uLabel}
                  focusedDatasetId={focusedDatasetId}
                />
              )}
            </div>
            {(agentsByOwner[u.id] || []).map((a) => {
              const aLabel = String(a.name || "agent");
              return (
                <div
                  key={a.id}
                  className={`ml-4 mt-1 rounded-[10px] border p-2 text-[9.5px] transition-opacity ${
                    a.id === askingPrincipalId
                      ? "border-[#F5A83C] shadow-[0_0_12px_rgba(245,168,60,0.25)]"
                      : "border-[#2A3652]"
                  } bg-[#1A2438] ${holdsFocused(a.id) ? "" : "opacity-25"}`}
                  onMouseEnter={() => onHoverPrincipal(a.id)}
                  onMouseLeave={() => onHoverPrincipal(null)}
                >
                  <div className="flex min-w-0 items-center gap-1">
                    <span className="inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-[#F5A83C]" />
                    <span className="min-w-0 truncate" title={aLabel}>{aLabel}</span>
                    <span className="ml-auto shrink-0 rounded bg-[#F5A83C] px-1 py-0.5 text-[8px] font-bold uppercase text-[#0E1526]">
                      agent
                    </span>
                  </div>
                  <div className="mt-1 flex flex-wrap items-center gap-1 text-[#7E8CA6]">
                    memory:
                    <span
                      className={`${memBadgeBase} border border-solid border-[#E9EEF6] text-[#E9EEF6]`}
                      title="permanent memory — searches its brains (with_memory)"
                    >
                      permanent
                    </span>
                    {restrictedReach(a.id) && <span className="ml-auto shrink-0">{restrictedReach(a.id)}</span>}
                  </div>
                  {!ownsEverything(a.id) && (
                    <AccessChipList
                      access={index.access[a.id] || {}}
                      datasets={index.datasets}
                      principalName={`${aLabel} (agent)`}
                      focusedDatasetId={focusedDatasetId}
                    />
                  )}
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
