import type { BrainState } from "./sceneTypes";
import type { GovernanceIndex } from "./useGovernanceIndex";
import { userLabel } from "./useGovernanceIndex";

// focusKnowledge (customer_tutorial.html:6255-6267) branches three ways —
// "external, has read access", "external, no read access", and "primary"
// (the workspace's own brain, richer: names its sources and who shares it).
// Every dataset /visualize/brains returns is the caller's own workspace
// data — there's no cross-tenant "external" brain in this product to tell
// apart from a "primary" one — so rather than port a 3-way branch for two
// cases that can never happen here, this always uses the "primary" wording,
// the more informative of the three.
//
// The counts come from the rendered scene, never from /brains: that payload is
// fetched once and left unpolled (CLO-597), so its node lists go stale the
// moment anything is ingested — and it is a different dataset's snapshot
// entirely until the switch completes. brainState is what the canvas, the
// sources rail and the dock are showing, so the line agrees with them.
export function computeViewingNarration(
  datasetId: string,
  index: GovernanceIndex,
  brainState: BrainState,
): string {
  const dataset = index.datasetById[datasetId];
  const name = String(dataset?.name || "brain");
  const holders = index.users
    .filter((u) => (index.access[u.id] || {})[datasetId])
    .map((u) => userLabel(index, u) + (index.access[u.id]?.[datasetId]?.owns ? " (owner)" : ""));
  const kind = holders.length > 1 ? "team" : "personal";
  const who = holders.join(", ") || "this workspace";
  const sourceCount = brainState.sourceNames.length;
  const entityCount = brainState.entities.length;
  return `${name} — ${kind} brain shared by ${who} · ${sourceCount} source${sourceCount === 1 ? "" : "s"} · ${entityCount} entities below`;
}
