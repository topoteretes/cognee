import type { CogneeInstance } from "@/modules/instances/types";
import type { VisualizationPayload } from "./types";

// The "who may do what" side of Business: tenant/user/role membership, dataset
// ownership and ACL grants, projected as the same payload shape a content
// graph comes back in (see types.ts) — so the two layers merge with no
// adapter beyond what computeBrainState already does.
export default function getGovernanceGraph(
  instance: CogneeInstance,
): Promise<VisualizationPayload> {
  return instance
    .fetch("/v1/schema/provenance/json")
    .then((response) => response.json());
}
