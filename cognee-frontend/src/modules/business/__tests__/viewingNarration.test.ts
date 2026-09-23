import { computeViewingNarration } from "../viewingNarration";
import { computeBrainState } from "../computeBrainState";
import type { GovernanceIndex } from "../useGovernanceIndex";
import type { BusinessGraphNode } from "../types";
import type { BrainState } from "../sceneTypes";

// ── helpers ──────────────────────────────────────────────────────────────────

function entityNode(id: string, sets: string[]): BusinessGraphNode {
  return { id, name: id, stage: "entity", belongs_to_set: sets };
}

// The real thing the canvas renders, built the same way useBusinessScene
// builds it — so these counts are exactly what's on screen.
function brainStateOf(nodes: BusinessGraphNode[]): BrainState {
  return computeBrainState(nodes, []);
}

function indexWithDataset(datasetId: string, name: string, holderIds: string[]): GovernanceIndex {
  const users = holderIds.map((id) => ({ id, name: `${id}@topoteretes.com`, type: "User" }));
  const access: GovernanceIndex["access"] = {};
  holderIds.forEach((id, i) => {
    access[id] = { [datasetId]: { owns: i === 0, perms: new Set(["read"]) } };
  });
  return {
    tenants: [],
    users,
    agents: [],
    datasets: [{ id: datasetId, name, type: "Dataset" }],
    datasetById: { [datasetId]: { id: datasetId, name, type: "Dataset" } },
    access,
    agentOwnerId: {},
  };
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe("computeViewingNarration", () => {
  it("counts sources and entities from the rendered graph, not from a /brains snapshot", () => {
    // CLO-597 left /brains unpolled, so its per-dataset node lists go stale the
    // moment anything is ingested — the line has to agree with the canvas.
    const brainState = brainStateOf([
      entityNode("e1", ["slack"]),
      entityNode("e2", ["slack"]),
      entityNode("e3", ["docs"]),
    ]);

    const line = computeViewingNarration("ds-1", indexWithDataset("ds-1", "Sales", ["u1"]), brainState);

    expect(line).toContain("2 sources · 3 entities below");
  });

  it("singularizes a lone source", () => {
    const brainState = brainStateOf([entityNode("e1", ["slack"])]);

    const line = computeViewingNarration("ds-1", indexWithDataset("ds-1", "Sales", ["u1"]), brainState);

    expect(line).toContain("1 source · 1 entities below");
  });

  it("reports zero entities for a dataset whose content produced no graph yet", () => {
    const line = computeViewingNarration("ds-1", indexWithDataset("ds-1", "Sales", ["u1"]), brainStateOf([]));

    expect(line).toContain("0 sources · 0 entities below");
  });

  it("names a single holder's brain personal, and marks the owner", () => {
    const line = computeViewingNarration(
      "ds-1",
      indexWithDataset("ds-1", "Sales", ["u1"]),
      brainStateOf([entityNode("e1", ["slack"])]),
    );

    expect(line).toBe("Sales — personal brain shared by you (owner) · 1 source · 1 entities below");
  });

  it("names a multi-holder brain a team brain", () => {
    const line = computeViewingNarration(
      "ds-1",
      indexWithDataset("ds-1", "Sales", ["u1", "u2"]),
      brainStateOf([entityNode("e1", ["slack"])]),
    );

    expect(line).toContain("team brain shared by u1@topoteretes.com (owner), u2@topoteretes.com");
  });

  it("falls back to workspace wording and a default name when governance knows nothing about the dataset", () => {
    const emptyIndex: GovernanceIndex = {
      tenants: [], users: [], agents: [], datasets: [], datasetById: {}, access: {}, agentOwnerId: {},
    };

    const line = computeViewingNarration("ds-unknown", emptyIndex, brainStateOf([entityNode("e1", ["slack"])]));

    expect(line).toBe("brain — personal brain shared by this workspace · 1 source · 1 entities below");
  });
});
