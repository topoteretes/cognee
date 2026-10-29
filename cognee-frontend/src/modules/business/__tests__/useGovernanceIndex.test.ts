import type { GovernanceIndex } from "../useGovernanceIndex";
import { narrowToReadable, userLabel } from "../useGovernanceIndex";
import type { BusinessGraphNode } from "../types";

// ── helpers ──────────────────────────────────────────────────────────────────

function user(id: string, name?: string): BusinessGraphNode {
  return { id, name, type: "User" };
}

function indexWithUsers(users: BusinessGraphNode[]): GovernanceIndex {
  return {
    tenants: [], users, agents: [], datasets: [], datasetById: {}, access: {}, agentOwnerId: {},
  };
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe("userLabel", () => {
  it("returns 'you' in single-user mode regardless of the user's actual name", () => {
    const index = indexWithUsers([user("u1", "goran@topoteretes.com")]);
    expect(userLabel(index, index.users[0])).toBe("you");
  });

  it("returns the full email address in multi-user mode", () => {
    const index = indexWithUsers([user("u1", "goran@topoteretes.com"), user("u2", "jane@topoteretes.com")]);
    expect(userLabel(index, index.users[0])).toBe("goran@topoteretes.com");
  });

  it("returns a real display name as-is when the backend sends one", () => {
    const index = indexWithUsers([user("u1", "Jane Doe"), user("u2", "jane@topoteretes.com")]);
    expect(userLabel(index, index.users[0])).toBe("Jane Doe");
  });

  it("falls back to a stable 'member N' label for a raw graph-node-id name", () => {
    // Some principals carry only a raw graph node id as their name (e.g.
    // "user:920fd9eb") — echoing that verbatim would leak an internal id.
    const index = indexWithUsers([user("u1", "user:920fd9eb"), user("u2", "jane@topoteretes.com")]);
    expect(userLabel(index, index.users[0])).toBe("member 1");
  });

  it("numbers the 'member N' fallback by position in the users list, not arrival order of the call", () => {
    const index = indexWithUsers([user("u1", "user:aaa"), user("u2", "user:bbb"), user("u3", "user:ccc")]);
    expect(userLabel(index, index.users[2])).toBe("member 3");
  });

  it("falls back to 'member N' when the user has no name at all", () => {
    const index = indexWithUsers([user("u1", undefined), user("u2", "jane@topoteretes.com")]);
    expect(userLabel(index, index.users[0])).toBe("member 1");
  });
});

// The provenance payload behind this index is tenant-scoped, so it carries
// datasets the caller holds no grant on. Every panel renders straight off the
// index, so this is the one place that decides what they may see.
describe("narrowToReadable", () => {
  function tenantIndex(): GovernanceIndex {
    const datasets: BusinessGraphNode[] = [
      { id: "ds-mine", name: "luca-onboarding", type: "Dataset" },
      { id: "ds-theirs", name: "acme-payroll", type: "Dataset" },
    ];
    return {
      tenants: [],
      users: [user("u1", "luca@acme.com"), user("u2", "mara@acme.com")],
      agents: [],
      datasets,
      datasetById: Object.fromEntries(datasets.map((d) => [d.id, d])),
      access: {
        u1: { "ds-mine": { owns: true, perms: new Set(["read"]) } },
        u2: { "ds-theirs": { owns: true, perms: new Set(["read"]) } },
      },
      agentOwnerId: {},
    };
  }

  it("drops a dataset the readable list does not vouch for", () => {
    const narrowed = narrowToReadable(tenantIndex(), ["ds-mine"]);
    expect(narrowed.datasets.map((d) => d.id)).toEqual(["ds-mine"]);
    expect(narrowed.datasetById["ds-theirs"]).toBeUndefined();
  });

  it("drops the access slots of a dropped dataset, so the rails cannot count it", () => {
    // "access · 1/2" would disclose that a second dataset exists even with
    // its name gone.
    const narrowed = narrowToReadable(tenantIndex(), ["ds-mine"]);
    expect(narrowed.access.u2).toBeUndefined();
    expect(Object.keys(narrowed.access.u1)).toEqual(["ds-mine"]);
  });

  it("narrows to nothing when the readable list is unknown", () => {
    // null is "the list has not answered, or failed". An unknown grant is not
    // a grant: falling back to the tenant's list is what leaked the names.
    const narrowed = narrowToReadable(tenantIndex(), null);
    expect(narrowed.datasets).toEqual([]);
    expect(narrowed.access).toEqual({});
  });

  it("keeps the workspace's principals, which are not what this narrows", () => {
    const narrowed = narrowToReadable(tenantIndex(), ["ds-mine"]);
    expect(narrowed.users.map((u) => u.id)).toEqual(["u1", "u2"]);
  });
});
