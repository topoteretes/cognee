import type { GovernanceIndex } from "../useGovernanceIndex";
import { userLabel } from "../useGovernanceIndex";
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
