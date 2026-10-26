import { getConnectedIntegrations, setConnectedIntegrations } from "@/utils/browserStorage";

const KEY = "cognee-connected-integrations-t1";

beforeEach(() => {
  localStorage.clear();
});

describe("getConnectedIntegrations", () => {
  it("reads back what was persisted", () => {
    setConnectedIntegrations("t1", { "claude-code": true });

    expect(getConnectedIntegrations("t1")).toEqual({ "claude-code": true });
  });

  it("records nothing for a tenant that has never connected anything", () => {
    expect(getConnectedIntegrations("t1")).toEqual({});
  });

  it("drops a stored value that is not an object of booleans", () => {
    // This record is now the memory graph's registration source, so a value
    // left by an older shape must read as "nothing recorded" rather than
    // arriving in the graph as truthy. It was cast, not checked.
    localStorage.setItem(KEY, JSON.stringify(["claude-code"]));
    expect(getConnectedIntegrations("t1")).toEqual({});

    localStorage.setItem(KEY, JSON.stringify("claude-code"));
    expect(getConnectedIntegrations("t1")).toEqual({});

    localStorage.setItem(KEY, JSON.stringify(null));
    expect(getConnectedIntegrations("t1")).toEqual({});
  });

  it("drops the entries that are not true and keeps the ones that are", () => {
    localStorage.setItem(KEY, JSON.stringify({ "claude-code": "yes", codex: true, cursor: false }));

    expect(getConnectedIntegrations("t1")).toEqual({ codex: true });
  });

  it("survives a value that is not JSON at all", () => {
    localStorage.setItem(KEY, "{not json");

    expect(getConnectedIntegrations("t1")).toEqual({});
  });
});
