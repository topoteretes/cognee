import { deriveConnectedFromAgentConnections } from "../useAgentConnectionStatus";

describe("deriveConnectedFromAgentConnections", () => {
  it("marks an integration connected from an active registry connection alone", () => {
    // Regression: an agent registered via POST /agents/register is reported
    // `active` by GET /agents/connections before it has produced any session
    // row. The card must show Connected from this signal alone, not only once
    // a matching session_id prefix shows up.
    const result = deriveConnectedFromAgentConnections([
      { type: "codex", status: "active" },
    ]);

    expect(result).toEqual({ codex: true });
  });

  it("ignores connections that are not active", () => {
    const result = deriveConnectedFromAgentConnections([
      { type: "claude_code", status: "inactive" },
      { type: "codex", status: "unknown" },
    ]);

    expect(result).toEqual({});
  });

  it("ignores connection types with no matching integration card", () => {
    const result = deriveConnectedFromAgentConnections([
      { type: "slack", status: "active" },
      { type: "mcp", status: "active" },
    ]);

    expect(result).toEqual({});
  });

  it("maps every known connection type to its integration key", () => {
    const result = deriveConnectedFromAgentConnections([
      { type: "claude_code", status: "active" },
      { type: "codex", status: "active" },
    ]);

    expect(result).toEqual({ "claude-code": true, codex: true });
  });

  it("returns an empty object for an empty connection list", () => {
    expect(deriveConnectedFromAgentConnections([])).toEqual({});
  });
});
