import { agentDisplayName } from "../getAgents";
import { channelNames } from "@/modules/sessions/getSessions";

describe("agentDisplayName", () => {
  it("restores the case the endpoint's de-slugging drops", () => {
    // GET /v1/activity/agents builds agent_type as
    // display_name.replace("-", " ").replace("_", " ") over a lowercase slug.
    expect(agentDisplayName("claude code")).toBe("Claude Code");
    expect(agentDisplayName("some inhouse tool")).toBe("Some Inhouse Tool");
  });

  it("matches EVERY channel name the Via column derives from session prefixes", () => {
    // Derived from CHANNEL_DEFS rather than a hand-picked pair: the previous
    // version asserted only the two that title-casing happens to get right,
    // so it stayed green while Openclaw/Vs Code/Gemini Cli were all wrong.
    for (const name of channelNames()) {
      const asWire = name.toLowerCase();
      expect(agentDisplayName(asWire)).toBe(name);
    }
  });

  it("keeps the acronyms the Via column uses", () => {
    expect(agentDisplayName("openclaw")).toBe("OpenClaw");
    expect(agentDisplayName("vs code")).toBe("VS Code");
    expect(agentDisplayName("gemini cli")).toBe("Gemini CLI");
  });

  it("capitalises a leading non-ASCII letter instead of the one after it", () => {
    // `\b` is ASCII-only even under the `u` flag, so the old `\b\p{Ll}`
    // produced "éLodie.Dupont".
    expect(agentDisplayName("élodie.dupont")).toBe("Élodie.Dupont");
  });

  it("leaves an already-capitalised name alone", () => {
    expect(agentDisplayName("Human User")).toBe("Human User");
  });

  it("does not lowercase an interior capital", () => {
    expect(agentDisplayName("openAI agent")).toBe("OpenAI Agent");
  });

  it("handles an empty name", () => {
    expect(agentDisplayName("")).toBe("");
  });
});
