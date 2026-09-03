import { agentSetupSteps, credentialsCommand, pluginInstallCommands, startCommand } from "../agentSetupSteps";
import { AGENT_CARDS } from "../agentCards";
import { getSteps } from "@/app/(app)/dashboard/partials/agentConnectionSteps";

const BASE_URL = "https://t1.aws.cognee.ai";
const API_KEY = "ck_abc";

describe("agentSetupSteps", () => {
  it.each(["claude-code", "codex"] as const)("gives %s the exact three actions onboarding uses", (agent) => {
    const steps = agentSetupSteps(agent, { os: "mac", baseUrl: BASE_URL, apiKey: API_KEY });

    expect(steps.map((s) => s.title)).toEqual([
      "Open Terminal",
      "Copy and run",
      `Start ${agent === "claude-code" ? "Claude Code" : "Codex"}`,
    ]);
    expect(steps[1].description).toBe("Saves your credentials and installs the Cognee plugin.");
  });

  it("puts credentials and the plugin install in ONE block, copied together", () => {
    const steps = agentSetupSteps("claude-code", { os: "mac", baseUrl: BASE_URL, apiKey: API_KEY });

    // Split blocks made it ambiguous whether they belonged in the terminal or
    // inside the agent — the whole reason they were combined.
    expect(steps[1].lines).toEqual([credentialsCommand("mac", BASE_URL, API_KEY), ...pluginInstallCommands("claude-code")]);
    expect(steps[1].copyLabel).toBe("Copy all");
  });

  it("drops the store-and-recall walkthrough the old wizards carried", () => {
    const titles = agentSetupSteps("claude-code", { os: "mac", baseUrl: BASE_URL, apiKey: API_KEY })
      .map((s) => s.title)
      .join(" ");

    expect(titles).not.toMatch(/Upload something|Recall it|You're all set|Save your credentials|Install the Cognee plugin/);
  });

  it("writes credentials to the shared env file rather than exporting them", () => {
    const cmd = credentialsCommand("mac", BASE_URL, API_KEY);

    expect(cmd).toContain("~/.cognee/.env");
    // The pre-CLO-532 form died with the terminal that ran it.
    expect(cmd).not.toMatch(/^export /);
  });
});

// The whole point of the shared module: these three surfaces used to hold
// independent copies of the same instructions, and had already drifted.
describe("connect instructions are identical across surfaces", () => {
  it.each(["claude-code", "codex"] as const)("%s: integrations wizard matches the shared steps", (agent) => {
    const card = AGENT_CARDS.find((c) => c.key === agent);
    const fromCard = card?.buildSteps(BASE_URL, API_KEY, false, "mac");

    expect(fromCard).toEqual(agentSetupSteps(agent, { os: "mac", baseUrl: BASE_URL, apiKey: API_KEY, loading: false }));
  });

  it.each(["claude-code", "codex"] as const)("%s: dashboard panel matches the shared steps", (agent) => {
    const fromDashboard = getSteps(agent, {
      baseUrl: BASE_URL,
      resolvedKey: API_KEY,
      isInitializing: false,
      os: "mac",
    });

    expect(fromDashboard).toEqual(agentSetupSteps(agent, { os: "mac", baseUrl: BASE_URL, apiKey: API_KEY, loading: false }));
  });

  it.each(["claude-code", "codex"] as const)("%s: the one combined block is credentials then install", (agent) => {
    const steps = agentSetupSteps(agent, { os: "mac", baseUrl: BASE_URL, apiKey: API_KEY });

    expect(steps[1].lines).toEqual([credentialsCommand("mac", BASE_URL, API_KEY), ...pluginInstallCommands(agent)]);
  });

  it("every surface launches the agent with the same command", () => {
    expect(startCommand("claude-code")).toBe("claude");
    expect(startCommand("codex")).toBe("codex");
  });
});

describe("os awareness", () => {
  it("marks exactly the shell-based connectors as os-aware", () => {
    const osAware = AGENT_CARDS.filter((c) => c.osAware).map((c) => c.key).sort();

    // MCP connectors configure themselves through JSON, so their steps read the
    // same either way and the toggle would be inert.
    expect(osAware).toEqual(["claude-code", "codex", "https-api", "openclaw"]);
  });

  it.each(["claude-code", "codex"] as const)("%s produces different commands per os", (agent) => {
    const mac = agentSetupSteps(agent, { os: "mac", baseUrl: BASE_URL, apiKey: API_KEY });
    const windows = agentSetupSteps(agent, { os: "windows", baseUrl: BASE_URL, apiKey: API_KEY });

    expect(mac[1].lines).not.toEqual(windows[1].lines);
    expect(windows[0].title).toBe("Open PowerShell");
  });
});

// Only the Claude Code and Codex plugins read ~/.cognee/.env. The other
// shell-based cards run a skill that curls with "X-Api-Key: $COGNEE_API_KEY",
// so writing the file instead of exporting would leave those users on a
// "verify these variables are available" check with nothing set.
describe("the env file is scoped to the plugin agents", () => {
  const shellCards = ["openclaw", "https-api"] as const;

  it.each(shellCards)("%s still exports the variables into the shell", (key) => {
    const card = AGENT_CARDS.find((c) => c.key === key);
    const first = card?.buildSteps(BASE_URL, API_KEY, false, "mac")[0];

    expect(first?.codeToCopy).toBe(`export COGNEE_BASE_URL="${BASE_URL}"\nexport COGNEE_API_KEY="${API_KEY}"`);
    expect(first?.codeToCopy).not.toContain(".cognee/.env");
  });

  it.each(shellCards)("%s exports through PowerShell syntax on Windows", (key) => {
    const card = AGENT_CARDS.find((c) => c.key === key);
    const first = card?.buildSteps(BASE_URL, API_KEY, false, "windows")[0];

    expect(first?.codeToCopy).toBe(`$env:COGNEE_BASE_URL = "${BASE_URL}"\n$env:COGNEE_API_KEY = "${API_KEY}"`);
  });

  it.each(["openclaw", "api-mcp"] as const)("%s exports on the dashboard panel too", (key) => {
    const first = getSteps(key, { baseUrl: BASE_URL, resolvedKey: API_KEY, isInitializing: false, os: "mac" })[0];

    expect(first.codeToCopy).toBe(`export COGNEE_BASE_URL="${BASE_URL}"\nexport COGNEE_API_KEY="${API_KEY}"`);
    expect(first.codeToCopy).not.toContain(".cognee/.env");
  });
});
