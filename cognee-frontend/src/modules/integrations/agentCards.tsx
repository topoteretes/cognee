import {
  OPENCLAW_PROMPT,
  CLAUDE_MARKETPLACE_ADD, CLAUDE_PLUGIN_INSTALL,
  CODEX_HOOKS_ENABLE, CODEX_MARKETPLACE_ADD, CODEX_PLUGIN_INSTALL,
  MCP_STDIO_CONFIG, HERMES_MCP_CONFIG, genericSkillInstall, fillTemplate,
  UPLOAD_MEMORY_PROMPT, UPLOAD_SAMPLE_PROMPT, RECALL_SAMPLE_PROMPT,
} from "@/data/prompts";
import type { SetupConnectorCfg } from "./types";
import {
  imgIcon, credStep, ApiIcon,
  installUvStep, InfoBox, ConfigPreview, CursorConfigPreview, GeminiConfigPreview,
} from "./connectorHelpers";

export const AGENT_CARDS: SetupConnectorCfg[] = [
  {
    key: "claude-code",
    name: "Claude Code",
    cta: "Connect via plugin",
    description: "Give Claude Code persistent memory across sessions.",
    icon: imgIcon("/visuals/logos/claude.svg", "Claude Code"),
    buildSteps: (baseUrl, apiKey, loading) => [
      credStep(baseUrl, apiKey, loading),
      {
        title: "Install the Cognee plugin",
        description: "Run these in your terminal one at a time — register the Cognee marketplace, then install the memory plugin.",
        codeBlocks: [
          { code: CLAUDE_MARKETPLACE_ADD },
          { code: CLAUDE_PLUGIN_INSTALL },
        ],
      },
      {
        title: "Upload something to Cognee",
        description: "Pick one and paste it into Claude Code — it stores the content in your Cognee memory so you can recall it next.",
        codeBlocks: [
          { label: "Option A · Your existing memory", code: UPLOAD_MEMORY_PROMPT },
          { label: "Option B · Try it with a sample", code: UPLOAD_SAMPLE_PROMPT },
        ],
      },
      {
        title: "Recall it from Cognee",
        description: "First run /exit to close the session — that syncs it into Cognee Cloud — then reopen Claude Code and ask the question below. Answering from a fresh session proves it's recalling from your cloud memory.",
        codeBlocks: [
          { code: "/exit" },
          { code: RECALL_SAMPLE_PROMPT },
        ],
      },
      {
        title: "You're all set",
        description: "The Cognee plugin hooks into Claude Code's lifecycle — no curl or manual API calls — and captures your session as you work. When a session ends (e.g. /exit), it consolidates that session into your Cognee Cloud knowledge graph, and every new session automatically recalls it back. Sessions are disposable; your memory isn't.",
      },
    ],
  },
  {
    key: "codex",
    name: "Codex",
    cta: "Connect via plugin",
    description: "Give Codex persistent memory across sessions.",
    icon: imgIcon("/visuals/logos/codex.svg", "Codex"),
    buildSteps: (baseUrl, apiKey, loading) => [
      credStep(baseUrl, apiKey, loading),
      {
        title: "Install the Cognee plugin",
        description: "Run these in your terminal one at a time — enable Codex hooks, register the Cognee marketplace, then install the memory plugin.",
        codeBlocks: [
          { code: CODEX_HOOKS_ENABLE },
          { code: CODEX_MARKETPLACE_ADD },
          { code: CODEX_PLUGIN_INSTALL },
        ],
      },
      {
        title: "Upload something to Cognee",
        description: "Pick one and paste it into Codex — it stores the content in your Cognee memory so you can recall it next.",
        codeBlocks: [
          { label: "Option A · Your existing memory", code: UPLOAD_MEMORY_PROMPT },
          { label: "Option B · Try it with a sample", code: UPLOAD_SAMPLE_PROMPT },
        ],
      },
      {
        title: "Recall it from Cognee",
        description: "First run /exit to close the session — that syncs it into Cognee Cloud — then reopen Codex and ask the question below. Answering from a fresh session proves it's recalling from your cloud memory.",
        codeBlocks: [
          { code: "/exit" },
          { code: RECALL_SAMPLE_PROMPT },
        ],
      },
      {
        title: "You're all set",
        description: "The Cognee plugin hooks into Codex's lifecycle — no curl or manual API calls — and captures your session as you work. When a session ends (e.g. /exit), it consolidates that session into your Cognee Cloud knowledge graph, and every new session automatically recalls it back. Sessions are disposable; your memory isn't.",
      },
    ],
  },
  {
    key: "openclaw",
    name: "OpenClaw",
    cta: "Connect via prompts",
    description: "Recall your Cognee memory in every OpenClaw conversation.",
    icon: imgIcon("/visuals/logos/openclaw.svg", "OpenClaw"),
    buildSteps: (baseUrl, apiKey, loading) => [
      credStep(baseUrl, apiKey, loading),
      // OpenClaw only loads AGENTS.md from its workspace directory, not the project root.
      { title: "Create the workspace AGENTS.md", description: "Run this command to add the Cognee memory instructions to OpenClaw's workspace. An existing AGENTS.md is backed up to AGENTS.md.bak — merge it manually afterwards.", code: "~/.openclaw/workspace/AGENTS.md", codeToCopy: `mkdir -p ~/.openclaw/workspace && [ -f ~/.openclaw/workspace/AGENTS.md ] && cp ~/.openclaw/workspace/AGENTS.md ~/.openclaw/workspace/AGENTS.md.bak; cat > ~/.openclaw/workspace/AGENTS.md << 'COGNEE_EOF'\n${OPENCLAW_PROMPT}\nCOGNEE_EOF` },
      { title: "Test the connection", description: `Open OpenClaw and ask: "What do you know from cognee?" — if it responds with knowledge from your brain, you're connected.` },
    ],
  },
  {
    key: "claude-desktop",
    name: "Claude Desktop",
    cta: "Connect via MCP",
    description: "Recall your Cognee memory in every conversation.",
    icon: imgIcon("/visuals/logos/claude.svg", "Claude Desktop"),
    buildSteps: (baseUrl, apiKey, loading) => [
      installUvStep(),
      {
        title: "Open the MCP config file",
        description: (
          <>
            In Claude Desktop, open <strong style={{ color: "#EDECEA" }}>Settings</strong> → <strong style={{ color: "#EDECEA" }}>Developer</strong> → <strong style={{ color: "#EDECEA" }}>Edit Config</strong> to create and open <strong style={{ color: "#EDECEA" }}>claude_desktop_config.json</strong>. Leave it open for the next step.
          </>
        ),
      },
      {
        title: "Add the Cognee server to the config",
        description: (
          <>
            Merge the highlighted block into your config — eyeball it against your open file.
            {"\n"}• No <strong style={{ color: "#EDECEA" }}>{'"mcpServers"'}</strong> yet → add the whole block at the top.
            {"\n"}• Already have one → add only the <strong style={{ color: "#EDECEA" }}>{'"cognee"'}</strong> entry inside it.
          </>
        ),
        content: <ConfigPreview baseUrl={baseUrl} apiKey={apiKey} loading={loading} />,
      },
      {
        title: "Restart and test",
        description: (
          <>
            Fully quit Claude Desktop — <strong style={{ color: "#EDECEA" }}>⌘Q</strong>, or quit from the tray on Windows (closing the window is not enough) — then reopen it.
            {"\n\n"}Nothing visibly changes in the UI, so confirm by using it: in a new chat, ask the prompt below. If Claude runs a <strong style={{ color: "#EDECEA" }}>cognee</strong> tool and answers from your memory, you are connected.
          </>
        ),
        code: "What do you know from cognee?",
        content: (
          <div style={{ marginTop: 12 }}>
            <InfoBox>
              If Claude never calls a <strong style={{ color: "#EDECEA" }}>cognee</strong> tool, it usually cannot find <strong style={{ color: "#EDECEA" }}>uvx</strong> on its PATH — run <strong style={{ color: "#EDECEA" }}>which uvx</strong> and use that absolute path as the <strong style={{ color: "#EDECEA" }}>command</strong> value in the config.{" "}<strong style={{ color: "#EDECEA" }}>(the uvx from step 1)</strong>
            </InfoBox>
          </div>
        ),
      },
    ],
  },
  {
    key: "cursor",
    name: "Cursor",
    cta: "Connect via MCP",
    description: "Ground Cursor's agent in your Cognee memory.",
    icon: imgIcon("/visuals/logos/cursor.svg", "Cursor"),
    buildSteps: (baseUrl, apiKey, loading) => [
      installUvStep(),
      {
        title: "Open Cursor's MCP config",
        description: (
          <>
            Open <strong style={{ color: "#EDECEA" }}>Cursor Settings</strong> → <strong style={{ color: "#EDECEA" }}>Tools &amp; MCPs</strong> (newer builds are moving this under <strong style={{ color: "#EDECEA" }}>Customize</strong> — follow the banner if you see it). Under <strong style={{ color: "#EDECEA" }}>Home MCP Servers</strong>, click <strong style={{ color: "#EDECEA" }}>Add Custom MCP</strong> — Cursor creates and opens <strong style={{ color: "#EDECEA" }}>~/.cursor/mcp.json</strong>.
            {"\n\n"}Prefer the terminal? Just open the file directly — the next step shows exactly what goes in it.
          </>
        ),
      },
      {
        title: "Add the Cognee server to mcp.json",
        description: (
          <>
            Paste the highlighted <strong style={{ color: "#EDECEA" }}>{'"cognee"'}</strong> entry inside the <strong style={{ color: "#EDECEA" }}>{'"mcpServers"'}</strong> block, then save. The copy button grabs just that entry — the dimmed braces show where it lands.
            {"\n"}• Cursor scaffolded the wrapper for you → drop it in alongside any existing servers.
            {"\n"}• Completely empty file → add the dimmed <strong style={{ color: "#EDECEA" }}>{'"mcpServers"'}</strong> braces around it too.
          </>
        ),
        content: <CursorConfigPreview baseUrl={baseUrl} apiKey={apiKey} loading={loading} />,
      },
      {
        title: "Enable it and test",
        description: (
          <>
            Back in <strong style={{ color: "#EDECEA" }}>Tools &amp; MCPs</strong> → <strong style={{ color: "#EDECEA" }}>Home MCP Servers</strong>, <strong style={{ color: "#EDECEA" }}>cognee</strong> now appears with a status dot — make sure its toggle is on and wait for the dot to turn <strong style={{ color: "#EDECEA" }}>green</strong> (hit the refresh icon if it doesn’t). Cursor may ask you to trust the server; accept it.
            {"\n\n"}Then open a chat, switch it to <strong style={{ color: "#EDECEA" }}>Agent</strong> mode (MCP tools only run there), and send the prompt below. If Cursor calls a <strong style={{ color: "#EDECEA" }}>cognee</strong> tool and answers from your memory, you’re connected.
          </>
        ),
        code: "What do you know from cognee?",
        content: (
          <div style={{ marginTop: 12 }}>
            <InfoBox>
              Red dot or “no tools available”? Cursor usually can’t find <strong style={{ color: "#EDECEA" }}>uvx</strong> on its PATH — run <strong style={{ color: "#EDECEA" }}>which uvx</strong> and use that absolute path as the <strong style={{ color: "#EDECEA" }}>command</strong> value in mcp.json <strong style={{ color: "#EDECEA" }}>(the uvx from step 1)</strong>.
            </InfoBox>
          </div>
        ),
      },
    ],
  },
  {
    key: "hermes",
    name: "Hermes Agent",
    cta: "Connect via MCP",
    description: "Ground Hermes Agent in your Cognee memory.",
    icon: imgIcon("/visuals/logos/hermes.svg", "Hermes Agent"),
    buildSteps: (baseUrl, apiKey, loading) => [
      { title: "Configure Hermes Agent", description: "Hermes reads YAML — add this block under mcp_servers in ~/.hermes/config.yaml and restart the agent. Cognee runs via uvx (requires uv) — no separate install.", code: "~/.hermes/config.yaml", codeToCopy: fillTemplate(HERMES_MCP_CONFIG, baseUrl, apiKey), loading },
      { title: "Test the connection", description: "Ask Hermes: \"What do you know from cognee?\" — it should use the Cognee memory tool and return a response from your brain." },
    ],
  },
  {
    key: "vscode",
    name: "VS Code",
    cta: "Connect via extension",
    description: "Give VS Code persistent, citable project memory.",
    icon: imgIcon("/visuals/logos/vscode.svg", "VS Code"),
    buildSteps: (baseUrl, apiKey, loading) => [
      {
        title: "Install the Cognee extension",
        description: (
          <>
            Open the <strong style={{ color: "#EDECEA" }}>Extensions</strong> view (<strong style={{ color: "#EDECEA" }}>⇧⌘X</strong> / <strong style={{ color: "#EDECEA" }}>Ctrl+Shift+X</strong>), search for <strong style={{ color: "#EDECEA" }}>Cognee</strong>, and click <strong style={{ color: "#EDECEA" }}>Install</strong> on <strong style={{ color: "#EDECEA" }}>“Cognee — Project Memory”</strong> by Cognee.
          </>
        ),
      },
      {
        title: "Run Cognee: Set Up",
        description: (
          <>
            Open the <strong style={{ color: "#EDECEA" }}>Command Palette</strong> (<strong style={{ color: "#EDECEA" }}>⇧⌘P</strong> / <strong style={{ color: "#EDECEA" }}>Ctrl+Shift+P</strong>) and run <strong style={{ color: "#EDECEA" }}>Cognee: Set Up</strong>. Paste your <strong style={{ color: "#EDECEA" }}>endpoint</strong>, then your <strong style={{ color: "#EDECEA" }}>API key</strong> — stored in your OS keychain, not settings. A health check confirms the connection.
          </>
        ),
        codeBlocks: [
          { label: "Endpoint", code: baseUrl, loading },
          { label: "API key", code: apiKey, loading },
        ],
      },
      {
        title: "Use the core commands",
        description: (
          <>
            Everything lives in the <strong style={{ color: "#EDECEA" }}>Command Palette</strong> under <strong style={{ color: "#EDECEA" }}>Cognee</strong>. Start with these:
            {"\n\n"}• <strong style={{ color: "#EDECEA" }}>Cognee: Remember Selection</strong> — store selected code (or the whole file) in this repo&apos;s memory.
            {"\n"}• <strong style={{ color: "#EDECEA" }}>Cognee: Ask My Project Memory</strong> — ask a question; answers come with clickable citations.
            {"\n"}• <strong style={{ color: "#EDECEA" }}>Cognee: Index Workspace</strong> — bulk-ingest the whole repo at once.
            {"\n\n"}
            <a href="https://docs.cognee.ai/cognee-cloud/agent-integrations/vscode" target="_blank" rel="noopener noreferrer" style={{ color: "#6510F4", textDecoration: "underline" }}>See all commands and settings →</a>
          </>
        ),
      },
    ],
  },
  {
    key: "gemini-cli",
    name: "Gemini CLI",
    cta: "Connect via MCP",
    description: "Ground Gemini CLI in your Cognee memory.",
    icon: imgIcon("/visuals/logos/gemini.svg", "Gemini CLI"),
    buildSteps: (baseUrl, apiKey, loading) => [
      installUvStep(),
      {
        title: "Open your Gemini config file",
        description: (
          <>
            Gemini CLI keeps its settings in <strong style={{ color: "#EDECEA" }}>~/.gemini/settings.json</strong>. Open that file in your editor.
            {"\n\n"}If the file or the <strong style={{ color: "#EDECEA" }}>~/.gemini</strong> folder does not exist yet, run <strong style={{ color: "#EDECEA" }}>gemini</strong> once (it creates them), or create the file yourself. Leave it open for the next step.
          </>
        ),
        code: "~/.gemini/settings.json",
      },
      {
        title: "Add the Cognee server and save",
        description: (
          <>
            Make your file look like the block below, then <strong style={{ color: "#EDECEA" }}>save</strong>. Everything is shown in full — your tenant URL and key included — so you can see exactly what you are adding.
            {"\n"}• <strong style={{ color: "#EDECEA" }}>New or empty file</strong> → paste the whole block (the copy button copies all of it).
            {"\n"}• <strong style={{ color: "#EDECEA" }}>Already have {'"mcpServers"'}</strong> → add just the highlighted <strong style={{ color: "#EDECEA" }}>{'"cognee"'}</strong> entry inside it.
          </>
        ),
        content: <GeminiConfigPreview baseUrl={baseUrl} apiKey={apiKey} loading={loading} />,
      },
      {
        title: "Open Gemini and confirm it connected",
        description: (
          <>
            Start Gemini in your terminal with <strong style={{ color: "#EDECEA" }}>gemini</strong>. Inside the session, type <strong style={{ color: "#EDECEA" }}>/mcp</strong> — you should see <strong style={{ color: "#EDECEA" }}>cognee</strong> listed with its tools. That is your confirmation it is connected.
            {"\n\n"}Then ask the prompt below. If Gemini uses a <strong style={{ color: "#EDECEA" }}>cognee</strong> tool and answers from your memory, you are all set.
          </>
        ),
        codeBlocks: [
          { label: "1 · Start Gemini (in your terminal)", code: "gemini" },
          { label: "2 · Confirm cognee is listed", code: "/mcp" },
          { label: "3 · Ask", code: "What do you know from cognee?" },
        ],
        content: (
          <div style={{ marginTop: 12 }}>
            <InfoBox>
              If <strong style={{ color: "#EDECEA" }}>/mcp</strong> does not list cognee, Gemini usually cannot find <strong style={{ color: "#EDECEA" }}>uvx</strong> on its PATH. Run <strong style={{ color: "#EDECEA" }}>which uvx</strong> and use that absolute path as the <strong style={{ color: "#EDECEA" }}>{'"command"'}</strong> value in settings.json <strong style={{ color: "#EDECEA" }}>(the uvx from step 1)</strong>.
            </InfoBox>
          </div>
        ),
      },
    ],
  },
  {
    key: "cline",
    name: "Cline",
    cta: "Connect via MCP",
    description: "Ground Cline in your Cognee memory.",
    icon: imgIcon("/visuals/logos/cline.svg", "Cline"),
    buildSteps: (baseUrl, apiKey, loading) => [
      { title: "Configure Cline", description: "In VS Code, open the Cline sidebar → MCP Servers → Configure MCP Servers — paste this JSON block and save. Cognee runs via uvx (requires uv) — no separate install.", code: '{ "mcpServers": { "cognee": … } }', codeToCopy: fillTemplate(MCP_STDIO_CONFIG, baseUrl, apiKey), loading },
      { title: "Test the connection", description: "Ask Cline: \"What do you know from cognee?\" — it should use the Cognee memory tool and return a response from your brain." },
    ],
  },
  {
    key: "https-api",
    name: "API / MCP",
    cta: "Connect via API or MCP",
    description: "Call Cognee directly from any HTTP client or custom agent.",
    icon: <ApiIcon />,
    buildSteps: (baseUrl, apiKey, loading) => [
      credStep(baseUrl, apiKey, loading),
      {
        title: "Query the REST API",
        description: "Send a recall query to your Cognee endpoint from any HTTP client or language.",
        code: `curl -X POST ${baseUrl}/api/v1/recall`,
        codeToCopy: fillTemplate(
          'curl -X POST {{BASE_URL}}/api/v1/recall \\\n  -H "X-Api-Key: {{API_KEY}}" \\\n  -H "Content-Type: application/json" \\\n  -d \'{"query": "What are the main entities?"}\'',
          baseUrl, apiKey,
        ),
        loading,
      },
      {
        title: "Or install the Cognee skill",
        description: "Prefer skills? Run this command from your project root to create the skill file, then point your agent at it (skills directory, instructions file, or system prompt). The skill teaches your agent to call the Cognee API using the credentials from step 1.",
        code: "skills/cognee/SKILL.md",
        codeToCopy: genericSkillInstall("mac"),
      },
      { title: "Test the connection", description: "Ask your agent: \"What do you know from cognee?\" — Cognee's memory tool should respond with knowledge from your brain." },
    ],
  },
];
