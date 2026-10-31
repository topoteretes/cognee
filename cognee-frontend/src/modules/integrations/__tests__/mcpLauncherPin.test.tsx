import { fireEvent, render } from "@testing-library/react";
import {
  CLAUDE_DESKTOP_MCP_ENTRY,
  COGNEE_MCP_SPEC,
  COGNEE_MCP_VERSION,
  HERMES_MCP_CONFIG,
  MCP_STDIO_CONFIG,
} from "@/data/prompts";
import { ConfigPreview, CursorConfigPreview, GeminiConfigPreview } from "@/modules/integrations/connectorHelpers";

// A "cognee-mcp" not followed by "@<digit>" is a launcher uvx resolves on its
// own: bare (the machine's cache decides the version) or "@latest" (PyPI
// decides). Either way an upstream release reaches Cloud users without a
// Cloud release.
const UNPINNED = /cognee-mcp(?!@\d)/;

const props = { baseUrl: "https://tenant-1.aws.cognee.ai", apiKey: "ck_test", loading: false };

describe("MCP launcher pin", () => {
  it("names one exact release", () => {
    expect(COGNEE_MCP_VERSION).toMatch(/^\d+\.\d+\.\d+$/);
    expect(COGNEE_MCP_SPEC).toBe(`cognee-mcp@${COGNEE_MCP_VERSION}`);
  });

  it.each([
    ["MCP_STDIO_CONFIG", MCP_STDIO_CONFIG],
    ["HERMES_MCP_CONFIG", HERMES_MCP_CONFIG],
    ["CLAUDE_DESKTOP_MCP_ENTRY", CLAUDE_DESKTOP_MCP_ENTRY],
  ])("%s launches the pinned release", (_name, template) => {
    expect(template).toContain(COGNEE_MCP_SPEC);
    expect(template).not.toMatch(UNPINNED);
  });

  it.each([
    ["Claude Desktop", ConfigPreview, "Copy the highlighted block"],
    ["Cursor", CursorConfigPreview, "Copy the cognee entry"],
    ["Gemini CLI", GeminiConfigPreview, "Copy the full config"],
  ])("the %s preview shows and copies the pinned release", (_name, Preview, copyLabel) => {
    const writeText = jest.fn();
    Object.assign(navigator, { clipboard: { writeText } });

    const { container, getByLabelText } = render(<Preview {...props} />);
    expect(container.textContent).toContain(COGNEE_MCP_SPEC);
    expect(container.textContent).not.toMatch(UNPINNED);

    fireEvent.click(getByLabelText(copyLabel));
    const copied = writeText.mock.calls[0][0];
    expect(copied).toContain(COGNEE_MCP_SPEC);
    expect(copied).not.toMatch(UNPINNED);
  });
});
