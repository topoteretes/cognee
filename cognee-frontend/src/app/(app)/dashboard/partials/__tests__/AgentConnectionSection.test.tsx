jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: jest.fn() }),
}));

// jsdom has no ResizeObserver, which Mantine's SegmentedControl (OsToggle)
// needs to position its FloatingIndicator.
class ResizeObserverStub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}
global.ResizeObserver = ResizeObserverStub;

import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen } from "@testing-library/react";
import { AgentConnectionSection } from "../AgentConnectionSection";
import { OsPreferenceProvider } from "@/ui/layout/OsPreferenceContext";

function renderSection(onOpenAgentSetup = jest.fn(), onUploadClick = jest.fn()) {
  render(
    <MantineProvider>
      <OsPreferenceProvider>
        <AgentConnectionSection
          onUploadClick={onUploadClick}
          onOpenAgentSetup={onOpenAgentSetup}
          isUploading={false}
          serviceUrl="https://t1.aws.cognee.ai"
          apiKey="ck_abc"
          isInitializing={false}
          hasDocuments={false}
        />
      </OsPreferenceProvider>
    </MantineProvider>,
  );
  return { onOpenAgentSetup, onUploadClick };
}

// CLO-989 review finding: a connected claude-code/codex card used to reopen
// this section's own install-steps popup, which baselines the existing
// registration as "not new" and has no connected-summary view — so a card
// that already read "Connected" opened to "Waiting for Claude Code to
// start…". Both plugin cards now route straight to AgentSetupModal instead.
describe("AgentConnectionSection plugin-agent routing", () => {
  it("opens AgentSetupModal for claude-code instead of the inline popup", () => {
    const { onOpenAgentSetup } = renderSection();

    fireEvent.click(screen.getByRole("button", { name: /Claude Code/ }));

    expect(onOpenAgentSetup).toHaveBeenCalledWith("claude-code");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("opens AgentSetupModal for codex instead of the inline popup", () => {
    const { onOpenAgentSetup } = renderSection();

    fireEvent.click(screen.getByRole("button", { name: /Codex/ }));

    expect(onOpenAgentSetup).toHaveBeenCalledWith("codex");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("still opens its own inline popup for openclaw", () => {
    const { onOpenAgentSetup } = renderSection();

    fireEvent.click(screen.getByRole("button", { name: /Openclaw/ }));

    expect(onOpenAgentSetup).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Connect Openclaw")).toBeInTheDocument();
  });

  it("still opens its own inline popup for API / MCP", () => {
    const { onOpenAgentSetup } = renderSection();

    fireEvent.click(screen.getByRole("button", { name: /API \/ MCP/ }));

    expect(onOpenAgentSetup).not.toHaveBeenCalled();
    expect(screen.getByText("Connect API / MCP")).toBeInTheDocument();
  });

  it("still routes the upload card to onUploadClick, not AgentSetupModal", () => {
    const { onOpenAgentSetup, onUploadClick } = renderSection();

    fireEvent.click(screen.getByRole("button", { name: /Company Dataset/ }));

    expect(onUploadClick).toHaveBeenCalledTimes(1);
    expect(onOpenAgentSetup).not.toHaveBeenCalled();
  });
});
