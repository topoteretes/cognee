import { fireEvent, render, screen } from "@testing-library/react";
import type { ReleaseLog } from "@/modules/releases/releaseLog";

const mockUseReleases = jest.fn();
jest.mock("next/navigation", () => ({ useRouter: () => ({ push: jest.fn() }) }));
jest.mock("../ExtractionSettingsModal", () => ({ __esModule: true, default: () => null }));
jest.mock("@/modules/releases/useReleases", () => ({ useReleases: () => mockUseReleases() }));

import HelpMenu from "../HelpMenu";

const LOG: ReleaseLog = {
  runningVersion: "1.6.2-b45dc59",
  available: true,
  stale: false,
  releases: [
    {
      version: "v1.6.2",
      releasedAt: "2026-10-01T13:52:19Z",
      summary: "Integrations can target a dataset you choose.",
      highlights: ["Pick the dataset a connection writes to.", "Linear joins the integrations.", "Agents keep their sessions.", "A fourth highlight."],
      fixes: ["Readiness probes no longer lie."],
      hasNotes: true,
    },
    { version: "v1.6.1", releasedAt: "2026-09-29T14:20:28Z", summary: "", highlights: [], fixes: [], hasNotes: false },
  ],
};

function openMenu(log: ReleaseLog | undefined): void {
  mockUseReleases.mockReturnValue({ data: log, isLoading: log === undefined, isError: false });
  render(<HelpMenu />);
  fireEvent.click(screen.getByText("?"));
}

describe("HelpMenu — What's new", () => {
  it("shows the newest release's notes, three lines at most, with its version and date", () => {
    openMenu(LOG);
    expect(screen.getByText("What's new")).toBeInTheDocument();
    expect(screen.getByText("v1.6.2 · Oct 1, 2026")).toBeInTheDocument();
    expect(screen.getByText("Pick the dataset a connection writes to.")).toBeInTheDocument();
    expect(screen.getByText("Linear joins the integrations.")).toBeInTheDocument();
    expect(screen.getByText("Agents keep their sessions.")).toBeInTheDocument();
    expect(screen.queryByText("A fourth highlight.")).not.toBeInTheDocument();
    expect(screen.queryByText("Readiness probes no longer lie.")).not.toBeInTheDocument();
  });

  it("links to the full release log and names the running version", () => {
    openMenu(LOG);
    expect(screen.getByText("All release notes →").closest("a")).toHaveAttribute("href", "/releases");
    expect(screen.getByText("Cloud v1.6.2")).toBeInTheDocument();
  });

  it("skips a newer release that has no notes", () => {
    openMenu({
      ...LOG,
      releases: [{ version: "v1.6.3", releasedAt: "2026-10-08T00:00:00Z", summary: "", highlights: [], fixes: [], hasNotes: false }, ...LOG.releases],
    });
    expect(screen.getByText("v1.6.2 · Oct 1, 2026")).toBeInTheDocument();
  });

  it("falls back to fixes when a release has fewer than three highlights", () => {
    openMenu({ ...LOG, releases: [{ ...LOG.releases[0], highlights: ["Only one highlight."] }] });
    expect(screen.getByText("Only one highlight.")).toBeInTheDocument();
    expect(screen.getByText("Readiness probes no longer lie.")).toBeInTheDocument();
  });

  it("still offers the page when no release has notes yet", () => {
    openMenu({ ...LOG, releases: [LOG.releases[1]] });
    expect(screen.getByText("No release notes yet")).toBeInTheDocument();
    expect(screen.getByText("All release notes →")).toBeInTheDocument();
  });

  it("keeps the link but shows no notes or version when the backend has no release log", () => {
    openMenu({ runningVersion: null, available: false, stale: false, releases: [] });
    expect(screen.getByText("What's new")).toBeInTheDocument();
    expect(screen.getByText("All release notes →").closest("a")).toHaveAttribute("href", "/releases");
    expect(screen.queryByText("No release notes yet")).not.toBeInTheDocument();
    expect(screen.queryByText(/^Cloud v/)).not.toBeInTheDocument();
    expect(screen.getByText("Docs")).toBeInTheDocument();
  });

  it("keeps the link while the log is still loading, with nothing hand-written in its place", () => {
    openMenu(undefined);
    expect(screen.getByText("What's new")).toBeInTheDocument();
    expect(screen.getByText("All release notes →")).toBeInTheDocument();
    expect(screen.queryByText("No release notes yet")).not.toBeInTheDocument();
    expect(screen.queryByText(/Oct 1, 2026/)).not.toBeInTheDocument();
    expect(screen.getByText("Keyboard shortcuts")).toBeInTheDocument();
  });
});
