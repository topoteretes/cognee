import { render, screen } from "@testing-library/react";
import type { ReleaseLog } from "@/modules/releases/releaseLog";

const mockUseReleases = jest.fn();
jest.mock("@/modules/analytics", () => ({ TrackPageView: () => null }));
jest.mock("@/modules/releases/useReleases", () => ({ useReleases: () => mockUseReleases() }));

import ReleasesPage from "../ReleasesPage";

const LOG: ReleaseLog = {
  runningVersion: "1.6.2-b45dc59",
  available: true,
  stale: false,
  releases: [
    {
      version: "v1.6.2",
      releasedAt: "2026-10-01T13:52:19Z",
      summary: "Integrations can target a dataset you choose.",
      highlights: ["Pick the dataset a connection writes to."],
      fixes: ["Readiness probes no longer lie."],
      hasNotes: true,
    },
    { version: "v1.6.1", releasedAt: "2026-09-29T14:20:28Z", summary: "", highlights: [], fixes: [], hasNotes: false },
  ],
};

function renderPage(state: { data?: ReleaseLog; isLoading?: boolean; isError?: boolean }): void {
  mockUseReleases.mockReturnValue({ data: state.data, isLoading: state.isLoading ?? false, isError: state.isError ?? false });
  render(<ReleasesPage />);
}

describe("ReleasesPage", () => {
  it("lists every release with its version, date and notes", () => {
    renderPage({ data: LOG });
    expect(screen.getByText("This workspace runs Cloud v1.6.2.")).toBeInTheDocument();
    expect(screen.getByText("v1.6.2")).toBeInTheDocument();
    expect(screen.getByText("Oct 1, 2026")).toBeInTheDocument();
    expect(screen.getByText("Integrations can target a dataset you choose.")).toBeInTheDocument();
    expect(screen.getByText("Highlights")).toBeInTheDocument();
    expect(screen.getByText("Pick the dataset a connection writes to.")).toBeInTheDocument();
    expect(screen.getByText("Fixes")).toBeInTheDocument();
    expect(screen.getByText("Readiness probes no longer lie.")).toBeInTheDocument();
    expect(screen.getByText("v1.6.1")).toBeInTheDocument();
    expect(screen.getByText("No release notes were recorded for this version.")).toBeInTheDocument();
    expect(screen.queryByText(/last copy/)).not.toBeInTheDocument();
  });

  it("says when the list is the last copy that could be loaded", () => {
    renderPage({ data: { ...LOG, stale: true } });
    expect(screen.getByText(/last copy that could be loaded/)).toBeInTheDocument();
  });

  it("explains an unavailable, empty, loading or failed log instead of rendering nothing", () => {
    renderPage({ data: { runningVersion: null, available: false, stale: false, releases: [] } });
    expect(screen.getByText("Release notes are not available in this environment.")).toBeInTheDocument();
    expect(screen.getByText("What changed in Cloud, release by release.")).toBeInTheDocument();

    renderPage({ data: { ...LOG, releases: [] } });
    expect(screen.getByText("No releases recorded yet.")).toBeInTheDocument();

    renderPage({ isLoading: true });
    expect(screen.getByText("Loading release notes…")).toBeInTheDocument();

    renderPage({ isError: true });
    expect(screen.getByText("Release notes could not be loaded. Try again in a moment.")).toBeInTheDocument();
  });
});
