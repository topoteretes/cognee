import { act, render, screen } from "@testing-library/react";
import WorkspaceWaking from "../WorkspaceWaking";

// Past the measured cold start the reassuring copy stops being honest, and the
// escalation is the one thing on this screen that is time-dependent.
const TAKING_LONGER_MS = 75_000;

describe("WorkspaceWaking", () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    // No runOnlyPendingTimers: RTL's cleanup has already unmounted the tree, so
    // flushing here only ticks a dead component and prints an act warning.
    jest.useRealTimers();
  });

  it("names the workspace being woken", () => {
    render(<WorkspaceWaking workspaceName="Research" />);
    expect(screen.getByRole("status")).toHaveTextContent("Waking Research");
  });

  it("falls back to a generic heading when the workspace name is unknown", () => {
    render(<WorkspaceWaking />);
    expect(screen.getByRole("status")).toHaveTextContent("Waking your workspace");
  });

  it("reassures that nothing was lost, since the workspace already existed", () => {
    render(<WorkspaceWaking workspaceName="Research" />);
    expect(screen.getByRole("status")).toHaveTextContent("nothing was lost");
  });

  it("counts the wait up in minutes and seconds", () => {
    render(<WorkspaceWaking workspaceName="Research" />);
    expect(screen.getByText("RESTORING · 0:00")).toBeInTheDocument();
    act(() => { jest.advanceTimersByTime(65_000); });
    expect(screen.getByText("RESTORING · 1:05")).toBeInTheDocument();
  });

  it("admits the wake is running long once it outlasts the cold start", () => {
    render(<WorkspaceWaking workspaceName="Research" />);
    act(() => { jest.advanceTimersByTime(TAKING_LONGER_MS + 1_000); });
    expect(screen.getByRole("status")).toHaveTextContent("taking longer than usual");
  });

  it("counts from when the wake started, not from when it mounted", () => {
    // The same wake is rendered from two places in the tree, so a route change
    // remounts this screen; a reload does too. Both used to restart the clock.
    render(<WorkspaceWaking workspaceName="Research" startedAt={Date.now() - 95_000} />);
    expect(screen.getByText("RESTORING · 1:35")).toBeInTheDocument();
  });

  it("carries the escalation across that remount too", () => {
    render(<WorkspaceWaking workspaceName="Research" startedAt={Date.now() - (TAKING_LONGER_MS + 5_000)} />);
    expect(screen.getByRole("status")).toHaveTextContent("taking longer than usual");
  });

  it("stops the clock when unmounted", () => {
    const { unmount } = render(<WorkspaceWaking workspaceName="Research" />);
    unmount();
    expect(jest.getTimerCount()).toBe(0);
  });
});
