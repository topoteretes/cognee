import { render, screen } from "@testing-library/react";
import PodUnreachableCard from "../PodUnreachableCard";

jest.mock("@/modules/analytics", () => ({ trackEvent: jest.fn() }));

describe("PodUnreachableCard", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it("blames the network by default, which is what an unreachable pod looks like", () => {
    render(<PodUnreachableCard />);
    expect(screen.getByRole("heading")).toHaveTextContent("We're having trouble reaching your workspace");
  });

  it("names the workspace that failed to wake", () => {
    render(<PodUnreachableCard wakeFailed workspaceName="Research" />);
    expect(screen.getByRole("heading")).toHaveTextContent("Research didn't wake up");
  });

  it("still reports a failed wake when the workspace name is unknown", () => {
    render(<PodUnreachableCard wakeFailed />);
    expect(screen.getByRole("heading")).toHaveTextContent("Your workspace didn't wake up");
  });

  it("says the restore was asked for, so the user knows this is not their connection", () => {
    render(<PodUnreachableCard wakeFailed workspaceName="Research" />);
    expect(screen.getByText(/we asked it to start/i)).toBeInTheDocument();
  });
});
