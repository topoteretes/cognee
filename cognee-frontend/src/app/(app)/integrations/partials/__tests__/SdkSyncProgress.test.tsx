import { render, screen } from "@testing-library/react";
import type { SdkSyncRun } from "@/modules/integrations/sdkApi";
import SdkSyncProgress from "../SdkSyncProgress";

const run: SdkSyncRun = { runId: "one", state: "running", phase: "processing", completed: 1, total: 4, unit: "repositories", endedAt: null, counts: {} };

test("shows measured phase progress with repository counts", () => {
  render(<SdkSyncProgress runs={[run]} />);
  expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "25");
  expect(screen.getByText("25% of this phase")).toBeInTheDocument();
  expect(screen.getByText("1 / 4 repositories")).toBeInTheDocument();
});

test("unknown totals show indeterminate progress without an invented percentage", () => {
  render(<SdkSyncProgress runs={[{ ...run, phase: "discovering", total: null }]} />);
  expect(screen.getByRole("progressbar")).not.toHaveAttribute("aria-valuenow");
  expect(screen.getByText("Discovering repositories")).toBeInTheDocument();
  expect(screen.queryByText(/%/)).not.toBeInTheDocument();
});

test("completion keeps the final progress visible", () => {
  render(<SdkSyncProgress runs={[{ ...run, state: "succeeded", completed: 4 }]} />);
  expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");
  expect(screen.getByText("Sync complete")).toBeInTheDocument();
});

test.each(["failed", "partial", "cancelled", "blocked"] as const)("%s never becomes a full success bar", (state) => {
  render(<SdkSyncProgress runs={[{ ...run, state }]} />);
  expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "25");
  expect(screen.queryByText("Sync complete")).not.toBeInTheDocument();
});

test("unconfirmed activity labels the last observation instead of claiming it is live", () => {
  render(<SdkSyncProgress runs={[run]} activityUnknown />);
  expect(screen.getByText(/activity unconfirmed/)).toBeInTheDocument();
  expect(screen.queryByText(/of this phase/)).not.toBeInTheDocument();
});

test("renders every active invocation separately", () => {
  render(<SdkSyncProgress runs={[run, { ...run, runId: "two", completed: 2 }]} />);
  expect(screen.getAllByRole("progressbar").map(bar => bar.getAttribute("aria-valuenow"))).toEqual(["25", "50"]);
});
