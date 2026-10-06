import { fireEvent, render, screen } from "@testing-library/react";
import ProcessingSummary from "../partials/ProcessingSummary";
import DocumentList from "../partials/DocumentList";

test("shows partial completion after a failed run without claiming all remaining items failed", () => {
  const refresh = jest.fn();
  render(<ProcessingSummary data={{ total: 164, completed: 162, pending: 2, items: [] }} running={false} failed error={false} onRefresh={refresh} />);
  expect(screen.getByText("162 of 164 ready to search")).toBeInTheDocument();
  expect(screen.getByText(/2 remaining · Last processing run failed/)).toBeInTheDocument();
  expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "162");
  fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
  expect(refresh).toHaveBeenCalledTimes(1);
});

test("an unavailable count is not shown as zero or complete", () => {
  render(<ProcessingSummary running={false} failed={false} error onRefresh={jest.fn()} />);
  expect(screen.getByText("Processing progress unavailable")).toBeInTheDocument();
  expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
});

test("shows stored completion per document, leaving unknown items unclassified", () => {
  render(<DocumentList docs={[{ id: "1", name: "ready.txt", completed: true }, { id: "2", name: "pending.txt", completed: false }, { id: "3", name: "unknown.txt" }]} onDelete={jest.fn()} />);
  expect(screen.getAllByText("Ready")).toHaveLength(1);
  expect(screen.getAllByText("Not ready")).toHaveLength(1);
});
