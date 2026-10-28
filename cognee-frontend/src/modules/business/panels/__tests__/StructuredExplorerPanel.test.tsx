import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen } from "@testing-library/react";
import StructuredExplorerPanel from "@/modules/business/panels/StructuredExplorerPanel";
import { useStructuredExplorer } from "@/modules/business/useStructuredExplorer";
import { computeBrainState } from "@/modules/business/computeBrainState";
import { buildStructuredExplorer } from "@/modules/business/structuredExplorer";

class ResizeObserverStub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}
const originalResizeObserver = global.ResizeObserver;
beforeAll(() => { global.ResizeObserver = ResizeObserverStub; });
afterAll(() => { global.ResizeObserver = originalResizeObserver; });

const brain = computeBrainState([
  { id: "a", type: "DltRow", document_name: "people.csv", text: "Row Data:\n  company: Atlas Labs" },
  { id: "b", type: "DltRow", document_name: "accounts.csv", text: "Row Data:\n  company: Atlas Labs" },
  { id: "c", type: "DltRow", document_name: "people.csv", text: "Row Data:\n  company: Other" },
], []);
function Harness() {
  const explorer = useStructuredExplorer(brain, "dataset");
  return <StructuredExplorerPanel index={explorer.index} filter={explorer.filter}
    onChange={explorer.setFilter} matchingCount={explorer.rowIds?.size ?? null} />;
}

describe("StructuredExplorerPanel", () => {
  it("stays mounted with a clearable filter when a refresh leaves no rows", () => {
    const onChange = jest.fn();
    render(<MantineProvider><StructuredExplorerPanel index={buildStructuredExplorer(null)}
      filter={{ column: "company", value: "Atlas Labs" }} onChange={onChange} matchingCount={0} /></MantineProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Clear filter" }));
    expect(onChange).toHaveBeenCalledWith({ column: null, value: null });
  });

  it("shows the absence of loaded links and lets a candidate filter rows without creating relationships", () => {
    render(<MantineProvider><Harness /></MantineProvider>);
    expect(screen.getByText(/No row-to-row links or shared value nodes/, { selector: "section > p" })).toBeVisible();
    fireEvent.click(screen.getByText("CSV & tables · 3 loaded rows"));
    expect(screen.getByText(/Possible matches only/)).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "company = Atlas Labs · 2 rows" }));
    expect(screen.getByRole("status")).toHaveTextContent("2 matching loaded rows");
    expect(brain.graphLinks).toEqual([]);
    fireEvent.click(screen.getByRole("button", { name: "Clear filter" }));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("supports searchable column and exact value selection", () => {
    render(<MantineProvider><Harness /></MantineProvider>);
    fireEvent.click(screen.getByText("CSV & tables · 3 loaded rows"));
    fireEvent.change(screen.getByRole("textbox", { name: "Column" }), { target: { value: "comp" } });
    fireEvent.click(screen.getByRole("option", { name: "company" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Value equals" }), { target: { value: "Atlas" } });
    fireEvent.click(screen.getByRole("option", { name: "Atlas Labs · 2 rows" }));
    expect(screen.getByRole("status")).toHaveTextContent("2 matching loaded rows");
  });

  it("explains unavailable row values and empty filter results", () => {
    const index = buildStructuredExplorer(computeBrainState([{ id: "missing", type: "DltRow" }], []));
    render(<MantineProvider><StructuredExplorerPanel index={index}
      filter={{ column: "company", value: "Atlas Labs" }} matchingCount={0} onChange={jest.fn()} /></MantineProvider>);
    fireEvent.click(screen.getByText("CSV & tables · 1 loaded rows"));
    expect(screen.getByText(/Column values are unavailable for 1/)).toBeVisible();
    expect(screen.getByRole("status")).toHaveTextContent("No loaded rows match this value.");
  });
});
