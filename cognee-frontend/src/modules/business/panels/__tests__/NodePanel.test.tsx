import { fireEvent, render, screen, within } from "@testing-library/react";
import NodePanel from "@/modules/business/panels/NodePanel";
import { computeBrainState } from "@/modules/business/computeBrainState";
import type { BusinessEntity } from "@/modules/business/sceneTypes";

function renderPanel(entity: BusinessEntity) {
  const onSimulateRemoval = jest.fn();
  render(<NodePanel entity={entity} docCount={0} connectionCount={2} events={[]}
    onClose={jest.fn()} pathTargetName={null} pathHops={null} onClearPath={jest.fn()}
    onSimulateRemoval={onSimulateRemoval} />);
  return onSimulateRemoval;
}

describe("NodePanel", () => {
  it("shows a short CSV label, expandable complete row data and no business failure simulation", () => {
    const text = "Table: people\nColumns:\n  - person (TEXT)\n\nRow Data:\n  person: Ana Horvat\n  company: Atlas Labs";
    const state = computeBrainState([{ id: "row", type: "DltRow", name: "Table: people Columns:", text }], []);
    renderPanel(state.entities[0]);

    expect(screen.getByText("Ana Horvat", { selector: "div" })).toBeInTheDocument();
    const summary = screen.getByText("Row details");
    expect(summary.closest("details")).not.toHaveAttribute("open");
    fireEvent.click(summary);
    expect(summary.closest("details")).toHaveAttribute("open");
    const table = screen.getByRole("table", { name: "Row values" });
    expect(within(table).getByRole("rowheader", { name: "person" })).toBeInTheDocument();
    expect(within(table).getByRole("cell", { name: "Ana Horvat" })).toBeInTheDocument();
    expect(within(table).getByRole("cell", { name: "Atlas Labs" })).toBeInTheDocument();
    expect(summary.closest("details")?.querySelector("pre")?.textContent).toBe(text);
    expect(screen.queryByRole("button", { name: /what breaks/ })).not.toBeInTheDocument();
  });

  it("retains the simulation for a semantic entity without showing row details", () => {
    const onSimulateRemoval = renderPanel({ id: "person", name: "Ana", type: "Entity", stage: "entity" });
    fireEvent.click(screen.getByRole("button", { name: /what breaks/ }));
    expect(onSimulateRemoval).toHaveBeenCalledTimes(1);
    expect(screen.queryByText("Row details")).not.toBeInTheDocument();
  });
});
