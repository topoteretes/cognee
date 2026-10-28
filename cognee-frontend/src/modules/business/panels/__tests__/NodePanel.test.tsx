import { fireEvent, render, screen, within } from "@testing-library/react";
import NodePanel from "@/modules/business/panels/NodePanel";
import { computeBrainState } from "@/modules/business/computeBrainState";
import type { BusinessEntity } from "@/modules/business/sceneTypes";

function renderPanel(entity: BusinessEntity) {
  const onSimulateRemoval = jest.fn();
  render(<NodePanel entity={entity} docCount={0} events={[]}
    onClose={jest.fn()} pathTargetName={null} pathHops={null} onClearPath={jest.fn()}
    onSimulateRemoval={onSimulateRemoval} instance={{ name: "test", instanceId: "t1", fetch: jest.fn() }}
    datasetId="d1" graphLinks={[]} docLinks={[]} nodesById={{ [entity.id]: entity }} onSelectEntity={jest.fn()} />);
  return onSimulateRemoval;
}

describe("NodePanel", () => {
  it("shows a short CSV label, expandable complete row data and no business failure simulation", () => {
    const text = "Table: people\nColumns:\n  - person (TEXT)\n\nRow Data:\n  person: Ana Horvat\n  company: Atlas Labs";
    const state = computeBrainState([{ id: "row", type: "DltRow", name: "Table: people Columns:", text }], []);
    renderPanel(state.entities[0]);

    expect(screen.getByRole("heading", { name: "Ana Horvat" })).toBeInTheDocument();
    const summary = screen.getByText("Row details");
    expect(summary.closest("details")).not.toHaveAttribute("open");
    fireEvent.click(summary);
    expect(summary.closest("details")).toHaveAttribute("open");
    const table = screen.getByRole("table", { name: "Row values" });
    expect(within(table).getByRole("rowheader", { name: "person" })).toBeInTheDocument();
    expect(within(table).getByRole("cell", { name: "Ana Horvat" })).toBeInTheDocument();
    expect(within(table).getByRole("cell", { name: "Atlas Labs" })).toBeInTheDocument();
    expect(summary.closest("details")?.querySelector("pre")?.textContent).toBe(text);
    expect(screen.queryByRole("button", { name: /Preview disconnections/ })).not.toBeInTheDocument();
  });

  it("retains the simulation for a semantic entity without showing row details", () => {
    const onSimulateRemoval = renderPanel({ id: "person", name: "Ana", type: "Entity", stage: "entity" });
    fireEvent.click(screen.getByRole("button", { name: /Preview disconnections/ }));
    expect(onSimulateRemoval).toHaveBeenCalledTimes(1);
    expect(screen.queryByText("Row details")).not.toBeInTheDocument();
  });

  it("labels available activity and view scope, supports Escape, and restores keyboard focus", () => {
    const opener = document.createElement("button");
    document.body.append(opener);
    opener.focus();
    const entity = { id: "p", name: "Ana", type: "Entity", stage: "entity" };
    const onClose = jest.fn();
    const other = { id: "o", name: "Atlas", type: "Entity" };
    const { unmount } = render(<NodePanel entity={entity} docCount={0}
      events={[{ node_ids: ["p"], question: "Who is Ana?" }]}
      onClose={onClose} pathTargetName={null} pathHops={null} onClearPath={jest.fn()}
      onSimulateRemoval={jest.fn()} instance={{ name: "test", instanceId: "t1", fetch: jest.fn() }}
      datasetId="d1" graphLinks={[{ source: "p", target: "o", _sid: "p", _tid: "o", relation: "works_for" }, { source: "o", target: "p", _sid: "o", _tid: "p", relation: "owns" }]}
      docLinks={[]} nodesById={{ p: entity, o: other }} onSelectEntity={jest.fn()} />);
    expect(screen.getByRole("region", { name: "Ana" })).toHaveFocus();
    expect(screen.getByText(/2 connections in the loaded view/)).toBeInTheDocument();
    expect(screen.getByText(/Used in 1 answer in available activity/)).toBeInTheDocument();
    fireEvent.keyDown(screen.getByRole("button", { name: "Close item details" }), { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
    unmount();
    expect(opener).toHaveFocus();
    opener.remove();
  });

  it("opens the exact endpoint of a loaded generic relationship", () => {
    const entity = { id: "p", name: "Ana", type: "Entity", stage: "entity" };
    const target = { id: "company", name: "Atlas", type: "Entity" };
    const onSelect = jest.fn();
    render(<NodePanel entity={entity} docCount={0} events={[]}
      onClose={jest.fn()} pathTargetName={null} pathHops={null} onClearPath={jest.fn()}
      onSimulateRemoval={jest.fn()} instance={{ name: "test", instanceId: "t1", fetch: jest.fn() }}
      datasetId="d1" graphLinks={[{ source: "p", target: "company", _sid: "p", _tid: "company", relation: "works_for" }]}
      docLinks={[]} nodesById={{ p: entity, company: target }} onSelectEntity={onSelect} />);
    fireEvent.click(screen.getByText(/Outgoing relationships/));
    fireEvent.click(screen.getByRole("button", { name: /Atlas/ }));
    expect(onSelect).toHaveBeenCalledWith(target);
    expect(screen.getByText(/Only loaded relationships are listed/)).toBeInTheDocument();
  });

  function panelWith(props: { graphLinks?: object[]; docLinks?: object[]; nodes: Record<string, object>; entityId: string; datasetId?: string | null }) {
    const entity = props.nodes[props.entityId] as BusinessEntity;
    render(<NodePanel entity={entity} docCount={0} events={[]}
      onClose={jest.fn()} pathTargetName={null} pathHops={null} onClearPath={jest.fn()}
      onSimulateRemoval={jest.fn()} instance={{ name: "test", instanceId: "t1", fetch: jest.fn() }}
      datasetId={props.datasetId === undefined ? "d1" : props.datasetId}
      graphLinks={(props.graphLinks ?? []) as never} docLinks={(props.docLinks ?? []) as never}
      nodesById={props.nodes as never} onSelectEntity={jest.fn()} />);
  }

  it("lists the source records an entity is linked to instead of claiming it has no relationships", () => {
    const nodes = { e: { id: "e", name: "Ana", type: "Entity", stage: "entity" }, d: { id: "d", name: "Report", type: "TextDocument", stage: "document" } };
    panelWith({ nodes, entityId: "e", docLinks: [{ source: "d", target: "e", _sid: "d", _tid: "e", relation: "contains" }] });
    expect(screen.queryByText(/No relationships for this item are loaded/)).not.toBeInTheDocument();
    expect(screen.getByText(/Source records · 1 shown/)).toBeInTheDocument();
  });

  it("mounts a hub's rows only when its group is opened, a page at a time", () => {
    const nodes: Record<string, object> = { hub: { id: "hub", name: "Hub", type: "Entity", stage: "entity" } };
    const graphLinks = Array.from({ length: 250 }, (_, i) => {
      nodes[`n${i}`] = { id: `n${i}`, name: `Node ${i}`, type: "Entity", stage: "entity" };
      return { source: "hub", target: `n${i}`, _sid: "hub", _tid: `n${i}`, relation: "knows" };
    });
    panelWith({ nodes, entityId: "hub", graphLinks });
    expect(screen.queryAllByRole("button", { name: /Node \d+/ })).toHaveLength(0);
    fireEvent.click(screen.getByText(/Outgoing relationships · 250 shown/));
    expect(screen.getAllByRole("button", { name: /Node \d+/ })).toHaveLength(100);
    fireEvent.click(screen.getByRole("button", { name: /Show 100 more of 150 remaining/ }));
    expect(screen.getAllByRole("button", { name: /Node \d+/ })).toHaveLength(200);
  });

  it("still opens a node of another layer while no dataset is focused", () => {
    const nodes = { t: { id: "t", name: "Tenant", type: "Tenant", stage: "entity" } };
    panelWith({ nodes, entityId: "t", datasetId: null });
    expect(screen.getByRole("heading", { name: "Tenant" })).toBeInTheDocument();
  });
});
