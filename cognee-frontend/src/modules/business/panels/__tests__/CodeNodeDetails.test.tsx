import { fireEvent, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import CodeNodeDetails from "../CodeNodeDetails";
import type { CogneeInstance } from "@/modules/instances/types";

const anchor = { id: "users", name: "src/modules/users", type: "CodeModule", kind: "module", doc: "Manages user data.", file: "src/modules/users.ts", line: 10, end_line: 34, repo: "frontend" };
const neighbor = (id: string, name: string) => ({ id, name, kind: "symbol", type: "CodeSymbol" });
const page = { operation: "neighbors", anchor, neighbors: [
  { direction: "out", relation: "imports", node: neighbor("a", "src/a/users") },
  { direction: "in", relation: "calls", node: neighbor("b", "src/b/users") },
], total: 3, offset: 0, limit: 2, has_more: true };

function setup() {
  const fetch = jest.fn().mockResolvedValue({ json: async () => [{ search_result: page }] });
  const instance: CogneeInstance = { name: "test", instanceId: "t1", fetch };
  const onSelect = jest.fn();
  const client = new QueryClient();
  render(<QueryClientProvider client={client}>
    <CodeNodeDetails instance={instance} datasetId="d1" nodeId="users" onSelect={onSelect} />
  </QueryClientProvider>);
  return { fetch, onSelect };
}

it("loads description and source, separates directions and opens exact identities with duplicate short names", async () => {
  const { onSelect } = setup();
  expect(screen.getByRole("status")).toHaveTextContent("Loading");
  expect(screen.queryByText(/No description/)).not.toBeInTheDocument();
  expect(await screen.findByText("Manages user data.")).toBeInTheDocument();
  expect(screen.getByRole("region", { name: "Source location" })).toHaveTextContent("src/modules/users.ts:10–34");
  expect(screen.getByText("Repository: frontend")).toBeInTheDocument();
  expect(screen.getByText(/2 of 3 relationships loaded/)).toBeInTheDocument();
  const uses = screen.getByRole("region", { name: "Uses" });
  const usedBy = screen.getByRole("region", { name: "Used by" });
  expect(within(uses).queryByText("src/a/users")).not.toBeInTheDocument(); // collapsed groups mount no rows
  fireEvent.click(within(uses).getByText(/Uses/));
  expect(within(uses).getByText("src/a/users")).toBeInTheDocument();
  fireEvent.click(within(usedBy).getByText(/Used by/));
  expect(within(usedBy).getByText("src/b/users")).toBeInTheDocument();
  fireEvent.click(within(usedBy).getByRole("button"));
  expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ id: "b", name: "src/b/users", type: "CodeSymbol" }));
});

it("loads the next page without losing details and makes all relationships reachable", async () => {
  const { fetch } = setup();
  const more = await screen.findByRole("button", { name: "Load more relationships" });
  fetch.mockResolvedValue({ json: async () => ({ ...page, offset: 2, neighbors: [
    { direction: "out", relation: "declares", node: neighbor("c", "src/users.fetchUsers") },
  ], has_more: false }) });
  fireEvent.click(more);
  fireEvent.click(await screen.findByText(/Declarations/));
  expect(await screen.findByText("fetchUsers")).toBeInTheDocument();
  expect(screen.getByText(/3 of 3 relationships loaded/)).toBeInTheDocument();
  expect(screen.getByText("Manages user data.")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Load more relationships" })).not.toBeInTheDocument();
  expect(JSON.parse(fetch.mock.calls[1][1].body).codeQuery.offset).toBe(2);
});

it("keeps earlier pages when loading more fails and retries that page", async () => {
  const { fetch } = setup();
  const more = await screen.findByRole("button", { name: "Load more relationships" });
  fetch.mockRejectedValue(new Error("No connection to the server."));
  fireEvent.click(more);
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not load more relationships.");
  expect(screen.getByText("Manages user data.")).toBeInTheDocument();
  expect(screen.getByRole("region", { name: "Uses" })).toBeInTheDocument();
  fetch.mockResolvedValue({ json: async () => ({ ...page, offset: 2, neighbors: [
    { direction: "out", relation: "declares", node: neighbor("c", "fetchUsers") },
  ], has_more: false }) });
  fireEvent.click(screen.getByRole("button", { name: "Retry details" }));
  expect(await screen.findByText(/3 of 3 relationships loaded/)).toBeInTheDocument();
});

it("shows unavailable metadata only after a successful read, and distinguishes it from a failed read", async () => {
  const fetch = jest.fn().mockRejectedValue(new Error("Unsupported CODE operation 'neighbors'."));
  const client = new QueryClient();
  render(<QueryClientProvider client={client}>
    <CodeNodeDetails instance={{ name: "test", instanceId: "t1", fetch }} datasetId="d1" nodeId="users" onSelect={jest.fn()} />
  </QueryClientProvider>);
  expect(await screen.findByRole("alert")).toHaveTextContent("Unsupported CODE operation");
  expect(screen.queryByText(/No description/)).not.toBeInTheDocument();
  fetch.mockResolvedValue({ json: async () => ({ ...page, anchor: { id: "users", kind: "module", type: "CodeModule" }, neighbors: [], total: 0, has_more: false }) });
  fireEvent.click(screen.getByRole("button", { name: "Retry details" }));
  expect(await screen.findByText(/No description is available/)).toBeInTheDocument();
  expect(screen.getByText(/No file location is available/)).toBeInTheDocument();
  expect(screen.getByText(/No relationships were found/)).toBeInTheDocument();
});
