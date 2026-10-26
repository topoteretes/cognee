import { fireEvent, render, screen } from "@testing-library/react";
import GroupRail from "@/app/(app)/datasets/partials/GroupRail";
import { ALL_FACET_ID, PERSONAL_FACET_ID, type DatasetFacet } from "@/app/(app)/datasets/partials/datasetFacets";

const ALL_FACET: DatasetFacet = { id: ALL_FACET_ID, kind: "all", label: "All datasets", hint: "Everything you can open", datasetIds: new Set() };
const PERSONAL_FACET: DatasetFacet = { id: PERSONAL_FACET_ID, kind: "personal", label: "Personal", hint: "Only you, unless shared", datasetIds: new Set(["a", "b"]) };
const GROUP_FACET: DatasetFacet = { id: "g-eng", kind: "group", label: "Engineering", hint: "8 members", datasetIds: new Set(["a"]) };

describe("GroupRail", () => {
  it("shows a skeleton row per facet while loading, not the facets themselves", () => {
    render(<GroupRail facets={[ALL_FACET, PERSONAL_FACET]} selectedId={ALL_FACET_ID} status="loading" allCount={2} onSelect={jest.fn()} />);

    expect(screen.queryByText("Personal")).not.toBeInTheDocument();
  });

  it("counts All from the page's own dataset count, not from the access list", () => {
    render(<GroupRail facets={[ALL_FACET, PERSONAL_FACET]} selectedId={ALL_FACET_ID} status="ready" allCount={42} onSelect={jest.fn()} />);

    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("counts every other facet from its own dataset set", () => {
    render(<GroupRail facets={[ALL_FACET, PERSONAL_FACET]} selectedId={ALL_FACET_ID} status="ready" allCount={42} onSelect={jest.fn()} />);

    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("reports which facet is selected through aria-pressed, not just styling", () => {
    render(<GroupRail facets={[ALL_FACET, PERSONAL_FACET]} selectedId={PERSONAL_FACET_ID} status="ready" allCount={2} onSelect={jest.fn()} />);

    expect(screen.getByRole("button", { name: /Personal/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /All datasets/ })).toHaveAttribute("aria-pressed", "false");
  });

  it("calls onSelect with the clicked facet's id", () => {
    const onSelect = jest.fn();
    render(<GroupRail facets={[ALL_FACET, PERSONAL_FACET]} selectedId={ALL_FACET_ID} status="ready" allCount={2} onSelect={onSelect} />);

    fireEvent.click(screen.getByRole("button", { name: /Personal/ }));

    expect(onSelect).toHaveBeenCalledWith(PERSONAL_FACET_ID);
  });

  it("drops every facet but All when the access list failed to load, and says why", () => {
    render(<GroupRail facets={[ALL_FACET, PERSONAL_FACET, GROUP_FACET]} selectedId={ALL_FACET_ID} status="error" allCount={2} onSelect={jest.fn()} />);

    expect(screen.getByRole("button", { name: /All datasets/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Personal/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Engineering/ })).not.toBeInTheDocument();
    expect(screen.getByText(/Groups could not be loaded/)).toBeInTheDocument();
  });
});
