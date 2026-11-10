import { MantineProvider } from "@mantine/core";
import { act, fireEvent, render, screen } from "@testing-library/react";
import BrainSwitcher from "../BrainSwitcher";
import type { BrainsSummaryPayload, BusinessGraphNode } from "../../types";
import type { GovernanceIndex } from "../../useGovernanceIndex";
import { narrowToReadable } from "../../useGovernanceIndex";

function dataset(id: string, name: string): BusinessGraphNode {
  return { id, name, type: "Dataset" } as BusinessGraphNode;
}

const EMPTY_INDEX: GovernanceIndex = {
  tenants: [], users: [], agents: [], datasets: [], datasetById: {}, access: {}, agentOwnerId: {},
};

function renderSwitcher(
  brains: BrainsSummaryPayload | null,
  listState: "loading" | "unavailable" | "ready" = "ready",
  datasets: BusinessGraphNode[] = [dataset("ds-1", "sales")],
  activeDatasetId: string | null = "ds-1",
) {
  render(
    <MantineProvider>
      <BrainSwitcher
        brains={brains}
        index={narrowToReadable(
          {
            ...EMPTY_INDEX,
            datasets,
            datasetById: Object.fromEntries(datasets.map((d) => [d.id, d])),
          },
          datasets.map((d) => d.id),
        )}
        listState={listState}
        activeDatasetId={activeDatasetId}
        onSelect={jest.fn()}
        hoveredPrincipalId={null}
      />
    </MantineProvider>,
  );
  fireEvent.click(screen.getByRole("button"));
}

// Mantine mounts the dropdown a tick after the click, into a portal that
// starts at display:none — so a negative assertion made straight after the
// click reads an unmounted menu as an empty one and passes for the wrong
// reason. Wait for the menu, then read it, and query it with hidden: true.
async function openedMenu(): Promise<HTMLElement> {
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 50)); });
  return screen.getByRole("menu", { hidden: true });
}

describe("BrainSwitcher", () => {
  it("previews a dataset's sources straight from the summary payload", async () => {
    renderSwitcher({
      "ds-1": { name: "sales", source_names: ["crm", "slack"], node_count: 42, node_set_colors: {} },
    });

    expect(await screen.findByText("crm · slack")).toBeInTheDocument();
  });

  it("renders the dataset without a source line when the summary has not loaded", () => {
    // The summary is enrichment only: the list itself comes from governance,
    // so a failed or pending brains-summary must never cost a dataset its row.
    renderSwitcher(null);

    expect(screen.getAllByText("sales").length).toBeGreaterThan(0);
    expect(screen.queryByText(/·/)).not.toBeInTheDocument();
  });
});

describe("BrainSwitcher — a dataset that has never been cognified", () => {
  it("says so instead of listing sources that lead to an empty Mindmap", async () => {
    renderSwitcher({
      "ds-1": { name: "sales", source_names: ["crm"], node_count: 0, node_set_colors: {} },
    });
    expect(await screen.findByText("not cognified yet")).toBeInTheDocument();
    expect(screen.queryByText("crm")).not.toBeInTheDocument();
  });

  it("lists sources once the dataset has a graph", async () => {
    renderSwitcher({
      "ds-1": { name: "sales", source_names: ["crm"], node_count: 42, node_set_colors: {} },
    });
    expect(await screen.findByText("crm")).toBeInTheDocument();
    expect(screen.queryByText("not cognified yet")).not.toBeInTheDocument();
  });
});

// The index this component receives is already narrowed to the caller's
// readable set (narrowToReadable, applied once in BusinessView), so the
// switcher can no longer decide what is offered — it can only explain why
// the list it was handed is empty. These three reasons used to be one empty
// array, and the "unavailable" one used to fall back to the tenant's full
// list, which named other members' datasets (CLO-932 follow-up).
describe("BrainSwitcher — an empty readable list", () => {
  it("says the list is still loading rather than offering nothing in silence", async () => {
    renderSwitcher(null, "loading", []);
    expect((await openedMenu()).textContent).toContain("loading your datasets");
  });

  it("says the list could not be loaded when the fetch failed", async () => {
    renderSwitcher(null, "unavailable", []);
    expect((await openedMenu()).textContent).toContain("couldn't load your datasets");
  });

  it("says the caller may read nothing when the list answered and was empty", async () => {
    renderSwitcher(null, "ready", []);
    expect((await openedMenu()).textContent).toContain("no datasets you can read");
  });

  it("offers exactly the datasets the narrowed index carries", async () => {
    renderSwitcher(null, "ready", [dataset("ds-1", "sales")]);
    const menu = await openedMenu();
    expect(menu.textContent).toContain("sales");
    expect(menu.textContent).not.toContain("loading your datasets");
  });
});

// The chip, not the dropdown. A dataset is already opening (from the id this
// browser remembered) while the readable list is still answering, and for
// those seconds the narrowed index cannot vouch for its name. Labelling the
// chip "select a dataset" through a visibly loading graph is the complaint
// b9cbef4f fixed by handing this component the tenant list, which is what
// leaked; saying what is happening instead costs no name.
describe("BrainSwitcher — the chip label", () => {
  function chipText(): string {
    return screen.getAllByRole("button")[0].textContent || "";
  }

  it("says a dataset is opening while the readable list is still answering", () => {
    renderSwitcher(null, "loading", [], "ds-remembered");
    expect(chipText()).toContain("opening");
    expect(chipText()).not.toContain("select a dataset");
  });

  it("asks for a pick when nothing is opening and the list has answered", () => {
    renderSwitcher(null, "ready", [], null);
    expect(chipText()).toContain("select a dataset");
  });

  it("does not claim nothing is selected when the list failed under a rendered graph", () => {
    // The seed fallback opens a governance-seeded dataset exactly in this
    // state, so "select a dataset" contradicts the canvas behind the chip.
    renderSwitcher(null, "unavailable", [], "ds-seeded");
    expect(chipText()).toContain("current dataset");
    expect(chipText()).not.toContain("select a dataset");
  });

  it("asks for a pick when the list failed and nothing is open", () => {
    renderSwitcher(null, "unavailable", [], null);
    expect(chipText()).toContain("select a dataset");
  });

  it("names the dataset as soon as the readable list vouches for it", () => {
    renderSwitcher(null, "ready", [dataset("ds-1", "sales")], "ds-1");
    expect(chipText()).toContain("sales");
  });
});

describe("BrainSwitcher — many datasets", () => {
  const many = Array.from({ length: 40 }, (_, i) => dataset(`ds-${i}`, `dataset ${i}`));

  // Row 48px, label 22px, 6.5 rows, dropdown padding and border 10px.
  const ONE_SECTION_HEIGHT = 10 + 22 + 6.5 * 48;

  it("caps the dropdown so its last visible row is cut in half, and scrolls", async () => {
    renderSwitcher(null, "ready", many, "ds-0");

    const menu = await openedMenu();
    expect(menu.style.maxHeight).toBe(`${ONE_SECTION_HEIGHT}px`);
    expect(menu.style.overflowY).toBe("auto");
  });

  it("gives every row the same height whether or not it has a subtitle", async () => {
    renderSwitcher({
      "ds-0": { name: "dataset 0", source_names: ["crm"], node_count: 4, node_set_colors: {} },
    }, "ready", many, "ds-0");

    const items = (await openedMenu()).querySelectorAll<HTMLElement>("[role='menuitem']");
    expect(new Set([...items].map((item) => item.style.height))).toEqual(new Set(["48px"]));
  });

  it("fades the bottom edge only while more rows are below", async () => {
    renderSwitcher(null, "ready", many, "ds-0");
    const menu = await openedMenu();
    const fade = menu.querySelector(".bv-brain-switcher-fade");
    const scrollTo = (scrollTop: number): void => {
      Object.defineProperty(menu, "scrollHeight", { configurable: true, value: 2000 });
      Object.defineProperty(menu, "clientHeight", { configurable: true, value: 300 });
      menu.scrollTop = scrollTop;
      fireEvent.scroll(menu);
    };

    scrollTo(0);
    expect(fade).toHaveAttribute("data-visible", "true");
    scrollTo(1700);
    expect(fade).toHaveAttribute("data-visible", "false");
  });

  it("shows no fade when the list fits", async () => {
    renderSwitcher(null, "ready", [dataset("ds-1", "sales")], "ds-1");
    const menu = await openedMenu();
    expect(menu.querySelector(".bv-brain-switcher-fade")).toHaveAttribute("data-visible", "false");
  });
});
