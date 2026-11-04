import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen } from "@testing-library/react";
import BrainList from "@/app/(app)/datasets/partials/BrainList";
import { PENDING_DATASET_PREFIX } from "@/app/(app)/datasets/brainsTypes";

const READY = { id: "ds-1", name: "notes", status: "completed" as const, documents: 3 };
const CREATING = { id: `${PENDING_DATASET_PREFIX}abc`, name: "fresh", status: "loading" as const, documents: -1 };

function renderList(onSelect = jest.fn()): jest.Mock {
  render(
    <MantineProvider>
      <BrainList brains={[READY, CREATING]} selectedId={null} outdatedIds={new Set()}
        onSelect={onSelect} onCreate={jest.fn()} onDelete={jest.fn()} />
    </MantineProvider>,
  );
  return onSelect;
}

describe("a dataset still being created", () => {
  it("is listed, so the dataset is on screen as soon as it is submitted", () => {
    renderList();

    expect(screen.getByText("fresh")).toBeInTheDocument();
  });

  it("cannot be opened or deleted before it exists", () => {
    const onSelect = renderList();

    fireEvent.click(screen.getByText("fresh"));

    expect(onSelect).not.toHaveBeenCalled();
    expect(screen.getAllByTitle("Delete dataset")).toHaveLength(1);
  });

  it("does not stop a finished dataset from being opened", () => {
    const onSelect = renderList();

    fireEvent.click(screen.getByText("notes"));

    expect(onSelect).toHaveBeenCalledWith("ds-1");
  });
});
