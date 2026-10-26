import { fireEvent, render, screen } from "@testing-library/react";
import DatasetListFilters from "@/app/(app)/datasets/partials/DatasetListFilters";

describe("DatasetListFilters", () => {
  it("reports the search text as the user types it", () => {
    const onSearch = jest.fn();
    render(<DatasetListFilters search="" status="all" onSearch={onSearch} onStatus={jest.fn()} />);

    fireEvent.change(screen.getByPlaceholderText("Search datasets"), { target: { value: "crm" } });

    expect(onSearch).toHaveBeenCalledWith("crm");
  });

  it("marks only the active status chip as pressed", () => {
    render(<DatasetListFilters search="" status="processing" onSearch={jest.fn()} onStatus={jest.fn()} />);

    expect(screen.getByRole("button", { name: "Processing" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Ready" })).toHaveAttribute("aria-pressed", "false");
  });

  it("calls onStatus with the chip's own filter value, not the one that was active", () => {
    const onStatus = jest.fn();
    render(<DatasetListFilters search="" status="all" onSearch={jest.fn()} onStatus={onStatus} />);

    fireEvent.click(screen.getByRole("button", { name: "Attention" }));

    expect(onStatus).toHaveBeenCalledWith("attention");
  });
});
