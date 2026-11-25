import { render, screen } from "@testing-library/react";
import HistorySkeleton from "../partials/HistorySkeleton";

function placeholderBars(container: HTMLElement): number {
  return container.querySelectorAll("span[style*='animation']").length;
}

describe("HistorySkeleton", () => {
  it("draws one placeholder per row it was asked for", () => {
    const { container } = render(<HistorySkeleton rows={4} showGroupLabel={false} />);

    expect(placeholderBars(container)).toBe(4);
  });

  it("still draws a visible placeholder when asked for none", () => {
    const { container } = render(<HistorySkeleton rows={0} showGroupLabel={false} />);

    expect(placeholderBars(container)).toBe(1);
  });

  it("adds a heading placeholder only when asked to", () => {
    const { container: withLabel } = render(<HistorySkeleton rows={2} showGroupLabel />);
    const { container: without } = render(<HistorySkeleton rows={2} showGroupLabel={false} />);

    expect(placeholderBars(withLabel)).toBe(3);
    expect(placeholderBars(without)).toBe(2);
  });

  it("announces itself as loading", () => {
    render(<HistorySkeleton rows={2} showGroupLabel={false} />);

    expect(screen.getByRole("status", { name: "Loading past conversations" })).toBeInTheDocument();
  });
});
