import { fireEvent, render, screen } from "@testing-library/react";
import BusinessLegend from "../BusinessLegend";

describe("BusinessLegend", () => {
  it("marks the active color mode as pressed", () => {
    render(<BusinessLegend colorMode="type" onColorModeChange={jest.fn()} />);
    expect(screen.getByRole("button", { name: "type" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", { name: "source" }).getAttribute("aria-pressed")).toBe("false");
  });

  it("reports the mode the user picks", () => {
    const onColorModeChange = jest.fn();
    render(<BusinessLegend colorMode="type" onColorModeChange={onColorModeChange} />);
    fireEvent.click(screen.getByRole("button", { name: "source" }));
    expect(onColorModeChange).toHaveBeenCalledWith("source");
  });

  it("describes the color rule of the active mode and says types are per dataset", () => {
    const { rerender } = render(<BusinessLegend colorMode="type" onColorModeChange={jest.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "? legend" }));
    expect(screen.getByText(/another color in another dataset/)).toBeTruthy();
    rerender(<BusinessLegend colorMode="source" onColorModeChange={jest.fn()} />);
    expect(screen.getByText("color = which source it came from")).toBeTruthy();
  });
});
