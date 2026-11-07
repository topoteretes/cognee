import React, { createRef } from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import GraphControls, { GraphControlsAPI } from "../GraphControls";

function renderControls(isAddNodeFormOpen = false) {
  const ref = createRef<GraphControlsAPI>() as React.RefObject<GraphControlsAPI>;
  const props = { onFitIntoView: jest.fn(), onGraphShapeChange: jest.fn(), ref };
  const view = render(<GraphControls {...props} isAddNodeFormOpen={false} />);

  act(() => ref.current.setSelectedNode({ id: "node-1", label: "Entity" }));
  if (isAddNodeFormOpen) {
    view.rerender(<GraphControls {...props} isAddNodeFormOpen />);
  }

  return view;
}

describe("GraphControls (#5574)", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  it("shows a selected node that has no properties instead of crashing", () => {
    renderControls();

    expect(screen.getByText("node-1")).toBeInTheDocument();
    expect(screen.getByText("Entity")).toBeInTheDocument();
  });

  it("does not reload the page when the add-node form is submitted", () => {
    const { container } = renderControls(true);

    // fireEvent returns false when a handler called preventDefault().
    expect(fireEvent.submit(container.querySelector("form")!)).toBe(false);
  });

  it("does not submit the form from a property's delete button", () => {
    const { container } = renderControls(true);
    const [name, value] = screen.getAllByPlaceholderText(/Property (name|value)/);
    fireEvent.change(name, { target: { value: "color" } });
    fireEvent.change(value, { target: { value: "red" } });
    fireEvent.click(screen.getByText("Add"));

    // A <button> inside a <form> defaults to type="submit".
    expect(container.querySelector("form button.rounded-sm")).toHaveAttribute("type", "button");
  });
});
