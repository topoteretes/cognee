import { Profiler, StrictMode, type ComponentProps } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import BusinessGraphCanvas from "@/modules/business/canvas/BusinessGraphCanvas";
import type BusinessCanvas from "@/modules/business/canvas/BusinessCanvas";
import { computeBrainState } from "@/modules/business/computeBrainState";

const mockCanvasRender = jest.fn();
jest.mock("@/modules/business/canvas/BusinessCanvas", () => ({
  __esModule: true,
  default: function Canvas(props: ComponentProps<typeof BusinessCanvas>) {
    mockCanvasRender();
    return <canvas data-testid="canvas" onMouseMove={(event) => {
      props.onHoverMove?.(event.clientX, event.clientY, 800);
      const entity = props.brainState?.entities[0];
      props.onHover(entity ? { kind: "entity", node: entity } : null);
    }} onMouseLeave={() => props.onHover(null)} />;
  },
}));

const brain = computeBrainState([
  { id: "row", type: "DltRow", text: "Table: people\nRow Data:\n  name: Ana Horvat", name: "row" },
  { id: "table", type: "SchemaTable", name: "people" },
], [{ source: "row", target: "table", relation: "is_row_of" }]);

function props(): ComponentProps<typeof BusinessGraphCanvas> {
  return {
    brainState: brain, colorMode: "type", selectedId: null,
    onSelectEntity: jest.fn(), onBackgroundClick: jest.fn(),
    spotlight: null, focusSets: null, focusTypes: null, activeDatasetId: "people",
  };
}

it("keeps repeated hover updates out of the parent and the canvas while moving the tooltip", () => {
  const parentRender = jest.fn();
  const graphProps = props();
  function Parent() {
    parentRender();
    return <BusinessGraphCanvas {...graphProps} />;
  }
  render(<StrictMode><Parent /></StrictMode>);
  const initialParentRenders = parentRender.mock.calls.length;
  const initialCanvasRenders = mockCanvasRender.mock.calls.length;
  const canvas = screen.getByTestId("canvas");
  for (let x = 0; x < 100; x += 1) fireEvent.mouseMove(canvas, { clientX: x, clientY: 50 });

  const tooltip = screen.getByText("Ana Horvat").parentElement;
  expect(tooltip).toHaveStyle({ left: "113px", top: "64px" });
  expect(screen.getByText(/1 connection/)).toBeInTheDocument();
  expect(parentRender).toHaveBeenCalledTimes(initialParentRenders);
  expect(mockCanvasRender).toHaveBeenCalledTimes(initialCanvasRenders);
  fireEvent.mouseLeave(canvas);
  expect(screen.queryByText("Ana Horvat")).not.toBeInTheDocument();
});

it("does not keep committing duplicate mouse coordinates while no node is hovered", () => {
  const commits = jest.fn();
  render(<Profiler id="graph" onRender={commits}><BusinessGraphCanvas {...props()} brainState={null} /></Profiler>);
  const canvas = screen.getByTestId("canvas");
  fireEvent.mouseMove(canvas, { clientX: 40, clientY: 50 });
  // React may perform one extra render before bailing out of equal state.
  fireEvent.mouseMove(canvas, { clientX: 40, clientY: 50 });
  const settledCommits = commits.mock.calls.length;
  for (let i = 0; i < 100; i += 1) fireEvent.mouseMove(canvas, { clientX: 40, clientY: 50 });
  expect(commits).toHaveBeenCalledTimes(settledCommits);
});

it("hides the previous dataset's tooltip when the dataset changes", () => {
  const graphProps = props();
  const { rerender } = render(<BusinessGraphCanvas {...graphProps} />);
  fireEvent.mouseMove(screen.getByTestId("canvas"), { clientX: 780, clientY: 50 });
  expect(screen.getByText("Ana Horvat").parentElement).toHaveStyle({ left: "540px" });
  rerender(<BusinessGraphCanvas {...graphProps} activeDatasetId="other" brainState={null} />);
  expect(screen.queryByText("Ana Horvat")).not.toBeInTheDocument();
});
