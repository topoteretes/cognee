import { fireEvent, render, screen } from "@testing-library/react";
import Pager from "../partials/Pager";

const setup = (props: Partial<React.ComponentProps<typeof Pager>> = {}) => {
  const onGoTo = jest.fn();
  render(<Pager page={0} pageSize={100} total={171828} onGoTo={onGoTo} {...props} />);
  return { onGoTo };
};

describe("Pager", () => {
  it("renders nothing when the list is the whole dataset", () => {
    const { container } = render(
      <Pager page={0} pageSize={100} total={100} onGoTo={jest.fn()} />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("states the range and the true total, not the page length", () => {
    setup();

    expect(screen.getByText(/1–100 of 171,828/)).toBeInTheDocument();
    expect(screen.getByText("1 / 1,719")).toBeInTheDocument();
  });

  it("reports a partial last page correctly", () => {
    setup({ page: 1718 });

    // 171,828 = 1718 full pages of 100, then 28.
    expect(screen.getByText(/171,801–171,828 of 171,828/)).toBeInTheDocument();
  });

  it("disables backward navigation on the first page", () => {
    const { onGoTo } = setup({ page: 0 });

    fireEvent.click(screen.getByText("Prev"));
    expect(onGoTo).not.toHaveBeenCalled();
  });

  it("disables forward navigation on the last page", () => {
    const { onGoTo } = setup({ page: 1718 });

    fireEvent.click(screen.getByText("Next"));
    expect(onGoTo).not.toHaveBeenCalled();
  });

  it("navigates to the neighbouring and terminal pages", () => {
    const { onGoTo } = setup({ page: 5 });

    fireEvent.click(screen.getByText("Prev"));
    fireEvent.click(screen.getByText("Next"));
    fireEvent.click(screen.getByText("First"));
    fireEvent.click(screen.getByText("Last"));

    expect(onGoTo.mock.calls.map(([p]) => p)).toEqual([4, 6, 0, 1718]);
  });

  it("ignores clicks while a page is loading", () => {
    const { onGoTo } = setup({ page: 5, busy: true });

    fireEvent.click(screen.getByText("Next"));
    expect(onGoTo).not.toHaveBeenCalled();
  });
});
