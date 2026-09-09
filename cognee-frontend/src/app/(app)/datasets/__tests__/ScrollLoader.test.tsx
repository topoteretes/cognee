import { fireEvent, render, screen } from "@testing-library/react";
import ScrollLoader from "../partials/ScrollLoader";

/** Capture the observer so a scroll into view can be simulated. */
let observed: { cb: IntersectionObserverCallback; disconnect: jest.Mock } | null = null;

beforeEach(() => {
  observed = null;
  (global as unknown as { IntersectionObserver: unknown }).IntersectionObserver = jest
    .fn()
    .mockImplementation((cb: IntersectionObserverCallback) => {
      const disconnect = jest.fn();
      observed = { cb, disconnect };
      return { observe: jest.fn(), unobserve: jest.fn(), disconnect };
    });
});

const scrollIntoView = () =>
  observed?.cb([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver);

const setup = (props: Partial<React.ComponentProps<typeof ScrollLoader>> = {}) => {
  const onLoadMore = jest.fn();
  const view = render(
    <ScrollLoader
      loaded={100}
      total={171828}
      maxLoaded={2000}
      onLoadMore={onLoadMore}
      {...props}
    />,
  );
  return { onLoadMore, ...view };
};

describe("ScrollLoader", () => {
  it("renders nothing when everything is already loaded", () => {
    const { container } = render(
      <ScrollLoader loaded={40} total={40} maxLoaded={2000} onLoadMore={jest.fn()} />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("states progress against the true total", () => {
    setup();

    expect(screen.getByText(/100 of 171,828 documents/)).toBeInTheDocument();
  });

  it("loads the next step when scrolled into view", () => {
    const { onLoadMore } = setup();

    scrollIntoView();

    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });

  it("does not observe while a step is already in flight", () => {
    const { onLoadMore } = setup({ busy: true });

    expect(observed).toBeNull();
    expect(onLoadMore).not.toHaveBeenCalled();
    expect(screen.getByText("Loading more…")).toBeInTheDocument();
  });

  it("stops loading at the render bound and says why", () => {
    const { onLoadMore } = setup({ loaded: 2000 });

    expect(observed).toBeNull();
    scrollIntoView();
    expect(onLoadMore).not.toHaveBeenCalled();
    expect(screen.getByText(/Showing the first 2,000/)).toBeInTheDocument();
    expect(screen.queryByText("Load more")).not.toBeInTheDocument();
  });

  it("offers a button as well, for browsers and keyboards the observer misses", () => {
    const { onLoadMore } = setup();

    fireEvent.click(screen.getByText("Load more"));

    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });

  it("does not claim a cap when the cap is exactly the total", () => {
    setup({ loaded: 2000, total: 2000, maxLoaded: 2000 });

    expect(screen.queryByText(/Showing the first/)).not.toBeInTheDocument();
  });

  it("degrades to the button where IntersectionObserver is unavailable", () => {
    (global as unknown as { IntersectionObserver?: unknown }).IntersectionObserver = undefined;

    const { onLoadMore } = setup();
    fireEvent.click(screen.getByText("Load more"));

    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });
});
