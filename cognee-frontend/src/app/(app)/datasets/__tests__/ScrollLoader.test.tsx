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
    expect(screen.getByText(/This view is limited to 2,000/)).toBeInTheDocument();
    expect(screen.queryByText("Load more")).not.toBeInTheDocument();
  });

  it("offers a button as well, for browsers and keyboards the observer misses", () => {
    const { onLoadMore } = setup();

    fireEvent.click(screen.getByText("Load more"));

    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });

  it("does not claim a cap when the cap is exactly the total", () => {
    setup({ loaded: 2000, total: 2000, maxLoaded: 2000 });

    expect(screen.queryByText(/This view is limited/)).not.toBeInTheDocument();
  });

  it("degrades to the button where IntersectionObserver is unavailable", () => {
    (global as unknown as { IntersectionObserver?: unknown }).IntersectionObserver = undefined;

    const { onLoadMore } = setup();
    fireEvent.click(screen.getByText("Load more"));

    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });
});


test("failed pages pause automatic loading and offer an explicit retry", () => {
  const { onLoadMore } = setup({ error: true });
  expect(observed).toBeNull();
  expect(screen.getByRole("alert")).toHaveTextContent("Couldn’t load more");
  fireEvent.click(screen.getByText("Retry loading more"));
  expect(onLoadMore).toHaveBeenCalledTimes(1);
});

test("duplicate observer callbacks request only one page", () => {
  const { onLoadMore } = setup();
  scrollIntoView();
  scrollIntoView();
  expect(onLoadMore).toHaveBeenCalledTimes(1);
});

test("a finished page stops scrolling even if the count is stale", () => {
  setup({ hasMore: false });
  expect(observed).toBeNull();
  expect(screen.queryByText("Load more")).not.toBeInTheDocument();
});

test("search can pause automatic loading without removing manual loading", () => {
  const { onLoadMore } = setup({ autoLoad: false });
  expect(observed).toBeNull();
  fireEvent.click(screen.getByText("Load more"));
  expect(onLoadMore).toHaveBeenCalledTimes(1);
});

test("unknown totals are reported as loaded rows", () => {
  setup({ total: null, hasMore: true });
  expect(screen.getByText("100 loaded documents")).toBeInTheDocument();
});


test("an exact full page at the known total does not offer an empty next page", () => {
  const { container } = setup({ loaded: 100, total: 100, hasMore: true });
  expect(container).toBeEmptyDOMElement();
  expect(observed).toBeNull();
});
