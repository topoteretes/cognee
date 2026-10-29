import {
  createGraphRefreshThrottle,
  IN_FLIGHT_RECHECK_MS,
  MAX_REFRESH_GAP_MS,
  MIN_REFRESH_GAP_MS,
  refreshGapFor,
  STUCK_AFTER_MS,
  type GraphRefreshThrottle,
} from "../graphRefreshThrottle";

// Each refresh is a read that stays in flight until the test settles it, the
// way a slow /visualize/json read holds isFetching true on the real query.
let pendingReads: Array<() => void>;
let refresh: jest.Mock<Promise<void>, []>;
let onStuck: jest.Mock<void, [number]>;
let readFailed: boolean;
let throttle: GraphRefreshThrottle;

function settleRead(): Promise<void> {
  const settle = pendingReads.shift();
  if (!settle) throw new Error("no read in flight");
  settle();
  // Lets the throttle's settle handler record the read's cost.
  return Promise.resolve();
}

async function readTaking(ms: number): Promise<void> {
  jest.advanceTimersByTime(ms);
  await settleRead();
}

beforeEach(() => {
  jest.useFakeTimers();
  pendingReads = [];
  refresh = jest.fn(() => new Promise<void>((resolve) => pendingReads.push(resolve)));
  onStuck = jest.fn();
  readFailed = false;
  throttle = createGraphRefreshThrottle({
    refresh,
    isFetching: () => pendingReads.length > 0,
    lastReadFailed: () => readFailed,
    onStuck,
  });
});

afterEach(() => {
  throttle.dispose();
  jest.useRealTimers();
});

describe("refreshGapFor", () => {
  it("spaces reads by twice what the last one cost", () => {
    expect(refreshGapFor(3_000)).toBe(6_000);
  });

  it("keeps a cheap read at the floor so small graphs grow near-live", () => {
    expect(refreshGapFor(200)).toBe(MIN_REFRESH_GAP_MS);
  });

  it("caps an expensive read at the ceiling so staleness stays bounded", () => {
    expect(refreshGapFor(60_000)).toBe(MAX_REFRESH_GAP_MS);
  });

  it("uses the floor before any read has been measured", () => {
    expect(refreshGapFor(null)).toBe(MIN_REFRESH_GAP_MS);
  });
});

describe("createGraphRefreshThrottle", () => {
  it("refreshes immediately on the first request", () => {
    throttle.request();
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it("follows a cheap read with the trailing refresh one floor gap after it started", async () => {
    throttle.request();
    await readTaking(100);
    throttle.request();

    jest.advanceTimersByTime(MIN_REFRESH_GAP_MS - 100 - 1);
    expect(refresh).toHaveBeenCalledTimes(1);
    jest.advanceTimersByTime(1);
    expect(refresh).toHaveBeenCalledTimes(2);
  });

  it("spaces the trailing refresh by the measured cost of a slower read", async () => {
    throttle.request();
    await readTaking(3_000);
    throttle.request();

    jest.advanceTimersByTime(6_000 - 3_000 - 1);
    expect(refresh).toHaveBeenCalledTimes(1);
    jest.advanceTimersByTime(1);
    expect(refresh).toHaveBeenCalledTimes(2);
  });

  it("folds a burst that lands during a read into one refresh after it", async () => {
    throttle.request();
    for (let i = 0; i < 20; i += 1) {
      jest.advanceTimersByTime(250);
      throttle.request();
    }
    expect(refresh).toHaveBeenCalledTimes(1);

    await settleRead();
    jest.advanceTimersByTime(MAX_REFRESH_GAP_MS);
    expect(refresh).toHaveBeenCalledTimes(2);
  });

  it("does not start a refresh while the previous read is still in flight", async () => {
    throttle.request();
    throttle.request();

    jest.advanceTimersByTime(MAX_REFRESH_GAP_MS * 2);
    expect(refresh).toHaveBeenCalledTimes(1);

    await settleRead();
    jest.advanceTimersByTime(IN_FLIGHT_RECHECK_MS + MAX_REFRESH_GAP_MS);
    expect(refresh).toHaveBeenCalledTimes(2);
  });

  it("spaces the first refresh after a read it did not start, instead of following it at once", () => {
    let foreignReadOut = true;
    throttle = createGraphRefreshThrottle({
      refresh,
      isFetching: () => foreignReadOut || pendingReads.length > 0,
      lastReadFailed: () => readFailed,
      onStuck,
    });
    throttle.request();
    jest.advanceTimersByTime(IN_FLIGHT_RECHECK_MS * 5);
    foreignReadOut = false;

    jest.advanceTimersByTime(IN_FLIGHT_RECHECK_MS);
    expect(refresh).not.toHaveBeenCalled();
    jest.advanceTimersByTime(MAX_REFRESH_GAP_MS);
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it("refreshes immediately again once the gap has passed with no requests", async () => {
    throttle.request();
    await readTaking(100);
    jest.advanceTimersByTime(MIN_REFRESH_GAP_MS);

    throttle.request();
    expect(refresh).toHaveBeenCalledTimes(2);
  });

  it("reports a read stuck in flight well past the request timeout, once", () => {
    throttle.request();
    throttle.request();

    jest.advanceTimersByTime(STUCK_AFTER_MS + IN_FLIGHT_RECHECK_MS * 10);

    expect(onStuck).toHaveBeenCalledTimes(1);
  });

  it("spaces the refresh after a failed read at the ceiling, however fast it failed", async () => {
    throttle.request();
    readFailed = true;
    await readTaking(100);
    throttle.request();

    jest.advanceTimersByTime(MAX_REFRESH_GAP_MS - 100 - 1);
    expect(refresh).toHaveBeenCalledTimes(1);
    jest.advanceTimersByTime(1);
    expect(refresh).toHaveBeenCalledTimes(2);
  });

  it("times a stuck read from when it started, not from the first request queued behind it", () => {
    throttle.request();
    jest.advanceTimersByTime(STUCK_AFTER_MS - IN_FLIGHT_RECHECK_MS);
    throttle.request();

    jest.advanceTimersByTime(IN_FLIGHT_RECHECK_MS);

    expect(onStuck).toHaveBeenCalledTimes(1);
  });

  it("reports a second stuck read in the same mount after the first one cleared", async () => {
    throttle.request();
    throttle.request();
    jest.advanceTimersByTime(STUCK_AFTER_MS);
    await settleRead();
    jest.advanceTimersByTime(IN_FLIGHT_RECHECK_MS);
    throttle.request();

    jest.advanceTimersByTime(STUCK_AFTER_MS + IN_FLIGHT_RECHECK_MS);

    expect(onStuck).toHaveBeenCalledTimes(2);
  });

  it("does not report a slow read that finishes inside the request timeout", async () => {
    throttle.request();
    throttle.request();
    await readTaking(STUCK_AFTER_MS - IN_FLIGHT_RECHECK_MS * 2);
    jest.advanceTimersByTime(MAX_REFRESH_GAP_MS);

    expect(onStuck).not.toHaveBeenCalled();
  });

  it("never refreshes after dispose, even with a trailing refresh pending", async () => {
    throttle.request();
    throttle.request();
    await settleRead();
    throttle.dispose();

    jest.advanceTimersByTime(MAX_REFRESH_GAP_MS * 2);
    throttle.request();
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it("leaves no timer behind once disposed with a trailing refresh pending", () => {
    throttle.request();
    throttle.request();
    expect(jest.getTimerCount()).toBe(1);

    throttle.dispose();

    expect(jest.getTimerCount()).toBe(0);
  });
});
