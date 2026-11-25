import forEachWithConcurrency from "../forEachWithConcurrency";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

describe("forEachWithConcurrency", () => {
  it("never has more than `limit` in flight", async () => {
    const gates = [0, 1, 2, 3, 4].map(() => deferred<string>());
    let inFlight = 0;
    let peak = 0;

    const run = forEachWithConcurrency([0, 1, 2, 3, 4], 2, async (i) => {
      inFlight += 1;
      peak = Math.max(peak, inFlight);
      const value = await gates[i].promise;
      inFlight -= 1;
      return value;
    }, () => {});

    // Let the first workers start before releasing anything.
    await Promise.resolve();
    expect(peak).toBe(2);

    gates.forEach((gate, i) => gate.resolve(`done-${i}`));
    await run;
    expect(peak).toBe(2);
  });

  it("covers every item exactly once", async () => {
    const seen: number[] = [];
    await forEachWithConcurrency([1, 2, 3, 4, 5, 6, 7], 3, async (n) => n * 2, (item) => { seen.push(item); });

    expect(seen.sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it("reports each result against its own item", async () => {
    const results = new Map<number, number>();
    await forEachWithConcurrency([1, 2, 3], 2, async (n) => n * 10, (item, result) => { results.set(item, result); });

    expect(results.get(1)).toBe(10);
    expect(results.get(3)).toBe(30);
  });

  it("does nothing, and does not hang, on an empty list", async () => {
    const onResult = jest.fn();
    await forEachWithConcurrency([], 4, async () => "x", onResult);

    expect(onResult).not.toHaveBeenCalled();
  });
});
