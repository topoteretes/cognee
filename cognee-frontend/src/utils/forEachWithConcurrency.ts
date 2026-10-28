/**
 * Run `work` over every item, at most `limit` at a time, reporting each result
 * as it lands.
 *
 * `Promise.all(items.map(...))` opens every request at once. That is fine for
 * a handful and not fine for a list whose length is the size of someone's
 * Google Drive: the tenant pod gets the whole list simultaneously, which is
 * the pattern CLO-976 already had to protect the graph endpoint from.
 *
 * Results arrive through `onResult` rather than as a returned array so a
 * caller can render each one as it resolves, instead of holding every row in a
 * loading state until the slowest request in the batch comes back.
 */
export default async function forEachWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  work: (item: T) => Promise<R>,
  onResult: (item: T, result: R) => void,
): Promise<void> {
  let cursor = 0;

  async function worker(): Promise<void> {
    while (cursor < items.length) {
      // Read and advance in one step: `await` below yields, and two workers
      // reading the same index would do the same item twice.
      const index = cursor;
      cursor += 1;
      const item = items[index];
      onResult(item, await work(item));
    }
  }

  const workers = Array.from({ length: Math.min(Math.max(limit, 1), items.length) }, worker);
  await Promise.all(workers);
}
