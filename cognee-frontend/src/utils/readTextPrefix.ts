/**
 * Read at most `maxBytes` from a response body and decode it as text.
 *
 * `response.text()` pulls the whole body. When the body is a synced document
 * and all that is wanted from it is a title, that is the document downloaded
 * to render one row. Reading the head of the stream and cancelling the rest
 * stops that without needing the server to honour a range request.
 *
 * Falls back to `text()` where the body is not a readable stream — older
 * runtimes, and the mocked responses in tests.
 */
export default async function readTextPrefix(response: Response, maxBytes: number): Promise<string> {
  const body = response.body;
  if (!body || typeof body.getReader !== "function") return response.text();

  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;

  try {
    while (total < maxBytes) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      chunks.push(value);
      total += value.byteLength;
    }
  } finally {
    // The connection is only released once the body is done with. Cancelling
    // a stream that already ended is a no-op, and a cancel that rejects must
    // not turn a successful read into a failed one.
    try {
      await reader.cancel();
    } catch {
      // Nothing to do: the bytes already read are what this returns.
    }
  }

  const joined = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.byteLength;
  }

  // A cut mid-character decodes to a replacement character rather than
  // throwing, which is the right trade for a prefix: the caller wants the
  // first lines, and the damage is confined to the last one.
  return new TextDecoder("utf-8").decode(joined.subarray(0, maxBytes));
}
