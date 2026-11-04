import {
  clearHistoryCache,
  readHistoryCache,
  writeHistoryCache,
} from "@/modules/searchHistory/historyCache";
import type { Conversation } from "@/modules/searchHistory/conversations";

const NOW = new Date("2026-09-03T12:00:00.000Z").getTime();
const FIVE_MINUTES = 5 * 60_000;

function conversation(overrides: Partial<Conversation> = {}): Conversation {
  return {
    id: "search-ui-1",
    title: "A question",
    messages: [
      { id: "u0", role: "user", content: "A question", timestamp: "2026-09-03T11:00:00.000Z" },
      { id: "a0", role: "assistant", content: "An answer", timestamp: "2026-09-03T11:00:00.000Z" },
    ],
    createdAt: "2026-09-03T11:00:00.000Z",
    updatedAt: "2026-09-03T11:00:00.000Z",
    ...overrides,
  };
}

describe("historyCache", () => {
  let nowSpy: jest.SpyInstance;

  beforeEach(() => {
    nowSpy = jest.spyOn(Date, "now").mockReturnValue(NOW);
  });

  afterEach(() => {
    nowSpy.mockRestore();
    clearHistoryCache();
  });

  it("returns nothing for a pod it has not seen", () => {
    expect(readHistoryCache("instance-1")).toBeNull();
  });

  it("returns what was stored for that pod", () => {
    const conversations = [conversation()];
    writeHistoryCache("instance-1", conversations);

    expect(readHistoryCache("instance-1")).toEqual(conversations);
  });

  it("keeps pods apart", () => {
    writeHistoryCache("instance-1", [conversation()]);

    expect(readHistoryCache("instance-2")).toBeNull();
  });

  it("still serves an entry just inside the TTL", () => {
    writeHistoryCache("instance-1", [conversation()]);
    nowSpy.mockReturnValue(NOW + FIVE_MINUTES - 1);

    expect(readHistoryCache("instance-1")).toHaveLength(1);
  });

  it("drops an entry once the TTL has passed", () => {
    writeHistoryCache("instance-1", [conversation()]);
    nowSpy.mockReturnValue(NOW + FIVE_MINUTES + 1);

    expect(readHistoryCache("instance-1")).toBeNull();
  });

  // A cached spinner would be restored as an answer that never arrives.
  it("refuses to store a conversation with an answer still in flight", () => {
    writeHistoryCache("instance-1", [
      conversation({
        messages: [
          { id: "u0", role: "user", content: "Pending?", timestamp: "2026-09-03T11:00:00.000Z" },
          { id: "a0", role: "assistant", content: "", timestamp: "2026-09-03T11:00:00.000Z", loading: true },
        ],
      }),
    ]);

    expect(readHistoryCache("instance-1")).toBeNull();
  });

  // "No conversations" and "asked too early" look identical, and a cache hit
  // short-circuits the reload that would have corrected it.
  it("refuses to store an empty history", () => {
    writeHistoryCache("instance-1", []);

    expect(readHistoryCache("instance-1")).toBeNull();
  });

  it("keeps an earlier real history rather than being emptied", () => {
    writeHistoryCache("instance-1", [conversation()]);
    writeHistoryCache("instance-1", []);

    expect(readHistoryCache("instance-1")).toHaveLength(1);
  });

  // The pod drops a session's Q&A after its 7-day TTL but keeps the label, so a
  // real conversation reaches the cache with a title and no messages. Filtering
  // on message count here would drop it for the whole TTL, since a cache read
  // skips the scan that would otherwise find it again. Keeping unused "+" rows
  // out is the caller's job — only it knows which list a conversation is from.
  it("stores a conversation whose turns the pod has aged out", () => {
    writeHistoryCache("instance-1", [conversation({ title: "An aged-out question", messages: [] })]);

    expect(readHistoryCache("instance-1")?.map((c) => c.title)).toEqual(["An aged-out question"]);
  });

  it("stores an aged-out conversation alongside ones that still have turns", () => {
    writeHistoryCache("instance-1", [
      conversation({ id: "search-ui-2", title: "An aged-out question", messages: [] }),
      conversation(),
    ]);

    expect(readHistoryCache("instance-1")?.map((c) => c.id)).toEqual(["search-ui-2", "search-ui-1"]);
  });

  // Otherwise every visit pushes the refetch out and the TTL never fires.
  it("expires from the first write, not the most recent one", () => {
    writeHistoryCache("instance-1", [conversation()]);
    nowSpy.mockReturnValue(NOW + FIVE_MINUTES - 1);
    writeHistoryCache("instance-1", [conversation()]);
    nowSpy.mockReturnValue(NOW + FIVE_MINUTES + 1);

    expect(readHistoryCache("instance-1")).toBeNull();
  });

  it("replaces an earlier entry for the same pod", () => {
    writeHistoryCache("instance-1", [conversation({ title: "Old" })]);
    writeHistoryCache("instance-1", [conversation({ title: "New" })]);

    expect(readHistoryCache("instance-1")?.map((c) => c.title)).toEqual(["New"]);
  });
});
