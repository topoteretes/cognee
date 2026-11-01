import {
  MAX_STORED_CODE_CONVERSATIONS,
  MAX_STORED_MESSAGES,
  readCodeConversations,
  writeCodeConversations,
} from "../codeConversations";
import type { ChatMessage, Conversation } from "../conversations";

const POD = "tenant-1";
const KEY = `cognee-search-code-conversations-${POD}`;

function message(i: number, extra: Partial<ChatMessage> = {}): ChatMessage {
  return { id: `m${i}`, role: i % 2 === 0 ? "user" : "assistant", content: `c${i}`, timestamp: "2026-09-30T10:00:00.000Z", ...extra };
}

function conversation(id: string, extra: Partial<Conversation> = {}): Conversation {
  return {
    id,
    title: id,
    messages: [message(0), message(1)],
    createdAt: "2026-09-30T10:00:00.000Z",
    updatedAt: "2026-09-30T10:00:00.000Z",
    searchType: "code",
    ...extra,
  };
}

afterEach(() => {
  localStorage.clear();
  jest.restoreAllMocks();
});

describe("codeConversations", () => {
  it("reads nothing for a pod with nothing stored", () => {
    expect(readCodeConversations(POD)).toEqual([]);
  });

  it("brings back what was written, for that pod only", () => {
    writeCodeConversations(POD, [conversation("search-ui-1")]);

    expect(readCodeConversations(POD)).toEqual([conversation("search-ui-1")]);
    expect(readCodeConversations("tenant-2")).toEqual([]);
  });

  // Text conversations live on the pod; an unused "+" has nothing to restore;
  // a pending answer would come back as a spinner that never ends.
  it("keeps only finished Code conversations that were used", () => {
    writeCodeConversations(POD, [
      conversation("code"),
      conversation("text", { searchType: "text" }),
      conversation("restored", { searchType: undefined }),
      conversation("empty", { messages: [] }),
      conversation("pending", { messages: [message(0), message(1, { loading: true })] }),
    ]);

    // Checked on what is stored, not on what reads back: the reader filters
    // too, and a Text conversation written here would still cost space and
    // sit in the browser for nothing.
    const stored = JSON.parse(localStorage.getItem(KEY) ?? "[]") as Conversation[];
    expect(stored.map((c) => c.id)).toEqual(["code"]);
  });

  it("keeps the most recent conversations, and the latest turns of each", () => {
    const many = Array.from({ length: MAX_STORED_CODE_CONVERSATIONS + 5 }, (_, i) =>
      conversation(`c${i}`, {
        updatedAt: `2026-09-${String(10 + (i % 20)).padStart(2, "0")}T${String(i).padStart(2, "0")}:00:00.000Z`,
        messages: Array.from({ length: MAX_STORED_MESSAGES + 6 }, (_, j) => message(j)),
      }),
    );

    writeCodeConversations(POD, many);
    const stored = readCodeConversations(POD);

    expect(stored).toHaveLength(MAX_STORED_CODE_CONVERSATIONS);
    const newest = [...many].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)).slice(0, MAX_STORED_CODE_CONVERSATIONS);
    expect(stored.map((c) => c.id)).toEqual(newest.map((c) => c.id));
    expect(stored[0].messages).toHaveLength(MAX_STORED_MESSAGES);
    expect(stored[0].messages[MAX_STORED_MESSAGES - 1].id).toBe(`m${MAX_STORED_MESSAGES + 5}`);
  });

  // One bad entry must not cost the rest.
  it("drops a malformed conversation and keeps the others", () => {
    localStorage.setItem(KEY, JSON.stringify([conversation("good"), { id: 7 }, conversation("bad", { messages: [{ role: "x" }] as never })]));

    expect(readCodeConversations(POD).map((c) => c.id)).toEqual(["good"]);
  });

  it("ignores a stored value that is not JSON", () => {
    jest.spyOn(console, "warn").mockImplementation(() => undefined);
    localStorage.setItem(KEY, "{not json");

    expect(readCodeConversations(POD)).toEqual([]);
  });

  // A full quota keeps the newest conversations instead of losing them all.
  it("drops the oldest conversations until the write fits", () => {
    const setItem = Storage.prototype.setItem;
    jest.spyOn(Storage.prototype, "setItem").mockImplementation(function (this: Storage, key: string, value: string) {
      if (value.length > 900) throw new DOMException("full", "QuotaExceededError");
      setItem.call(this, key, value);
    });
    const four = ["a", "b", "c", "d"].map((id, i) => conversation(id, { updatedAt: `2026-09-30T1${i}:00:00.000Z` }));

    writeCodeConversations(POD, four);

    const kept = readCodeConversations(POD).map((c) => c.id);
    expect(kept.length).toBeGreaterThan(0);
    expect(kept.length).toBeLessThan(4);
    expect(kept[0]).toBe("d");
  });
});
