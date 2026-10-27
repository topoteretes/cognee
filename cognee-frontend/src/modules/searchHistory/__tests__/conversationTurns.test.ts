import {
  absorbRestoredTurns,
  beginTurn,
  dropMessage,
  patchMessage,
  type ChatMessage,
  type Conversation,
} from "@/modules/searchHistory/conversations";

const TS = "2026-09-03T11:00:00.000Z";

function user(id: string, content: string): ChatMessage {
  return { id, role: "user", content, timestamp: TS };
}

function assistant(id: string, content: string, extra: Partial<ChatMessage> = {}): ChatMessage {
  return { id, role: "assistant", content, timestamp: TS, ...extra };
}

function conversation(id: string, messages: ChatMessage[], title = "A title"): Conversation {
  return { id, title, messages, createdAt: TS, updatedAt: TS };
}

const QUESTION = user("u-1", "What is in my graph?");
const PENDING = assistant("a-1", "", { loading: true });

describe("beginTurn", () => {
  it("starts a conversation under the given id, titled by the question", () => {
    const local = beginTurn([conversation("other", [])], null, "search-ui-1", QUESTION, PENDING);

    expect(local.map((c) => c.id)).toEqual(["search-ui-1", "other"]);
    expect(local[0].title).toBe("What is in my graph?");
    expect(local[0].messages).toEqual([QUESTION, PENDING]);
  });

  it("appends to a conversation this visit already holds and keeps its title", () => {
    const existing = conversation("search-ui-1", [user("u-0", "earlier"), assistant("a-0", "answer")], "earlier");

    const local = beginTurn([existing], null, "search-ui-1", QUESTION, PENDING);

    expect(local).toHaveLength(1);
    expect(local[0].title).toBe("earlier");
    expect(local[0].messages.map((m) => m.id)).toEqual(["u-0", "a-0", "u-1", "a-1"]);
  });

  it("adopts the restored copy the user is continuing", () => {
    const restored = conversation("search-ui-1", [user("search-ui-1-u0", "earlier"), assistant("search-ui-1-a0", "answer")]);

    const local = beginTurn([], restored, "search-ui-1", QUESTION, PENDING);

    expect(local[0].title).toBe("A title");
    expect(local[0].messages.map((m) => m.id)).toEqual(["search-ui-1-u0", "search-ui-1-a0", "u-1", "a-1"]);
  });

  it("retitles an unused new conversation with its first question", () => {
    const local = beginTurn([conversation("search-ui-1", [], "New conversation")], null, "search-ui-1", QUESTION, PENDING);

    expect(local[0].title).toBe("What is in my graph?");
  });

  it("gives a new conversation the type it was started with", () => {
    expect(beginTurn([], null, "search-ui-1", QUESTION, PENDING, "code")[0].searchType).toBe("code");
    expect(beginTurn([], null, "search-ui-1", QUESTION, PENDING)[0].searchType).toBe("text");
  });

  // A conversation holding both would be neither a pod session nor a local one.
  it("never changes the type of a conversation already under way", () => {
    const existing: Conversation = { ...conversation("search-ui-1", [user("u-0", "retry"), assistant("a-0", "x")]), searchType: "code" };

    expect(beginTurn([existing], null, "search-ui-1", QUESTION, PENDING, "text")[0].searchType).toBe("code");
  });

  it("leaves a restored conversation untyped, which reads as Text", () => {
    const restored = conversation("search-ui-1", [user("search-ui-1-u0", "earlier")]);

    expect(beginTurn([], restored, "search-ui-1", QUESTION, PENDING, "code")[0].searchType).toBeUndefined();
  });
});

describe("patchMessage / dropMessage", () => {
  const local = [conversation("search-ui-1", [QUESTION, PENDING]), conversation("search-ui-2", [PENDING])];

  it("patches one message in one conversation", () => {
    const next = patchMessage(local, "search-ui-1", "a-1", (m) => ({ ...m, content: "done", loading: false }));

    expect(next[0].messages[1]).toEqual(assistant("a-1", "done", { loading: false }));
    expect(next[1].messages[0]).toEqual(PENDING);
  });

  it("drops one message from one conversation", () => {
    const next = dropMessage(local, "search-ui-1", "a-1");

    expect(next[0].messages).toEqual([QUESTION]);
    expect(next[1].messages).toEqual([PENDING]);
  });
});

describe("absorbRestoredTurns", () => {
  const restored = conversation("search-ui-1", [
    user("search-ui-1-u0", "first"),
    assistant("search-ui-1-a0", "one"),
    user("search-ui-1-u1", "second"),
    assistant("search-ui-1-a1", "two"),
  ]);

  it("leaves the list untouched when this visit has no copy of the conversation", () => {
    const local = [conversation("search-ui-9", [QUESTION])];

    expect(absorbRestoredTurns(local, restored)).toBe(local);
  });

  it("prepends the past to a copy that started from a follow-up", () => {
    const local = [conversation("search-ui-1", [user("u-5", "third"), assistant("a-5", "three")])];

    const next = absorbRestoredTurns(local, restored);

    expect(next[0].messages.map((m) => m.content)).toEqual(["first", "one", "second", "two", "third", "three"]);
    expect(next[0].createdAt).toBe(restored.createdAt);
  });

  it("skips a restored turn the local copy already asked", () => {
    const local = [conversation("search-ui-1", [user("u-5", "second"), assistant("a-5", "two, again")])];

    const next = absorbRestoredTurns(local, restored);

    expect(next[0].messages.map((m) => m.content)).toEqual(["first", "one", "second", "two, again"]);
  });

  it("returns the same list when every restored turn is already present", () => {
    const local = [conversation("search-ui-1", restored.messages)];

    expect(absorbRestoredTurns(local, restored)).toBe(local);
  });
});
