import {
  mergeConversations,
  qaTimestamp,
  sessionToConversation,
  type Conversation,
} from "@/modules/searchHistory/conversations";
import type { SessionRow } from "@/modules/sessions/getSessions";

function session(overrides: Partial<SessionRow> = {}): SessionRow {
  return {
    session_id: "search-ui-1",
    user_id: "user-1",
    dataset_id: null,
    status: "completed",
    effective_status: "completed",
    started_at: "2026-09-01T10:00:00.000Z",
    last_activity_at: "2026-09-01T10:05:00.000Z",
    ended_at: null,
    tokens_in: 0,
    tokens_out: 0,
    cost_usd: 0,
    error_count: 0,
    last_model: null,
    ...overrides,
  };
}

function conversation(overrides: Partial<Conversation> = {}): Conversation {
  return {
    id: "search-ui-1",
    title: "A question",
    messages: [],
    createdAt: "2026-09-01T10:00:00.000Z",
    updatedAt: "2026-09-01T10:00:00.000Z",
    ...overrides,
  };
}

describe("sessionToConversation", () => {
  it("titles the conversation with the first question that was asked", () => {
    const convo = sessionToConversation(session(), [
      { question: "What is in my graph?", answer: "Three entities.", time: "2026-09-01T10:01:00" },
    ]);

    expect(convo?.title).toBe("What is in my graph?");
  });

  // The detail endpoint ships only the last 20 turns, so on a longer
  // conversation the first visible row is not the first question asked. The
  // pod computes its label from the untruncated list.
  it("prefers the backend label over the oldest visible turn", () => {
    const convo = sessionToConversation(
      session(),
      [{ question: "Turn twenty-one", answer: "...", time: "2026-09-01T10:21:00" }],
      "The actual first question",
    );

    expect(convo?.title).toBe("The actual first question");
  });

  it("keeps a conversation whose visible turns are all blank but which has a label", () => {
    const convo = sessionToConversation(
      session(),
      [{ question: "", answer: "orphan", time: "2026-09-01T10:21:00" }],
      "Asked long ago",
    );

    expect(convo?.title).toBe("Asked long ago");
  });

  it("falls back to the visible turns when the label is blank", () => {
    const convo = sessionToConversation(
      session(),
      [{ question: "Only question", answer: "A", time: "2026-09-01T10:01:00" }],
      "   ",
    );

    expect(convo?.title).toBe("Only question");
  });

  it("returns null when there is neither a label nor an answerable turn", () => {
    expect(sessionToConversation(session(), [{ question: "", answer: null, time: null }], null)).toBeNull();
  });

  it("returns null for a session whose rows carry no question", () => {
    const convo = sessionToConversation(session(), [
      { question: "", answer: "orphaned answer", time: "2026-09-01T10:01:00" },
      { question: "   ", answer: null, time: "2026-09-01T10:02:00" },
    ]);

    expect(convo).toBeNull();
  });

  it("returns null for a session with no rows at all", () => {
    expect(sessionToConversation(session(), [])).toBeNull();
  });

  it("drops blank rows but keeps the session when one real question remains", () => {
    const convo = sessionToConversation(session(), [
      { question: "", answer: "orphaned answer", time: "2026-09-01T10:01:00" },
      { question: "Which datasets exist?", answer: "Two.", time: "2026-09-01T10:02:00" },
    ]);

    expect(convo?.messages.map((m) => m.content)).toEqual(["Which datasets exist?", "Two."]);
  });

  it("orders the messages by their recorded time, not their arrival order", () => {
    const convo = sessionToConversation(session(), [
      { question: "Second", answer: "B", time: "2026-09-01T10:02:00" },
      { question: "First", answer: "A", time: "2026-09-01T10:01:00" },
    ]);

    expect(convo?.messages.map((m) => m.content)).toEqual(["First", "A", "Second", "B"]);
  });

  it("shows 'No results found.' only where an answer is genuinely absent", () => {
    const convo = sessionToConversation(session(), [
      { question: "Unanswered?", answer: null, time: "2026-09-01T10:01:00" },
    ]);

    expect(convo?.messages[1].content).toBe("No results found.");
  });
});

describe("qaTimestamp", () => {
  it("reads a naive timestamp as UTC rather than local time", () => {
    expect(qaTimestamp("2026-09-01T10:01:00", "fallback")).toBe("2026-09-01T10:01:00.000Z");
  });

  it("keeps a timestamp that already carries a zone", () => {
    expect(qaTimestamp("2026-09-01T12:01:00+02:00", "fallback")).toBe("2026-09-01T10:01:00.000Z");
  });

  it("falls back when the timestamp is missing", () => {
    expect(qaTimestamp(null, "2026-09-01T00:00:00.000Z")).toBe("2026-09-01T00:00:00.000Z");
  });

  it("falls back when the timestamp cannot be parsed", () => {
    expect(qaTimestamp("not a date", "2026-09-01T00:00:00.000Z")).toBe("2026-09-01T00:00:00.000Z");
  });
});

describe("mergeConversations", () => {
  it("lists a continued conversation once, keeping the local copy", () => {
    const local = conversation({ id: "search-ui-1", title: "Continued locally" });
    const restored = conversation({ id: "search-ui-1", title: "Stale from backend" });

    expect(mergeConversations([local], [restored])).toEqual([local]);
  });

  it("orders conversations newest first", () => {
    const older = conversation({ id: "search-ui-1", updatedAt: "2026-09-01T10:00:00.000Z" });
    const newer = conversation({ id: "search-ui-2", updatedAt: "2026-09-02T10:00:00.000Z" });

    expect(mergeConversations([], [older, newer]).map((c) => c.id)).toEqual([
      "search-ui-2",
      "search-ui-1",
    ]);
  });

  it("returns nothing when neither source has a conversation", () => {
    expect(mergeConversations([], [])).toEqual([]);
  });
});
