import type { SessionRow } from "../sessions/getSessions";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  dataset?: string;
  timestamp: string;
  loading?: boolean;
  error?: boolean;
  /** Where a pending answer is: `retrieving` or `generating` (streamed recall). */
  stage?: string;
}

export const TITLE_LENGTH = 60;

/**
 * Which retriever a conversation's questions go to. A conversation keeps the
 * type it started with: Text turns are pod sessions and Code turns are not
 * recorded anywhere, so a conversation holding both would be neither.
 */
export type ConversationSearchType = "text" | "code";

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
  /** Absent on every conversation restored from the pod, all of which are Text. */
  searchType?: ConversationSearchType;
}

export function qaTimestamp(time: string | null, fallback: string): string {
  if (!time) return fallback;
  // QA timestamps arrive as naive UTC — append Z so Date parses them as UTC.
  const iso = /Z$|[+-]\d{2}:?\d{2}$/.test(time) ? time : `${time}Z`;
  const ms = new Date(iso).getTime();
  return Number.isNaN(ms) ? fallback : new Date(ms).toISOString();
}

/**
 * Build a sidebar conversation from a backend session and its Q&A rows.
 *
 * `label` is the pod's own title for the session, computed from the full Q&A
 * list. It must win over the rows: the detail endpoint ships only the last 20
 * turns, so on a longer conversation qas[0] is not the first question asked.
 *
 * Returns null only when there is nothing to show at all — no label and no
 * answerable row. A long conversation whose visible tail happens to be blank
 * still belongs in the sidebar, so the label alone is enough to keep it.
 */
export function sessionToConversation(
  s: SessionRow,
  qasRaw: Record<string, unknown>[],
  label?: string | null,
): Conversation | null {
  const fallbackTs = s.started_at ?? new Date().toISOString();
  const qas = qasRaw
    .map((qa) => {
      const row = qa as { question?: unknown; answer?: unknown; time?: unknown };
      return {
        question: String(row.question ?? "").trim(),
        answer: row.answer ? String(row.answer) : null,
        time: row.time ? String(row.time) : null,
      };
    })
    .filter((qa) => qa.question)
    .sort((a, b) => (a.time ?? "").localeCompare(b.time ?? ""));
  const title = (label ?? "").trim() || qas[0]?.question || "";
  if (!title) return null;
  const messages: ChatMessage[] = qas.flatMap((qa, i) => {
    const ts = qaTimestamp(qa.time, fallbackTs);
    return [
      { id: `${s.session_id}-u${i}`, role: "user" as const, content: qa.question, timestamp: ts },
      { id: `${s.session_id}-a${i}`, role: "assistant" as const, content: qa.answer || "No results found.", timestamp: ts },
    ];
  });
  return {
    id: s.session_id,
    title: title.slice(0, TITLE_LENGTH),
    createdAt: fallbackTs,
    updatedAt: s.last_activity_at ?? fallbackTs,
    messages,
  };
}

/**
 * Sidebar list: conversations started in this visit first, then the ones
 * restored from the backend, deduped by id — a restored conversation the user
 * has continued lives in `local` and must not appear twice. Newest first so the
 * date grouping stays coherent.
 */
export function mergeConversations(local: Conversation[], restored: Conversation[]): Conversation[] {
  const localIds = new Set(local.map((c) => c.id));
  return [...local, ...restored.filter((c) => !localIds.has(c.id))].sort((a, b) =>
    (b.updatedAt ?? "").localeCompare(a.updatedAt ?? ""),
  );
}

/**
 * Append a question and its pending answer to the conversation `id` in
 * `local`. A conversation this visit has not touched yet is adopted from
 * `seed` (the restored copy the user is continuing) so its earlier turns stay
 * on screen; with no seed either, a new one is started under `id` — which may
 * be an id remembered from before a reload, so the pod appends to that session.
 * Only a new conversation takes `searchType`; one that exists keeps its own.
 */
export function beginTurn(
  local: Conversation[],
  seed: Conversation | null,
  id: string,
  question: ChatMessage,
  pending: ChatMessage,
  searchType: ConversationSearchType = "text",
): Conversation[] {
  const existing = local.find((c) => c.id === id);
  const base: Conversation = existing ?? seed ?? {
    id,
    title: "",
    messages: [],
    createdAt: pending.timestamp,
    updatedAt: pending.timestamp,
    searchType,
  };
  const started: Conversation = {
    ...base,
    title: base.messages.length === 0 ? question.content.slice(0, TITLE_LENGTH) : base.title,
    messages: [...base.messages, question, pending],
    updatedAt: pending.timestamp,
  };
  return existing ? local.map((c) => (c.id === id ? started : c)) : [started, ...local];
}

export function patchMessage(
  local: Conversation[],
  conversationId: string,
  messageId: string,
  patch: (message: ChatMessage) => ChatMessage,
): Conversation[] {
  return local.map((c) =>
    c.id !== conversationId
      ? c
      : { ...c, messages: c.messages.map((m) => (m.id === messageId ? patch(m) : m)) },
  );
}

export function dropMessage(local: Conversation[], conversationId: string, messageId: string): Conversation[] {
  return local.map((c) =>
    c.id !== conversationId ? c : { ...c, messages: c.messages.filter((m) => m.id !== messageId) },
  );
}

/** Groups a message list into turns: each user message with the answers that follow it. */
function turns(messages: ChatMessage[]): ChatMessage[][] {
  const grouped: ChatMessage[][] = [];
  for (const message of messages) {
    if (message.role === "user" || grouped.length === 0) grouped.push([message]);
    else grouped[grouped.length - 1].push(message);
  }
  return grouped;
}

/**
 * Fold a restored conversation's turns into this visit's copy of it, when
 * there is one. The local copy wins the sidebar merge, so a follow-up sent
 * under a remembered id before its history had loaded would otherwise hide
 * every earlier turn. Restored turns are older and go first; one whose
 * question the local copy already asked is the pod's record of this visit's
 * own turn, not a new one. Returns `local` itself when nothing changes.
 */
export function absorbRestoredTurns(local: Conversation[], restored: Conversation): Conversation[] {
  const current = local.find((c) => c.id === restored.id);
  if (!current) return local;
  const asked = new Set(current.messages.filter((m) => m.role === "user").map((m) => m.content));
  const missing = turns(restored.messages)
    .filter((turn) => turn[0].role !== "user" || !asked.has(turn[0].content))
    .flat();
  if (missing.length === 0) return local;
  const merged: Conversation = {
    ...current,
    createdAt: restored.createdAt,
    messages: [...missing, ...current.messages],
  };
  return local.map((c) => (c.id === restored.id ? merged : c));
}
