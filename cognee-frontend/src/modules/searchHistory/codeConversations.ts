import { getSearchCodeConversations, setSearchCodeConversations } from "@/utils/browserStorage";
import type { ChatMessage, Conversation } from "./conversations";

/**
 * Code conversations, persisted in the browser because the pod cannot hold
 * them: a Code turn is a code-graph lookup that writes no session, and the
 * sessions API offers no way to write one. Text conversations are sessions on
 * the pod and are restored from there; these are restored from here, so they
 * come back after a reload on this browser but not on another device.
 */

// The same reach as the pod-backed history the sidebar lists beside them.
export const MAX_STORED_CODE_CONVERSATIONS = 20;
// A Code answer can list hundreds of entries, so a long conversation keeps its
// most recent turns rather than growing without bound.
export const MAX_STORED_MESSAGES = 20;

function isMessage(value: unknown): value is ChatMessage {
  if (typeof value !== "object" || value === null) return false;
  const m = value as Partial<ChatMessage>;
  return (
    typeof m.id === "string" &&
    (m.role === "user" || m.role === "assistant") &&
    typeof m.content === "string" &&
    typeof m.timestamp === "string"
  );
}

function isCodeConversation(value: unknown): value is Conversation {
  if (typeof value !== "object" || value === null) return false;
  const c = value as Partial<Conversation>;
  return (
    c.searchType === "code" &&
    typeof c.id === "string" &&
    typeof c.title === "string" &&
    typeof c.createdAt === "string" &&
    typeof c.updatedAt === "string" &&
    Array.isArray(c.messages) &&
    c.messages.every(isMessage)
  );
}

/**
 * The stored Code conversations for a pod. Validated entry by entry: the key
 * outlives builds, and one malformed conversation must not take the rest, or
 * the page, down with it.
 */
export function readCodeConversations(instanceId: string): Conversation[] {
  const raw = getSearchCodeConversations(instanceId);
  if (!raw) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    console.warn("[search] stored Code conversations are unreadable and were ignored", error);
    return [];
  }
  return Array.isArray(parsed) ? parsed.filter(isCodeConversation) : [];
}

/** What is worth keeping: finished Code conversations the user actually used. */
function storable(conversations: Conversation[]): Conversation[] {
  return conversations
    .filter((c) => c.searchType === "code" && c.messages.length > 0 && !c.messages.some((m) => m.loading))
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
    .slice(0, MAX_STORED_CODE_CONVERSATIONS)
    .map((c) => ({ ...c, messages: c.messages.slice(-MAX_STORED_MESSAGES) }));
}

/**
 * Replaces what is stored for the pod with the Code conversations among
 * `conversations`. On a full quota it drops the oldest conversations until the
 * write fits, rather than failing and keeping none.
 */
export function writeCodeConversations(instanceId: string, conversations: Conversation[]): void {
  let kept = storable(conversations);
  while (!setSearchCodeConversations(instanceId, JSON.stringify(kept))) {
    if (kept.length === 0) {
      console.warn("[search] Code conversations could not be stored");
      return;
    }
    kept = kept.slice(0, Math.floor(kept.length / 2));
  }
}
