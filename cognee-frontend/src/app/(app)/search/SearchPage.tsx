"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import UpgradeBanner from "@/ui/elements/UpgradeBanner";
import recallKnowledge from "@/modules/datasets/recallKnowledge";
import {
  absorbRestoredTurns,
  beginTurn,
  dropMessage,
  mergeConversations,
  patchMessage,
  sessionToConversation,
  type ChatMessage,
  type Conversation,
} from "@/modules/searchHistory/conversations";
import { readSearchState, updateSearchState, type SearchState } from "@/modules/searchHistory/conversationStore";
import { useSearchState } from "@/modules/searchHistory/useSearchState";
import { normalizeResults, type SearchResultItem } from "@/modules/searchHistory/normalizeResults";
import { readHistoryCache, writeHistoryCache } from "@/modules/searchHistory/historyCache";
import { readCodeConversations, writeCodeConversations } from "@/modules/searchHistory/codeConversations";
import { listSearchSessions, getSessionDetail, SEARCH_SESSION_PREFIX } from "@/modules/sessions/getSessions";
import { TrackPageView, trackEvent } from "@/modules/analytics";
import BrainSelector from "@/ui/elements/BrainSelector";
import getPodUserId from "@/modules/users/getPodUserId";
import { isInsufficientCreditsError } from "@/utils/insufficientCredits";
import { answerCodeQuestion } from "@/app/(app)/search/codeSearch";
import { isRepositoryDataset, searchTypeOption } from "@/app/(app)/search/searchTypes";
import useSearchType from "@/app/(app)/search/useSearchType";
import CodeSearchHint from "./partials/CodeSearchHint";
import HistorySkeleton from "./partials/HistorySkeleton";
import MarkdownContent from "./partials/MarkdownContent";
import SearchTypeSelector from "./partials/SearchTypeSelector";

// Conversations ARE backend sessions: the conversation id doubles as the
// session_id sent with every recall, so the backend records each Q&A and the
// sidebar survives page refreshes. The prefix is what separates user searches
// from agent sessions — only sessions starting with it are listed here.
const SESSION_PREFIX = SEARCH_SESSION_PREFIX;

// Sidebar history is capped at 20 sessions (see the load effect).
const MAX_HISTORY_SESSIONS = 20;
// Placeholder rows shown before the session list lands and the real count is
// known — enough to read as a list, few enough not to overstate what's coming.
const PRESUMED_HISTORY_ROWS = 3;
// However many are still loading, never show more placeholders than this.
const MAX_SKELETON_ROWS = 6;

// What a pending answer says about itself, per stage the pod streams.
const STAGE_LABELS: Record<string, string> = {
  retrieving: "Searching your data…",
  generating: "Writing…",
};

// ── Date grouping ──

function dateLabel(dateStr: string): string {
  const d = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return "This week";
  if (diffDays < 30) return "This month";
  return "Older";
}

// ── Icons ──

function SidebarIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" /><line x1="9" y1="3" x2="9" y2="21" /></svg>;
}
function PlusIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>;
}
function SendIcon({ active }: { active: boolean }) {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#fff" : "#A1A1AA"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" /></svg>;
}

// ── Main ──

export default function SearchPage() {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { hasAccess, tenantReady } = useTenant();
  const { selectedDataset, datasets } = useFilter();

  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Conversations started or continued in this tab, the open one, and the
  // answer in flight all live outside the component (CLO-846): a navigation
  // unmounts this page, and state kept here closed the chat the user was in —
  // the next question then opened a new session the pod had no history for.
  const instanceId = cogniInstance?.instanceId ?? null;
  const { conversations: sessionConvos, activeId: activeConvoId, inFlightId } = useSearchState(instanceId);
  const isSearching = inFlightId !== null;
  const update = useCallback(
    (change: (state: SearchState) => SearchState) => {
      if (instanceId) updateSearchState(instanceId, change);
    },
    [instanceId],
  );

  // Session-backed history restored from the pod.
  const [historyConvos, setHistoryConvos] = useState<Conversation[]>([]);
  // History rows still on their way, driving the sidebar skeleton. Starts at a
  // nominal count because the session list has to load before the real number
  // is known; each conversation that resolves takes one placeholder with it.
  const [pendingHistory, setPendingHistory] = useState(PRESUMED_HISTORY_ROWS);
  // History could not be read in full, so an empty sidebar is unproven — say so
  // rather than asserting "No conversations yet".
  const [historyFailed, setHistoryFailed] = useState(false);

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Code conversations restored from this browser: the pod records only Text
  // turns, so these would otherwise vanish on every reload. Tagged with the pod
  // they were read for, so nothing is written back until that read has landed —
  // an early write would replace what is stored with an empty list.
  const [codeHistory, setCodeHistory] = useState<{ instanceId: string | null; conversations: Conversation[] }>({
    instanceId: null,
    conversations: [],
  });
  useEffect(() => {
    if (!instanceId) return;
    setCodeHistory({ instanceId, conversations: readCodeConversations(instanceId) });
  }, [instanceId]);

  // The history cache already carries the Code conversations of this visit, so
  // the two restored lists are deduped against each other as well as the store.
  const allConversations = mergeConversations(
    sessionConvos,
    mergeConversations(historyConvos, codeHistory.conversations),
  );

  // Rebuild the sidebar from backend sessions on mount: only this page's own
  // sessions are shown — agent sessions never match the search-UI prefix.
  //
  // Loading history never selects a conversation. Only the user's own click
  // (or the conversation they already had open, restored from the store) leaves
  // the new-chat state — history arrives seconds after first paint, so
  // auto-selecting yanked the user out of the chat they had already started
  // typing in and into an unrelated old one (CLO-714).
  //
  // Gated on tenantReady, not just isInitializing: on a first login the pod is
  // provisioned but not yet confirmed reachable, and firing here returned an
  // empty history that the sidebar reported as "no conversations" — correct
  // only after a refresh, once the init cache made the pod ready immediately.
  // tenantReady is in the deps so the load happens when the pod actually is.
  useEffect(() => {
    if (!cogniInstance || isInitializing || !tenantReady) return;
    let cancelled = false;
    // Aborts the page scan on unmount: `cancelled` only stops the state
    // updates, the requests themselves kept running against the pod.
    const scan = new AbortController();
    // Served from this tab's cache, so bouncing between pages doesn't re-scan
    // the pod. The page writes its own conversations back, so anything the user
    // did here is already in it.
    const cached = readHistoryCache(cogniInstance.instanceId);
    if (cached) {
      setHistoryConvos(cached);
      setPendingHistory(0);
      return;
    }
    (async () => {
      try {
        // Fail closed: without an id there is no way to tell this user's
        // conversations from a teammate's, and showing theirs would also let a
        // follow-up be written under their session.
        const userId = await getPodUserId(cogniInstance);
        if (cancelled) return;
        if (!userId) {
          setHistoryFailed(true);
          setPendingHistory(0);
          return;
        }
        const { sessions: own, complete } = await listSearchSessions(
          cogniInstance,
          MAX_HISTORY_SESSIONS,
          userId,
          scan.signal,
        );
        if (cancelled) return;
        if (!complete) setHistoryFailed(true);
        setPendingHistory(own.length);
        if (own.length === 0) return;
        // Render incrementally: each conversation appears as soon as its own
        // detail request resolves — waiting on Promise.all made the sidebar
        // hang on the slowest of up to 20 responses.
        own.forEach((s) => {
          getSessionDetail(cogniInstance, s.session_id)
            .then((detail) => {
              if (cancelled) return;
              // A detail that failed or timed out drops that conversation from
              // the sidebar; without this the shortened list would be cached
              // and reported as the user's complete history.
              if (!detail) {
                setHistoryFailed(true);
                return;
              }
              // detail.label is the pod's title from the untruncated Q&A list;
              // detail.qas is only the last 20 turns.
              const convo = sessionToConversation(s, detail.qas ?? [], detail.label);
              if (!convo) return;
              setHistoryConvos((prev) =>
                [...prev.filter((c) => c.id !== convo.id), convo]
                  .sort((a, b) => (b.updatedAt ?? "").localeCompare(a.updatedAt ?? "")),
              );
              // A follow-up sent under a remembered id before this arrived
              // started a local copy holding only that turn; give it its past.
              // Same state object when there is nothing to fold in: a store
              // notification re-renders synchronously, which is not free.
              update((state) => {
                const conversations = absorbRestoredTurns(state.conversations, convo);
                return conversations === state.conversations ? state : { ...state, conversations };
              });
            })
            .catch((err) => console.error("Failed to fetch session detail:", err))
            // Placeholder retired whether or not the session produced a row —
            // a failed or questionless one must not leave a skeleton spinning.
            .finally(() => {
              if (!cancelled) setPendingHistory((n) => Math.max(0, n - 1));
            });
        });
      } catch (err) {
        // Not an unavailable endpoint — that is swallowed two layers down and
        // comes back as an incomplete result above. This catches a malformed
        // 200, e.g. rows with no string session_id blowing up the prefix filter.
        console.error("Failed to load search history:", err);
        if (!cancelled) {
          setHistoryFailed(true);
          setPendingHistory(0);
        }
      }
    })();
    return () => {
      cancelled = true;
      scan.abort();
    };
  }, [cogniInstance, isInitializing, tenantReady, update]);

  // Cache only once nothing is in flight: a half-loaded list would be restored
  // as a short history, and an in-flight answer as a spinner that never ends.
  //
  // Only the local list is filtered on message count. A message-less local
  // conversation is a "+" click the user never used, and caching it would bring
  // it back on the next mount as a real history row. Restored conversations
  // must not be filtered the same way: the pod ages a session's Q&A out after
  // its cache TTL but keeps the label, so a real conversation legitimately
  // arrives with no messages, and dropping it here would make it vanish from
  // the sidebar for the whole TTL (the cache read skips the scan).
  useEffect(() => {
    if (!cogniInstance || isSearching || pendingHistory > 0 || historyFailed) return;
    const used = sessionConvos.filter((c) => c.messages.length > 0);
    writeHistoryCache(cogniInstance.instanceId, mergeConversations(used, historyConvos));
  }, [cogniInstance, isSearching, pendingHistory, historyFailed, sessionConvos, historyConvos]);

  // Store this pod's Code conversations whenever no answer is in flight. Merged
  // with what was restored, so a conversation not opened this visit is kept.
  useEffect(() => {
    if (!instanceId || codeHistory.instanceId !== instanceId || isSearching) return;
    writeCodeConversations(instanceId, mergeConversations(sessionConvos, codeHistory.conversations));
  }, [instanceId, codeHistory, isSearching, sessionConvos]);

  // Ensure a dataset is always selected — default to the first one
  const effectiveDataset = selectedDataset || datasets[0] || null;
  const searchDatasetIds = effectiveDataset ? [effectiveDataset.id] : [];

  // Active conversation
  const activeConvo = allConversations.find((c) => c.id === activeConvoId) || null;
  const messages = activeConvo?.messages || [];

  // Unlike newConversation below, keeps what has been typed: switching type
  // mid-conversation is usually about the question already in the box.
  const leaveConversation = useCallback(
    () => update((state) => (state.activeId === null ? state : { ...state, activeId: null })),
    [update],
  );
  const { searchType, changeSearchType } = useSearchType(instanceId, activeConvoId, activeConvo, leaveConversation);

  // A remembered conversation that history, once fully read, does not contain
  // is gone from the pod; forgetting it returns the page to the new-chat state
  // instead of highlighting nothing. Kept while an answer is in flight — the
  // conversation being answered is about to exist.
  useEffect(() => {
    if (!activeConvoId || activeConvo || pendingHistory > 0 || historyFailed || isSearching) return;
    update((state) => (state.activeId === activeConvoId ? { ...state, activeId: null } : state));
  }, [activeConvoId, activeConvo, pendingHistory, historyFailed, isSearching, update]);

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  // Deselects rather than listing an empty "New conversation": the store now
  // outlives the page, so a placeholder row would follow the user around the
  // app until it was used. The session is created by the first question.
  function newConversation() {
    update((state) => (state.activeId === null ? state : { ...state, activeId: null }));
    setInput("");
    setTimeout(() => inputRef.current?.focus(), 100);
  }

  function switchConversation(id: string) {
    update((state) => ({ ...state, activeId: id }));
    setInput("");
    setTimeout(() => inputRef.current?.focus(), 100);
  }

  const handleSend = async (q: string) => {
    if (!q.trim() || !cogniInstance || isSearching) return;
    const podId = cogniInstance.instanceId;
    const query = q.trim();
    setInput("");
    trackEvent({ pageName: "Search", eventName: "search_executed", additionalProperties: { query_length: String(query.length), search_type: searchType } });

    // Continue the open conversation (= session) or start one. A conversation
    // restored from backend history is adopted into the store so the user can
    // continue it — same session id, so the backend appends to it. An id
    // remembered from before a reload is kept even if its history has not
    // loaded yet: the pod has the turns, and that is what the answer builds on.
    const now = new Date().toISOString();
    const convoId = readSearchState(podId).activeId ?? `${SESSION_PREFIX}${Date.now()}`;
    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: "user", content: query, timestamp: now };
    const loadingMsg: ChatMessage = { id: `a-${Date.now()}`, role: "assistant", content: "", timestamp: now, loading: true };
    updateSearchState(podId, (state) => ({
      ...state,
      activeId: convoId,
      inFlightId: convoId,
      conversations: beginTurn(state.conversations, activeConvo, convoId, userMsg, loadingMsg, searchType),
    }));
    // Every update goes through the store, not component state: the answer
    // keeps arriving after a navigation, and a remount picks it up mid-stream.
    const patchAnswer = (patch: (message: ChatMessage) => ChatMessage) =>
      updateSearchState(podId, (state) => ({
        ...state,
        conversations: patchMessage(state.conversations, convoId, loadingMsg.id, patch),
      }));

    try {
      if (searchType === "code") {
        // A lookup in the code graph, not a question put to a model: nothing
        // streams, and the pod has no turn to record, so the conversation is
        // kept in this browser instead (codeConversations.ts).
        const content = await answerCodeQuestion(cogniInstance, query, searchDatasetIds);
        patchAnswer((m) => ({ ...m, content, dataset: effectiveDataset?.name, loading: false, stage: undefined }));
        return;
      }
      // Fails to compile when a third type is added and not handled above,
      // instead of sending it through recall as though it were Text.
      const textOnly: "text" = searchType;
      void textOnly;
      // The conversation's session id is sent with every search so the
      // backend records the Q&A — that's what persists the chat across refreshes.
      // Streamed tokens are a preview; the resolved payload is the answer.
      const data = await recallKnowledge(
        cogniInstance,
        { query, scope: "graph", sessionId: convoId, datasetIds: searchDatasetIds },
        {
          onStage: (stage) => patchAnswer((m) => ({ ...m, stage })),
          onDelta: (text) => patchAnswer((m) => ({ ...m, content: m.content + text })),
          onReset: () => patchAnswer((m) => ({ ...m, content: "" })),
        },
      );
      const resultData = (Array.isArray(data) ? data : []) as SearchResultItem[];
      const content = normalizeResults(resultData);
      const datasetLabel = resultData[0]?.dataset_name || selectedDataset?.name;
      patchAnswer((m) => ({ ...m, content, dataset: datasetLabel, loading: false, stage: undefined }));
    } catch (err) {
      // On insufficient credits, the pod interceptor (services/http/pod.ts)
      // already opened the global modal — drop the placeholder message
      // instead of showing a redundant inline error bubble underneath it.
      if (isInsufficientCreditsError(err)) {
        updateSearchState(podId, (state) => ({
          ...state,
          conversations: dropMessage(state.conversations, convoId, loadingMsg.id),
        }));
      } else {
        const content = err instanceof Error ? err.message : "Search failed";
        patchAnswer((m) => ({ ...m, content, loading: false, stage: undefined, error: true }));
      }
    } finally {
      updateSearchState(podId, (state) => (state.inFlightId === convoId ? { ...state, inFlightId: null } : state));
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend(input);
    }
  };

  const { suggestions, placeholder } = searchTypeOption(searchType);

  const isEmpty = messages.length === 0;

  // Group conversations by date
  const grouped: { label: string; items: Conversation[] }[] = [];
  const labelOrder: string[] = [];
  for (const c of allConversations) {
    const lbl = dateLabel(c.updatedAt);
    if (!labelOrder.includes(lbl)) { labelOrder.push(lbl); grouped.push({ label: lbl, items: [] }); }
    grouped.find((g) => g.label === lbl)!.items.push(c);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <TrackPageView page="Search" />

      {/* Header */}
      <div style={{ padding: "24px 32px 16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexShrink: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <h1 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif' }}>Search</h1>
          <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Ask questions about your knowledge graph and agent memory.</p>
        </div>
      </div>

      {/* Main panel — same dark-glass container as Sessions / Dataset */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden", marginInline: 32, marginBottom: 32, border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12, background: "rgba(0,0,0,0.82)", backdropFilter: "blur(20px)" }}>

      {/* Sidebar */}
      {sidebarOpen && (
        <div style={{ width: 260, borderRight: "1px solid rgba(255,255,255,0.08)", display: "flex", flexDirection: "column", flexShrink: 0 }}>
          {/* Sidebar header */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 16px", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
            <span style={{ fontSize: 13, fontWeight: 700, color: "#EDECEA" }}>History</span>
            <div style={{ display: "flex", gap: 4 }}>
              <button onClick={newConversation} className="cursor-pointer rounded p-1" style={{ background: "none", border: "none", color: "#BC9BFF", display: "flex" }} title="New conversation">
                <PlusIcon />
              </button>
              <button onClick={() => setSidebarOpen(false)} className="cursor-pointer rounded p-1" style={{ background: "none", border: "none", color: "rgba(237,236,234,0.5)", display: "flex" }} title="Close sidebar">
                <SidebarIcon />
              </button>
            </div>
          </div>

          {/* Conversation list */}
          <div style={{ flex: 1, overflowY: "auto", padding: "8px 8px" }}>
            {allConversations.length === 0 && pendingHistory === 0 && (
              <div style={{ padding: "24px 12px", textAlign: "center" }}>
                <span style={{ fontSize: 12, color: "rgba(237,236,234,0.35)" }}>
                  {historyFailed ? "Couldn't load past conversations" : "No conversations yet"}
                </span>
              </div>
            )}
            {grouped.map((group) => (
              <div key={group.label}>
                <span style={{ display: "block", fontSize: 11, fontWeight: 500, color: "rgba(237,236,234,0.35)", padding: "10px 8px 4px", textTransform: "uppercase", letterSpacing: "0.04em" }}>{group.label}</span>
                {group.items.map((c) => (
                  <div
                    key={c.id}
                    onClick={() => switchConversation(c.id)}
                    className="cursor-pointer"
                    style={{
                      display: "flex", alignItems: "center", gap: 8,
                      padding: "8px 10px", borderRadius: 6,
                      background: activeConvoId === c.id ? "rgba(188,155,255,0.20)" : "transparent",
                      transition: "background 150ms",
                    }}
                    onMouseEnter={e => { if (activeConvoId !== c.id) (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.06)"; }}
                    onMouseLeave={e => { if (activeConvoId !== c.id) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
                  >
                    <span style={{
                      flex: 1, fontSize: 13, color: activeConvoId === c.id ? "#BC9BFF" : "rgba(237,236,234,0.7)",
                      fontWeight: activeConvoId === c.id ? 500 : 400,
                      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    }}>
                      {c.title}
                    </span>
                  </div>
                ))}
              </div>
            ))}
            {pendingHistory > 0 && (
              <HistorySkeleton
                rows={Math.min(pendingHistory, MAX_SKELETON_ROWS)}
                showGroupLabel={allConversations.length === 0}
              />
            )}
          </div>
        </div>
      )}

      {/* Main chat area */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        {!hasAccess && (
          <div style={{ padding: "12px 32px 0" }}>
            <UpgradeBanner />
          </div>
        )}

        {/* Sidebar toggle when collapsed */}
        {!sidebarOpen && (
          <div style={{ padding: "10px 12px 0" }}>
            <button onClick={() => setSidebarOpen(true)} className="cursor-pointer rounded p-1.5" style={{ background: "none", border: "none", color: "rgba(237,236,234,0.5)", display: "flex" }} title="Open history">
              <SidebarIcon />
            </button>
          </div>
        )}

        {/* Messages area */}
        <div style={{ flex: 1, overflowY: "auto", padding: "24px 32px" }}>
          {isEmpty && (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 16 }}>
              <h2 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif', textAlign: "center" }}>What are you looking for today?</h2>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", marginTop: 8 }}>
                {suggestions.map((s) => (
                  <button
                    key={s}
                    onClick={() => handleSend(s)}
                    disabled={!hasAccess}
                    className="cursor-pointer transition-colors"
                    style={{ background: "rgba(255,255,255,0.06)", backdropFilter: "blur(8px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 100, padding: "8px 16px", fontSize: 13, color: "rgba(237,236,234,0.8)", fontFamily: "inherit", opacity: hasAccess ? 1 : 0.4, cursor: hasAccess ? "pointer" : "not-allowed" }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {!isEmpty && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16, maxWidth: 800, marginInline: "auto" }}>
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  style={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}
                >
                  <div
                    // Distinguishes a painted answer from a pending one so e2e
                    // can fail on a stuck spinner instead of timing out on the
                    // absence of an error. A streaming answer is still pending:
                    // its text is a preview until the payload lands.
                    data-testid={
                      msg.role === "assistant"
                        ? msg.loading
                          ? "assistant-message-loading"
                          : "assistant-message"
                        : "user-message"
                    }
                    style={{
                      maxWidth: msg.role === "user" ? "70%" : "85%",
                      background: msg.role === "user" ? "#6510F4" : msg.error ? "rgba(239,68,68,0.1)" : "rgba(255,255,255,0.06)",
                      backdropFilter: msg.role !== "user" ? "blur(12px)" : undefined,
                      color: msg.role === "user" ? "#fff" : msg.error ? "#FCA5A5" : "#EDECEA",
                      borderRadius: msg.role === "user" ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
                      padding: msg.role === "user" ? "10px 16px" : "14px 18px",
                      border: msg.role === "user" ? "none" : msg.error ? "1px solid rgba(239,68,68,0.3)" : "1px solid rgba(255,255,255,0.1)",
                    }}
                  >
                    {msg.role === "user" && (
                      <span style={{ fontSize: 14, lineHeight: "22px" }}>{msg.content}</span>
                    )}
                    {msg.role === "assistant" && msg.loading && !msg.content && (
                      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 0" }}>
                        <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#6510F4", animation: "pulse 1.2s ease-in-out infinite" }} />
                        <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#6510F4", animation: "pulse 1.2s ease-in-out infinite 0.2s" }} />
                        <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#6510F4", animation: "pulse 1.2s ease-in-out infinite 0.4s" }} />
                        {msg.stage && STAGE_LABELS[msg.stage] && (
                          <span style={{ fontSize: 12, color: "rgba(237,236,234,0.5)", marginLeft: 4 }}>{STAGE_LABELS[msg.stage]}</span>
                        )}
                      </div>
                    )}
                    {msg.role === "assistant" && msg.loading && msg.content && (
                      <>
                        <MarkdownContent text={msg.content} />
                        <span aria-hidden style={{ display: "inline-block", width: 6, height: 14, borderRadius: 1, background: "#BC9BFF", animation: "pulse 1s ease-in-out infinite", verticalAlign: "text-bottom" }} />
                      </>
                    )}
                    {msg.role === "assistant" && !msg.loading && (
                      <>
                        {msg.dataset && (
                          <span style={{ display: "inline-block", fontSize: 11, fontWeight: 700, letterSpacing: "0.04em", color: "#6510F4", textTransform: "uppercase", marginBottom: 4 }}>
                            {msg.dataset}
                          </span>
                        )}
                        <MarkdownContent text={msg.content} />
                      </>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input area */}
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", padding: "12px 32px 16px" }}>
          <div style={{ maxWidth: 800, marginInline: "auto", display: "flex", flexDirection: "column", gap: 8 }}>
            {/* What a search is scoped to: the type picks the retrieval lane,
                the dataset picks what that lane reads (searchDatasetIds). Both
                belong where the question is typed — they are changed per
                question, not configured once. */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
              {/* Only where there is a repository to find: elsewhere it points
                  at a type that has nothing to search. */}
              {datasets.some((d) => isRepositoryDataset(d.name)) && <CodeSearchHint />}
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: "auto" }}>
                <SearchTypeSelector value={searchType} onChange={changeSearchType} />
                <BrainSelector allowAll={false} align="right" direction="up" />
              </div>
            </div>

            {/* Input bar */}
            <div style={{ display: "flex", alignItems: "center", gap: 12, background: "rgba(255,255,255,0.06)", border: `1px solid ${input ? "#BC9BFF" : "rgba(255,255,255,0.12)"}`, borderRadius: 10, padding: "12px 16px", transition: "border-color 0.2s", opacity: hasAccess ? 1 : 0.4, pointerEvents: hasAccess ? "auto" : "none" }}>
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none" style={{ flexShrink: 0 }}>
                <circle cx="8" cy="8" r="5.5" stroke="rgba(237,236,234,0.35)" strokeWidth="1.5" />
                <path d="M12.5 12.5L16 16" stroke="rgba(237,236,234,0.35)" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={hasAccess ? placeholder : "Subscribe to use search..."}
                rows={1}
                style={{ flex: 1, border: "none", outline: "none", fontSize: 14, color: "#EDECEA", fontFamily: "inherit", background: "transparent", resize: "none", minHeight: 24, maxHeight: 120 }}
                onInput={(e) => { const t = e.target as HTMLTextAreaElement; t.style.height = "24px"; t.style.height = t.scrollHeight + "px"; }}
              />
              <button
                onClick={() => handleSend(input)}
                disabled={!input.trim() || isSearching}
                className="cursor-pointer"
                style={{
                  background: input.trim() && !isSearching ? "#6510F4" : "rgba(255,255,255,0.08)",
                  border: "none", borderRadius: 6, padding: "8px 10px",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  flexShrink: 0, transition: "background 150ms",
                }}
              >
                <SendIcon active={!!input.trim() && !isSearching} />
              </button>
            </div>
          </div>
        </div>
      </div>

      </div>{/* end panel */}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.3; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  );
}
