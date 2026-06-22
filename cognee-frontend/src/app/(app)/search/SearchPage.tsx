"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import UpgradeBanner from "@/ui/elements/UpgradeBanner";
import recallKnowledge from "@/modules/datasets/recallKnowledge";
import getSearchHistory, { type SearchHistoryEntry } from "@/modules/searchHistory/getSearchHistory";
import { listSessions, type SessionRow } from "@/modules/sessions/getSessions";
import { TrackPageView, trackEvent } from "@/modules/analytics";

type SearchScope = "documents" | "agent";

interface SearchResultItem {
  search_result?: string[];
  dataset_id?: string;
  dataset_name?: string;
  question?: string;
  answer?: string;
  text?: string;
  raw?: { value?: string };
  kind?: string;
  origin_function?: string;
  method_return_value?: unknown;
  session_feedback?: string;
  _source?: string;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  dataset?: string;
  timestamp: string;
  loading?: boolean;
  error?: boolean;
}

interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
  /** true = sourced from backend history (read-only in sidebar) */
  fromHistory?: boolean;
}

// ── Result normalization ──

function normalizeResults(data: SearchResultItem[]): string {
  const parts: string[] = [];
  for (const raw of data) {
    const r = raw as SearchResultItem;
    // Recall response format: { text, raw: { value }, kind, ... }
    if (r.text) {
      parts.push(r.text);
    } else if (r.raw?.value) {
      parts.push(typeof r.raw.value === "string" ? r.raw.value : JSON.stringify(r.raw.value));
    } else if (Array.isArray(r.search_result)) {
      for (const x of r.search_result) {
        parts.push(typeof x === "string" ? x : JSON.stringify(x));
      }
    } else if (typeof r.search_result === "string") {
      parts.push(r.search_result);
    } else if (r.answer) {
      parts.push(r.answer);
    } else if (r.session_feedback) {
      parts.push(r.session_feedback);
    } else if (r.question) {
      parts.push(`Q: ${r.question}`);
    } else if (typeof raw === "string") {
      parts.push(raw as unknown as string);
    } else {
      try { parts.push(JSON.stringify(raw)); } catch { parts.push(String(raw)); }
    }
  }
  return parts.join("\n\n") || "No results found.";
}

// ── Plain text result renderer ──

function PlainContent({ text }: { text: string }) {
  return (
    <div style={{ fontSize: 14, color: "#18181B", lineHeight: "22px", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
      {text}
    </div>
  );
}

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

// ── Convert backend history entries → Conversation[] ──

function historyToConversations(entries: SearchHistoryEntry[]): Conversation[] {
  return entries.map((e) => ({
    id: `hist-${e.id}`,
    title: e.query.slice(0, 60),
    fromHistory: true,
    createdAt: e.created_at,
    updatedAt: e.created_at,
    messages: [
      { id: `hu-${e.id}`, role: "user" as const, content: e.query, timestamp: e.created_at },
      { id: `ha-${e.id}`, role: "assistant" as const, content: e.answer || "No results found.", dataset: e.dataset_name, timestamp: e.created_at },
    ],
  }));
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
  const { hasAccess } = useTenant();
  const { selectedDataset, datasets } = useFilter();

  const [input, setInput] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [scope, setScope] = useState<SearchScope>("documents");
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Conversations: in-session (new searches) + backend history
  const [sessionConvos, setSessionConvos] = useState<Conversation[]>([]);
  const [historyConvos, setHistoryConvos] = useState<Conversation[]>([]);
  const [activeConvoId, setActiveConvoId] = useState<string | null>(null);

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Merged conversations: session-local first, then backend history (deduped)
  const allConversations = (() => {
    const sessionIds = new Set(sessionConvos.map((c) => c.title));
    const deduped = historyConvos.filter((h) => !sessionIds.has(h.title));
    return [...sessionConvos, ...deduped];
  })();

  // Fetch sessions and search history on mount (datasets come from FilterContext)
  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    listSessions(cogniInstance, { range: "30d", limit: 10 })
      .then((page) => setSessions(page.sessions))
      .catch(() => {});
    getSearchHistory(cogniInstance)
      .then((entries) => {
        const convos = historyToConversations(entries);
        setHistoryConvos(convos);
        // Select first history entry if nothing is active
        if (convos.length > 0) {
          setActiveConvoId((prev) => prev || convos[0].id);
        }
      })
      .catch(() => {});
  }, [cogniInstance, isInitializing]);

  // Ensure a dataset is always selected — default to the first one
  const effectiveDataset = selectedDataset || datasets[0] || null;
  const searchDatasetIds = effectiveDataset ? [effectiveDataset.id] : [];

  // Active conversation
  const activeConvo = allConversations.find((c) => c.id === activeConvoId) || null;
  const messages = activeConvo?.messages || [];

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  function newConversation() {
    const id = `conv-${Date.now()}`;
    const convo: Conversation = { id, title: "New conversation", messages: [], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
    setSessionConvos((prev) => [convo, ...prev]);
    setActiveConvoId(id);
    setInput("");
    setTimeout(() => inputRef.current?.focus(), 100);
  }

  function switchConversation(id: string) {
    setActiveConvoId(id);
    setInput("");
    setTimeout(() => inputRef.current?.focus(), 100);
  }

  const handleSend = async (q: string) => {
    if (!q.trim() || !cogniInstance || isSearching) return;
    const query = q.trim();
    setInput("");
    trackEvent({ pageName: "Search", eventName: "search_executed", additionalProperties: { query_length: String(query.length) } });

    // Create conversation if none active or if viewing a history entry
    let convoId = activeConvoId;
    const isHistoryEntry = activeConvo?.fromHistory;
    if (!convoId || isHistoryEntry) {
      convoId = `conv-${Date.now()}`;
      const convo: Conversation = { id: convoId, title: query.slice(0, 60), messages: [], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
      setSessionConvos((prev) => [convo, ...prev]);
      setActiveConvoId(convoId);
    }

    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: "user", content: query, timestamp: new Date().toISOString() };
    const loadingMsg: ChatMessage = { id: `a-${Date.now()}`, role: "assistant", content: "", timestamp: new Date().toISOString(), loading: true };

    // Update conversation with user message + loading indicator
    const finalConvoId = convoId;
    setSessionConvos((prev) =>
      prev.map((c) => {
        if (c.id !== finalConvoId) return c;
        const isFirstMessage = c.messages.length === 0;
        return {
          ...c,
          title: isFirstMessage ? query.slice(0, 60) : c.title,
          messages: [...c.messages, userMsg, loadingMsg],
          updatedAt: new Date().toISOString(),
        };
      })
    );
    setIsSearching(true);

    try {
      const mostRecentSessionId = sessions[0]?.session_id;
      const sendScope = scope === "agent" ? ["session", "trace"] : "graph";
      const sendSessionId = scope === "agent" ? mostRecentSessionId : undefined;
      const data = await recallKnowledge(cogniInstance, {
        query,
        scope: sendScope as never,
        sessionId: sendSessionId,
        datasetIds: searchDatasetIds,
      });
      const resultData = (Array.isArray(data) ? data : []) as SearchResultItem[];
      const content = normalizeResults(resultData);
      const datasetLabel = resultData[0]?.dataset_name || selectedDataset?.name;

      setSessionConvos((prev) =>
        prev.map((c) => {
          if (c.id !== finalConvoId) return c;
          return { ...c, messages: c.messages.map((m) => m.id === loadingMsg.id ? { ...m, content, dataset: datasetLabel, loading: false } : m) };
        })
      );
    } catch (err) {
      setSessionConvos((prev) =>
        prev.map((c) => {
          if (c.id !== finalConvoId) return c;
          return { ...c, messages: c.messages.map((m) => m.id === loadingMsg.id ? { ...m, content: err instanceof Error ? err.message : "Search failed", loading: false, error: true } : m) };
        })
      );
    } finally {
      setIsSearching(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend(input);
    }
  };

  const suggestions = [
    "What are the main entities?",
    "Summarize the uploaded documents",
    "What relationships exist in the data?",
  ];

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
    <div style={{ display: "flex", height: "100%", fontFamily: '"Inter", system-ui, sans-serif', background: "#FFFFFF" }}>
      <TrackPageView page="Search" />

      {/* Sidebar */}
      {sidebarOpen && (
        <div style={{ width: 260, borderRight: "1px solid #E4E4E7", background: "#FAFAFA", display: "flex", flexDirection: "column", flexShrink: 0 }}>
          {/* Sidebar header */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 16px", borderBottom: "1px solid #E4E4E7" }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: "#18181B" }}>History</span>
            <div style={{ display: "flex", gap: 4 }}>
              <button onClick={newConversation} className="cursor-pointer hover:bg-cognee-hover rounded p-1" style={{ background: "none", border: "none", color: "#6510F4", display: "flex" }} title="New conversation">
                <PlusIcon />
              </button>
              <button onClick={() => setSidebarOpen(false)} className="cursor-pointer hover:bg-cognee-hover rounded p-1" style={{ background: "none", border: "none", color: "#71717A", display: "flex" }} title="Close sidebar">
                <SidebarIcon />
              </button>
            </div>
          </div>

          {/* Conversation list */}
          <div style={{ flex: 1, overflowY: "auto", padding: "8px 8px" }}>
            {allConversations.length === 0 && (
              <div style={{ padding: "24px 12px", textAlign: "center" }}>
                <span style={{ fontSize: 12, color: "#A1A1AA" }}>No conversations yet</span>
              </div>
            )}
            {grouped.map((group) => (
              <div key={group.label}>
                <span style={{ display: "block", fontSize: 11, fontWeight: 500, color: "#A1A1AA", padding: "10px 8px 4px", textTransform: "uppercase", letterSpacing: "0.04em" }}>{group.label}</span>
                {group.items.map((c) => (
                  <div
                    key={c.id}
                    onClick={() => switchConversation(c.id)}
                    className="cursor-pointer hover:bg-cognee-hover"
                    style={{
                      display: "flex", alignItems: "center", gap: 8,
                      padding: "8px 10px", borderRadius: 6,
                      background: activeConvoId === c.id ? "#F0EDFF" : "transparent",
                      transition: "background 150ms",
                    }}
                  >
                    <span style={{
                      flex: 1, fontSize: 13, color: activeConvoId === c.id ? "#6510F4" : "#3F3F46",
                      fontWeight: activeConvoId === c.id ? 500 : 400,
                      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    }}>
                      {c.title}
                    </span>
                  </div>
                ))}
              </div>
            ))}
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
            <button onClick={() => setSidebarOpen(true)} className="cursor-pointer hover:bg-cognee-hover rounded p-1.5" style={{ background: "none", border: "none", color: "#71717A", display: "flex" }} title="Open history">
              <SidebarIcon />
            </button>
          </div>
        )}

        {/* Messages area */}
        <div style={{ flex: 1, overflowY: "auto", padding: "24px 32px" }}>
          {isEmpty && (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 16 }}>
              <video src="/videos/mascot-wink.mp4" autoPlay loop muted playsInline style={{ width: 192, height: "auto" }} />
              <h2 style={{ fontSize: 20, fontWeight: 300, color: "#18181B", margin: 0, fontFamily: '"TWK Lausanne", system-ui, sans-serif', textAlign: "center" }}>What are you looking for today?</h2>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", marginTop: 8 }}>
                {suggestions.map((s) => (
                  <button
                    key={s}
                    onClick={() => handleSend(s)}
                    disabled={!hasAccess}
                    className="cursor-pointer transition-colors"
                    style={{ background: "#F4F4F5", border: "none", borderRadius: 100, padding: "8px 16px", fontSize: 13, color: "#18181B", fontFamily: "inherit", opacity: hasAccess ? 1 : 0.4, cursor: hasAccess ? "pointer" : "not-allowed" }}
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
                    style={{
                      maxWidth: msg.role === "user" ? "70%" : "85%",
                      background: msg.role === "user" ? "#6510F4" : msg.error ? "#FEF2F2" : "#F9FAFB",
                      color: msg.role === "user" ? "#fff" : msg.error ? "#991B1B" : "#18181B",
                      borderRadius: msg.role === "user" ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
                      padding: msg.role === "user" ? "10px 16px" : "14px 18px",
                      border: msg.role === "user" ? "none" : msg.error ? "1px solid #FECACA" : "1px solid #E5E7EB",
                    }}
                  >
                    {msg.role === "user" && (
                      <span style={{ fontSize: 14, lineHeight: "22px" }}>{msg.content}</span>
                    )}
                    {msg.role === "assistant" && msg.loading && (
                      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 0" }}>
                        <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#6510F4", animation: "pulse 1.2s ease-in-out infinite" }} />
                        <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#6510F4", animation: "pulse 1.2s ease-in-out infinite 0.2s" }} />
                        <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#6510F4", animation: "pulse 1.2s ease-in-out infinite 0.4s" }} />
                      </div>
                    )}
                    {msg.role === "assistant" && !msg.loading && (
                      <>
                        {msg.dataset && (
                          <span style={{ display: "inline-block", fontSize: 11, fontWeight: 600, letterSpacing: "0.04em", color: "#6510F4", textTransform: "uppercase", marginBottom: 4 }}>
                            {msg.dataset}
                          </span>
                        )}
                        <PlainContent text={msg.content} />
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
        <div style={{ background: "#fff", padding: "12px 32px 16px" }}>
          <div style={{ maxWidth: 800, marginInline: "auto", display: "flex", flexDirection: "column", gap: 8 }}>
            {/* Controls: scope pills — dataset selected via breadcrumb */}
            <div style={{ display: "flex", alignItems: "center" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                {(["documents", "agent"] as const).map((s) => {
                  const active = scope === s;
                  const label = s === "documents" ? "Company Brain" : "Agent Memory";
                  return (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setScope(s)}
                      className="cursor-pointer"
                      style={{
                        background: active ? "#18181B" : "#FFFFFF",
                        color: active ? "#FFFFFF" : "#3F3F46",
                        border: active ? "none" : "1px solid #E4E4E7",
                        borderRadius: 100, paddingBlock: 4, paddingInline: 10,
                        fontSize: 11, lineHeight: "16px", fontFamily: "inherit",
                        cursor: "pointer",
                      }}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Input bar */}
            <div style={{ display: "flex", alignItems: "center", gap: 12, background: "#fff", border: `1px solid ${input ? "#6510F4" : "#EEEEEE"}`, borderRadius: 10, padding: "12px 16px", transition: "border-color 0.2s", opacity: hasAccess ? 1 : 0.4, pointerEvents: hasAccess ? "auto" : "none" }}>
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none" style={{ flexShrink: 0 }}>
                <circle cx="8" cy="8" r="5.5" stroke="#A1A1AA" strokeWidth="1.5" />
                <path d="M12.5 12.5L16 16" stroke="#A1A1AA" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={hasAccess ? "Ask a question about your data..." : "Subscribe to use search..."}
                rows={1}
                style={{ flex: 1, border: "none", outline: "none", fontSize: 14, color: "#18181B", fontFamily: "inherit", background: "transparent", resize: "none", minHeight: 24, maxHeight: 120 }}
                onInput={(e) => { const t = e.target as HTMLTextAreaElement; t.style.height = "24px"; t.style.height = t.scrollHeight + "px"; }}
              />
              <button
                onClick={() => handleSend(input)}
                disabled={!input.trim() || isSearching}
                className="cursor-pointer"
                style={{
                  background: input.trim() && !isSearching ? "#6510F4" : "#E4E4E7",
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

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.3; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  );
}
