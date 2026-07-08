"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import UpgradeBanner from "@/ui/elements/UpgradeBanner";
import recallKnowledge from "@/modules/datasets/recallKnowledge";
import getSearchHistory, { type SearchHistoryEntry } from "@/modules/searchHistory/getSearchHistory";
import { listSessions, getSessionDetail, SEARCH_SESSION_PREFIX, type SessionRow } from "@/modules/sessions/getSessions";
import { TrackPageView, trackEvent } from "@/modules/analytics";
import BrainSelector from "@/ui/elements/BrainSelector";

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

// Conversations ARE backend sessions: the conversation id doubles as the
// session_id sent with every recall, so the backend records each Q&A and the
// sidebar survives page refreshes. The prefix is what separates user searches
// from agent sessions — only sessions starting with it are listed here.
const SESSION_PREFIX = SEARCH_SESSION_PREFIX;

interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
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

// ── Markdown result renderer ──

function MarkdownContent({ text }: { text: string }) {
  return (
    <div style={{ fontSize: 14, color: "#EDECEA", lineHeight: "22px", wordBreak: "break-word", minWidth: 0 }}>
      <ReactMarkdown
        components={{
          p: ({ children }) => (
            <p style={{ margin: "0 0 10px", fontSize: 14, lineHeight: "22px", color: "#EDECEA" }}>{children}</p>
          ),
          h1: ({ children }) => (
            <h1 style={{ fontSize: 18, fontWeight: 700, margin: "0 0 8px", color: "#EDECEA", lineHeight: "26px" }}>{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 8px", color: "#EDECEA", lineHeight: "24px" }}>{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 style={{ fontSize: 14, fontWeight: 700, margin: "0 0 6px", color: "#EDECEA" }}>{children}</h3>
          ),
          strong: ({ children }) => (
            <strong style={{ fontWeight: 700, color: "#EDECEA" }}>{children}</strong>
          ),
          em: ({ children }) => (
            <em style={{ fontStyle: "italic" }}>{children}</em>
          ),
          code: ({ className, children }) => {
            const isBlock = Boolean(className?.startsWith("language-"));
            if (isBlock) {
              return (
                <code style={{
                  display: "block", background: "rgba(0,0,0,0.4)", color: "#CDD6F4",
                  padding: "12px 16px", fontSize: 12,
                  fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace',
                  overflowX: "auto", lineHeight: "20px", whiteSpace: "pre",
                }}>
                  {children}
                </code>
              );
            }
            return (
              <code style={{
                background: "rgba(255,255,255,0.1)", borderRadius: 3, padding: "1px 5px",
                fontSize: 12, fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace', color: "#CBA6F7",
              }}>
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre style={{ margin: "0 0 10px", borderRadius: 8, overflow: "hidden", background: "rgba(0,0,0,0.4)" }}>{children}</pre>
          ),
          ul: ({ children }) => (
            <ul style={{ margin: "0 0 10px", paddingLeft: 20, lineHeight: "22px" }}>{children}</ul>
          ),
          ol: ({ children }) => (
            <ol style={{ margin: "0 0 10px", paddingLeft: 20, lineHeight: "22px" }}>{children}</ol>
          ),
          li: ({ children }) => (
            <li style={{ fontSize: 14, color: "#EDECEA", marginBottom: 4 }}>{children}</li>
          ),
          blockquote: ({ children }) => (
            <blockquote style={{ borderLeft: "3px solid rgba(188,155,255,0.35)", paddingLeft: 12, margin: "0 0 10px", color: "rgba(237,236,234,0.6)" }}>{children}</blockquote>
          ),
          hr: () => (
            <hr style={{ border: "none", borderTop: "1px solid rgba(255,255,255,0.1)", margin: "12px 0" }} />
          ),
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: "#BC9BFF", textDecoration: "underline" }}>{children}</a>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
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

// ── Convert legacy GET /v1/search entries → read-only Conversation[] ──
// These predate session-backed conversations; continuing one forks into a
// new session instead of appending (legacy entries cannot grow).

const LEGACY_PREFIX = "hist-";

function legacyToConversations(entries: SearchHistoryEntry[]): Conversation[] {
  return entries.map((e) => ({
    id: `${LEGACY_PREFIX}${e.id}`,
    title: e.query.slice(0, 60),
    createdAt: e.created_at,
    updatedAt: e.created_at,
    messages: [
      { id: `hu-${e.id}`, role: "user" as const, content: e.query, timestamp: e.created_at },
      { id: `ha-${e.id}`, role: "assistant" as const, content: e.answer || "No results found.", dataset: e.dataset_name, timestamp: e.created_at },
    ],
  }));
}

// ── Convert a backend session (QA entries) → Conversation ──

function qaTimestamp(time: string | null, fallback: string): string {
  if (!time) return fallback;
  // QA timestamps arrive as naive UTC — append Z so Date parses them as UTC.
  const iso = /Z$|[+-]\d{2}:?\d{2}$/.test(time) ? time : `${time}Z`;
  const ms = new Date(iso).getTime();
  return Number.isNaN(ms) ? fallback : new Date(ms).toISOString();
}

function sessionToConversation(s: SessionRow, qasRaw: Record<string, unknown>[]): Conversation | null {
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
  if (qas.length === 0) return null;
  const messages: ChatMessage[] = qas.flatMap((qa, i) => {
    const ts = qaTimestamp(qa.time, fallbackTs);
    return [
      { id: `${s.session_id}-u${i}`, role: "user" as const, content: qa.question, timestamp: ts },
      { id: `${s.session_id}-a${i}`, role: "assistant" as const, content: qa.answer || "No results found.", timestamp: ts },
    ];
  });
  return {
    id: s.session_id,
    title: qas[0].question.slice(0, 60),
    createdAt: fallbackTs,
    updatedAt: s.last_activity_at ?? fallbackTs,
    messages,
  };
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
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Conversations: local (this visit) + session-backed history + legacy entries
  const [sessionConvos, setSessionConvos] = useState<Conversation[]>([]);
  const [historyConvos, setHistoryConvos] = useState<Conversation[]>([]);
  const [legacyConvos, setLegacyConvos] = useState<Conversation[]>([]);
  const [activeConvoId, setActiveConvoId] = useState<string | null>(null);

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Merged conversations: local (this visit), persisted sessions (deduped by
  // id — a continued session lives in sessionConvos), then legacy entries
  // (deduped by title against session-backed conversations). Sorted newest
  // first so date grouping stays coherent across sources.
  const allConversations = (() => {
    const localIds = new Set(sessionConvos.map((c) => c.id));
    const sessionBacked = [...sessionConvos, ...historyConvos.filter((h) => !localIds.has(h.id))];
    const sessionTitles = new Set(sessionBacked.map((c) => c.title));
    const legacy = legacyConvos.filter((l) => !sessionTitles.has(l.title));
    return [...sessionBacked, ...legacy].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  })();

  // Rebuild the sidebar from backend sessions on mount: only this page's own
  // sessions (SESSION_PREFIX) are shown — agent sessions never match.
  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    let cancelled = false;
    // Legacy single-Q&A history (pre-session searches) — read-only sidebar
    // entries until they age out of the backend.
    getSearchHistory(cogniInstance)
      .then((entries) => {
        if (cancelled) return;
        const convos = legacyToConversations(entries);
        setLegacyConvos(convos);
        if (convos.length > 0) setActiveConvoId((prev) => prev || convos[0].id);
      })
      .catch(() => {});
    (async () => {
      try {
        const page = await listSessions(cogniInstance, { range: "30d", limit: 50 });
        if (cancelled) return;
        setSessions(page.sessions);
        const own = page.sessions
          .filter((s) => s.session_id.startsWith(SESSION_PREFIX))
          .slice(0, 20);
        if (own.length === 0) return;
        // Render incrementally: each conversation appears as soon as its own
        // detail request resolves — waiting on Promise.all made the sidebar
        // hang on the slowest of up to 20 responses.
        own.forEach((s) => {
          getSessionDetail(cogniInstance, s.session_id)
            .then((detail) => {
              if (cancelled || !detail) return;
              const convo = sessionToConversation(s, detail.qas ?? []);
              if (!convo) return;
              setHistoryConvos((prev) =>
                [...prev.filter((c) => c.id !== convo.id), convo]
                  .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)),
              );
              setActiveConvoId((prev) => prev || convo.id);
            })
            .catch(() => {});
        });
      } catch {
        // Sessions endpoint unavailable — sidebar just starts empty.
      }
    })();
    return () => { cancelled = true; };
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
    const id = `${SESSION_PREFIX}${Date.now()}`;
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

    // Create a conversation (= session) if none is active. A conversation
    // restored from backend history is adopted into local state so the user
    // can continue it — same session id, so the backend appends to it.
    // Legacy (pre-session) entries are read-only: continuing one forks into
    // a fresh session conversation instead.
    let convoId = activeConvoId;
    if (!convoId || !activeConvo || activeConvo.id.startsWith(LEGACY_PREFIX)) {
      convoId = `${SESSION_PREFIX}${Date.now()}`;
      const convo: Conversation = { id: convoId, title: query.slice(0, 60), messages: [], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
      setSessionConvos((prev) => [convo, ...prev]);
      setActiveConvoId(convoId);
    } else if (!sessionConvos.some((c) => c.id === convoId)) {
      const adopted = activeConvo;
      setSessionConvos((prev) => [adopted, ...prev]);
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
      // Agent-memory scope retrieves FROM an agent session, so it must not
      // pick up this page's own search sessions.
      const mostRecentAgentSessionId = sessions.find(
        (s) => !s.session_id.startsWith(SESSION_PREFIX),
      )?.session_id;
      const sendScope = scope === "agent" ? ["session", "trace"] : "graph";
      // Graph searches carry the conversation's session id so the backend
      // records the Q&A — that's what persists the chat across refreshes.
      const sendSessionId = scope === "agent" ? mostRecentAgentSessionId : finalConvoId;
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
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <TrackPageView page="Search" />

      {/* Header */}
      <div style={{ padding: "24px 32px 16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexShrink: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <h1 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif' }}>Search</h1>
          <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Ask questions about your knowledge graph and agent memory.</p>
        </div>
      </div>

      {/* Main panel — same dark-glass container as Sessions / Brain */}
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
            {allConversations.length === 0 && (
              <div style={{ padding: "24px 12px", textAlign: "center" }}>
                <span style={{ fontSize: 12, color: "rgba(237,236,234,0.35)" }}>No conversations yet</span>
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
            {/* Controls: scope pills + brain selector */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
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
                        background: active ? "rgba(188,155,255,0.35)" : "rgba(255,255,255,0.06)",
                        color: active ? "#EDECEA" : "rgba(237,236,234,0.6)",
                        border: active ? "none" : "1px solid rgba(255,255,255,0.1)",
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
              <BrainSelector allowAll={false} align="right" direction="up" />
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
                placeholder={hasAccess ? "Ask a question about your data..." : "Subscribe to use search..."}
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
