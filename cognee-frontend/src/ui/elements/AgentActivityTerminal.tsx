"use client";

import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { SEARCH_SESSION_PREFIX } from "@/modules/sessions/getSessions";

export interface PipelineRun { id: string; pipeline_name: string; status: string; dataset_id: string | null; dataset_name: string | null; owner_email: string | null; created_at: string | null; pipeline_run_id: string | null }

export type Range = "24h" | "7d" | "30d";

export function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function pipelineLabel(name: string): string {
  if (name.includes("cognify")) return "cognee.cognify";
  if (name.includes("add")) return "cognee.add";
  if (name.includes("search")) return "cognee.search";
  if (name.includes("recall")) return "cognee.recall";
  return name;
}

export function ownerDisplayName(email: string | null): string {
  if (!email) return "System";
  if (email.endsWith("@cognee.agent")) {
    const local = email.split("@")[0];
    const parts = local.split("-");
    const shortId = parts.pop() || "";
    const type = parts.join(" ");
    return `${type} ${shortId}`;
  }
  if (email === "default_user@example.com") return "You";
  return email.split("@")[0];
}

// Stable per-actor accent so each agent reads as a distinct connection in the
// log. "You" always gets the brand violet.
const ACTOR_COLORS = ["#89B4FA", "#A6E3A1", "#F9E2AF", "#F5C2E7", "#94E2D5", "#FAB387", "#B4BEFE", "#F38BA8"];
export function actorColor(name: string): string {
  if (name === "You") return "#CBA6F7";
  let h = 0;
  for (let i = 0; i < name.length; i++) h = ((h << 5) - h + name.charCodeAt(i)) | 0;
  return ACTOR_COLORS[Math.abs(h) % ACTOR_COLORS.length];
}

// Typed outcome of a memory interaction — this is the core of the evidence
// model: a recall that returns nothing is "empty" with a reason, never silence.
type Outcome = "hit" | "empty" | "error" | "running" | "done";
const OUTCOME_META: Record<Outcome, { color: string; label: string }> = {
  hit:     { color: "#A6E3A1", label: "hit" },
  empty:   { color: "#F9E2AF", label: "empty" },
  error:   { color: "#F38BA8", label: "error" },
  running: { color: "#F9E2AF", label: "running" },
  done:    { color: "#A6E3A1", label: "done" },
};
const FILTER_HELP: Record<"all" | "mine" | "agents" | "searches" | "errors", string> = {
  all: "Show every memory event in this time range.",
  mine: "Show searches you typed in this terminal.",
  agents: "Show searches made by connected agents.",
  searches: "Show memory lookups and their answers.",
  errors: "Show failed searches or memory actions.",
};

function ActorDot({ name, live }: { name: string; live?: boolean }) {
  return (
    <span style={{ width: 7, height: 7, borderRadius: "50%", background: actorColor(name), flexShrink: 0, display: "inline-block", animation: live ? "term-live 1.8s ease-in-out infinite" : undefined }} />
  );
}

function OutcomeBadge({ outcome }: { outcome: Outcome }) {
  const m = OUTCOME_META[outcome];
  return (
    <span style={{ color: m.color, flexShrink: 0, display: "inline-flex", alignItems: "center", gap: 4 }}>
      {outcome === "running"
        ? <span style={{ width: 8, height: 8, borderRadius: "50%", border: "1.5px solid #313244", borderTopColor: m.color, animation: "term-spin 0.8s linear infinite", display: "inline-block" }} />
        : <span>{outcome === "hit" || outcome === "done" ? "✓" : outcome === "error" ? "✗" : "∅"}</span>}
      {m.label}
    </span>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <span style={{ color: "#585B70", flexShrink: 0, display: "inline-block", width: 9, transition: "transform 120ms", transform: open ? "rotate(90deg)" : "none" }}>▸</span>
  );
}

export const DEMO_QUERIES = [
  "What are the main topics in this dataset?",
  "What are the key entities and relationships?",
  "Summarize the most important concepts",
];

// Typewriter prompts that animate in the search input's placeholder so it's
// obvious the field accepts free text. Cycled character-by-character.
// Dataset-agnostic prompts — every one of these should make sense against
// any cognified corpus (notes, docs, code, research, transcripts, etc.).
// Avoid domain specifics (Einstein, PR #3076, Auth0, etc.) that produce
// "no results" on most tenants.
const PLACEHOLDER_PROMPTS = [
  "What are the main topics in my data?",
  "Summarize the most important findings",
  "Who are the people mentioned here?",
  "What patterns appear across my notes?",
  "Give me a high-level overview",
  "Find connections between the key concepts",
  "What did I add most recently?",
  "List the recurring themes",
  "What entities show up most often?",
  "Show me the strongest relationships in the graph",
  "What questions does this data answer?",
  "Surface anything that looks contradictory",
  "Compare the most-cited ideas",
  "What's the most important decision recorded?",
  "List every project or initiative mentioned",
  "Which topics are connected to each other?",
  "What should I review first?",
  "Summarize each document briefly",
  "What's missing from my knowledge graph?",
  "Walk me through this dataset",
];

// Entry shape used by the onboarding parent: it kicks off the recalls when
// cognify finishes and feeds entries in as they settle. The terminal then
// reveals them sequentially with a typewriter.
export interface OnboardingDemoEntry {
  query: string;
  result: string | null;
  status: "pending" | "done" | "error";
}

type DemoEntry = { query: string; result: string | null; status: "idle" | "running" | "done" | "error" };
type QAItem = { question: string; answer: string | null; time: string | null; source: "recall" | "remember" };

// QA timestamps arrive as naive UTC ("2026-06-10T08:32:02.940762"); treat
// them as UTC so they interleave correctly with timezone-aware events.
function parseQATime(t: string | null): number | null {
  if (!t) return null;
  const iso = /Z$|[+-]\d{2}:?\d{2}$/.test(t) ? t : `${t}Z`;
  const ms = new Date(iso).getTime();
  return Number.isNaN(ms) ? null : ms;
}

type DatasetResult = { dataset: string | null; text: string };

// Boilerplate "nothing found" completions returned by datasets that hold no
// relevant knowledge — dropped so one query only shows brains that answered.
const NO_ANSWER_PATTERNS = [
  /no (relevant |specific |such )?(information|data|context|knowledge|results?|answer)/i,
  /\b(does not|doesn'?t|do not|don'?t)\b[^.]{0,60}\b(contain|include|provide|mention|have|appear)/i,
  /\b(i|we) (don'?t|do not|cannot|can'?t) (have|find|know|answer|provide|determine)/i,
  /unable to (find|answer|provide|determine|locate)/i,
  /not enough (information|context)/i,
  /cannot be (determined|answered|found)/i,
  /there is no (information|mention|data|reference)/i,
  /no results? (were |was )?(found|returned)/i,
];

function isNoAnswer(text: string): boolean {
  const t = text.trim();
  if (!t) return true;
  // Only short responses can be pure boilerplate — long answers that merely
  // contain a negative phrase somewhere still carry substance.
  return t.length < 300 && NO_ANSWER_PATTERNS.some((p) => p.test(t));
}

// Maps /v1/recall rows to per-dataset entries; in cloud mode each row carries
// dataset_name alongside the completion.
function extractDatasetResults(data: unknown): DatasetResult[] {
  if (!Array.isArray(data)) return [];
  const out: DatasetResult[] = [];
  for (const r of data) {
    const row = r as { text?: unknown; answer?: unknown; search_result?: unknown; dataset_name?: unknown };
    let text: string | null = null;
    if (typeof row.text === "string") text = row.text;
    else if (typeof row.answer === "string") text = row.answer;
    else if (typeof row.search_result === "string") text = row.search_result;
    else if (Array.isArray(row.search_result) && row.search_result.every((x) => typeof x === "string")) text = (row.search_result as string[]).join("\n\n");
    else { try { text = JSON.stringify(r); } catch { text = null; } }
    if (!text || isNoAnswer(text)) continue;
    out.push({ dataset: typeof row.dataset_name === "string" ? row.dataset_name : null, text });
  }
  return out;
}

export function AgentActivityTerminal({
  sessions, runs, agents, datasets, selectedDataset, cogniInstance, dataLoading, onNavigate, variant, onboardingDemo,
}: {
  sessions: import("@/modules/sessions/getSessions").SessionRow[];
  runs: PipelineRun[];
  agents: { id: string; agent_type: string; is_agent: boolean; is_default: boolean; email: string }[];
  datasets: { id: string; name: string }[];
  selectedDataset: { id: string; name: string } | null;
  cogniInstance: ReturnType<typeof useCogniInstance>["cogniInstance"];
  dataLoading: boolean;
  range: Range;
  onNavigate: (path: string) => void;
  variant?: "onboarding";
  // Pre-issued demo recalls from the onboarding parent. When provided, the
  // terminal reveals the entries one by one (typewriter); when omitted it
  // falls back to firing its own recalls (dashboard demo path).
  onboardingDemo?: OnboardingDemoEntry[] | null;
}) {
  const logRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const atBottomRef = useRef(true);
  const demoStartedRef = useRef(false);
  const fetchedQAKeys = useRef<Record<string, string>>({});
  const [termFilter, setTermFilter] = useState<"all" | "mine" | "agents" | "searches" | "errors">("all");
  // Demo plays exactly once per browser (flag persisted in localStorage).
  // Two trigger points:
  //   - Step 3 of onboarding ("Ask cognee anything"), where the variant is
  //     "onboarding" — this is the intended showcase moment.
  //   - The dashboard terminal once onboarding has been explicitly completed.
  const [demoEligible] = useState(() => {
    try {
      if (localStorage.getItem("cognee-terminal-demo-shown") === "1") return false;
      if (variant === "onboarding") return true;
      return !!localStorage.getItem("cognee-onboarding-complete");
    } catch { return false; }
  });
  const [sessionQAs, setSessionQAs] = useState<Record<string, QAItem[]>>({});
  const [searchInput, setSearchInput] = useState("");
  const [userQueries, setUserQueries] = useState<{ id: number; query: string; results: DatasetResult[] | null; searching: boolean; error?: boolean; errorMessage?: string; ts: number }[]>([]);
  const [demoEntries, setDemoEntries] = useState<DemoEntry[]>([]);
  const [inputFocused, setInputFocused] = useState(false);
  // Which event card is expanded to show its retrieval evidence.
  // `expandedKey === null` AND `!userSelected` means "auto-pick the latest
  // settled card" (the default-open behaviour). Once the user clicks any
  // card — including the auto-expanded one to collapse it — we flip
  // `userSelected = true` so null becomes an explicit "nothing open"
  // instead of falling back to the default.
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const [userSelected, setUserSelected] = useState(false);

  // Placeholder typewriter: cycles through PLACEHOLDER_PROMPTS char-by-char
  // while the input is empty + unfocused + connected. Pauses when the user
  // takes over.
  const [typedPlaceholder, setTypedPlaceholder] = useState("");
  const [typingActive, setTypingActive] = useState(false);

  useEffect(() => {
    if (searchInput || inputFocused || !cogniInstance) {
      setTypedPlaceholder("");
      setTypingActive(false);
      return;
    }
    let cancelled = false;
    let idx = 0;
    let pos = 0;
    let mode: "type" | "pause" | "erase" = "type";
    let timer: ReturnType<typeof setTimeout>;
    const tick = () => {
      if (cancelled) return;
      const current = PLACEHOLDER_PROMPTS[idx];
      if (mode === "type") {
        setTypingActive(true);
        pos += 1;
        setTypedPlaceholder(current.slice(0, pos));
        if (pos >= current.length) {
          mode = "pause";
          timer = setTimeout(tick, 1800);
          return;
        }
        timer = setTimeout(tick, 55 + Math.random() * 40);
      } else if (mode === "pause") {
        setTypingActive(false);
        mode = "erase";
        timer = setTimeout(tick, 250);
      } else {
        setTypingActive(true);
        pos -= 1;
        setTypedPlaceholder(current.slice(0, pos));
        if (pos <= 0) {
          idx = (idx + 1) % PLACEHOLDER_PROMPTS.length;
          mode = "type";
          timer = setTimeout(tick, 350);
          return;
        }
        timer = setTimeout(tick, 22);
      }
    };
    timer = setTimeout(tick, 400);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [searchInput, inputFocused, cogniInstance]);

  const FONT: React.CSSProperties = { fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace', fontSize: 12, lineHeight: "19px" };

  type TermEvent =
    | { kind: "run"; r: PipelineRun; ts: number }
    | { kind: "session"; s: import("@/modules/sessions/getSessions").SessionRow; agent: typeof agents[number] | null; ts: number }
    | { kind: "agentQuery"; sessionId: string; label: string; question: string; answer: string | null; ts: number; source: "recall" | "remember" }
    | { kind: "userQuery"; id: number; query: string; results: DatasetResult[] | null; searching: boolean; error?: boolean; errorMessage?: string; ts: number };

  const allEvents: TermEvent[] = [];
  // Only surface agent-requested operations (searches/recalls) — internal
  // pipelines (add, cognify, memify, indexing) are noise in this view.
  // Items without timestamps sort to the bottom (treated as newest) rather
  // than being dropped or pinned to 1970.
  for (const r of runs) {
    const pipelineName = r.pipeline_name.toLowerCase();
    if (!pipelineName.includes("search") && !pipelineName.includes("recall")) continue;
    allEvents.push({ kind: "run", r, ts: r.created_at ? new Date(r.created_at).getTime() : Number.MAX_SAFE_INTEGER });
  }
  // One recall across N datasets logs N session-QA entries with the same
  // question at ~the same moment — collapse those into a single line. Queries
  // typed into this terminal also echo back through the session log; drop
  // them entirely since the userQuery line already shows the full answer.
  const lastQATime = new Map<string, number>();
  for (const s of sessions) {
    // Search-page conversations are sessions too — they live in the Search
    // sidebar, not here, and would otherwise crowd agent sessions out of
    // the limited fetch window.
    if (s.session_id.startsWith(SEARCH_SESSION_PREFIX)) continue;
    const agent = agents.find(a => a.id === s.user_id) ?? null;
    const sessionTs = s.last_activity_at ? new Date(s.last_activity_at).getTime() : (s.started_at ? new Date(s.started_at).getTime() : Number.MAX_SAFE_INTEGER);
    allEvents.push({ kind: "session", s, agent, ts: sessionTs });
    // Each recall question is its own log line at its own timestamp, so the
    // terminal shows the actual recall order instead of grouping by session.
    const label = agent?.agent_type ?? s.session_id;
    for (const qa of sessionQAs[s.session_id] ?? []) {
      const qaTs = parseQATime(qa.time) ?? sessionTs;
      if (userQueries.some(uq => uq.query === qa.question && Math.abs(qaTs - uq.ts) < 5 * 60_000)) continue;
      // Dedupe per source so a recall and a remember with the same question both render
      // (the whole point of the source label) — only collapse within the same source.
      const dedupeKey = `${s.session_id}|${qa.source}|${qa.question}`;
      const prevTs = lastQATime.get(dedupeKey);
      if (prevTs !== undefined && Math.abs(qaTs - prevTs) < 120_000) continue;
      lastQATime.set(dedupeKey, qaTs);
      allEvents.push({
        kind: "agentQuery",
        sessionId: s.session_id,
        label,
        question: qa.question,
        answer: qa.answer,
        ts: qaTs,
        source: qa.source,
      });
    }
  }
  for (const uq of userQueries) {
    allEvents.push({ kind: "userQuery", id: uq.id, query: uq.query, results: uq.results, searching: uq.searching, error: uq.error, errorMessage: uq.errorMessage, ts: uq.ts });
  }
  allEvents.sort((a, b) => a.ts - b.ts);

  // Cap how many distinct agents are shown at once. The demo (and a clean live
  // view) should never surface more than two agents — keep the two most
  // recently active; the user's own queries are always exempt.
  const MAX_AGENTS = 2;
  function agentOf(ev: TermEvent): string | null {
    if (ev.kind === "agentQuery") return ev.label;
    if (ev.kind === "session") return ev.agent?.agent_type ?? (ev.s.user_id?.includes("@") ? ownerDisplayName(ev.s.user_id) : ev.s.session_id);
    if (ev.kind === "run") { const o = ownerDisplayName(ev.r.owner_email); return o === "You" || o === "System" ? null : o; }
    return null; // userQuery — that's "You", never counted as an agent
  }
  const latestByAgent = new Map<string, number>();
  for (const ev of allEvents) {
    const a = agentOf(ev);
    if (a) latestByAgent.set(a, Math.max(latestByAgent.get(a) ?? 0, ev.ts));
  }
  const allowedAgents = new Set(
    [...latestByAgent.entries()].sort((x, y) => y[1] - x[1]).slice(0, MAX_AGENTS).map(e => e[0]),
  );

  const events = allEvents.filter(ev => {
    // Enforce the two-agent cap before anything else.
    const ag = agentOf(ev);
    if (ag && !allowedAgents.has(ag)) return false;
    // "Mine" = only the queries you typed here. "Agents" = everything else
    // (other actors interacting with memory), so the you-vs-them split the
    // log is named for is one click away.
    if (termFilter === "mine") return ev.kind === "userQuery";
    // The user's own queries are filter-exempt on the remaining tabs: a typed
    // query must never silently vanish from the log, whatever tab is active.
    if (ev.kind === "userQuery") return termFilter !== "agents";
    if (termFilter === "agents") return true; // all non-user events are agent activity
    if (termFilter === "searches") {
      if (ev.kind === "run") return true; // runs are pre-filtered to search/recall
      return ev.kind === "agentQuery";
    }
    if (termFilter === "errors") {
      if (ev.kind === "run") return ev.r.status.includes("ERRORED");
      if (ev.kind === "session") return ev.s.effective_status === "failed" || ev.s.error_count > 0;
      return false;
    }
    return true;
  });

  // Key of the most recent expandable card (a settled recall by an agent or
  // you). When the user hasn't explicitly opened another card, this one shows
  // its result by default — so the latest answer is always visible un-clicked.
  let lastExpandableKey: string | null = null;
  for (const ev of events) {
    if (ev.kind === "userQuery") {
      if (!ev.searching) lastExpandableKey = `uq-${ev.id}`;
    } else if (ev.kind === "agentQuery") {
      lastExpandableKey = `aq-${ev.sessionId}-${ev.ts}-${ev.question.slice(0, 24)}`;
    }
  }

  const hasRunning = sessions.some(s => s.effective_status === "running");
  // The demo is a one-shot prologue for a freshly onboarded account. It only
  // plays when there is genuinely no real activity of any kind.
  const showDemo = demoEligible && allEvents.length === 0 && !dataLoading && datasets.length > 0;

  async function handleSearch(q: string) {
    if (!q.trim() || !cogniInstance) return;
    const query = q.trim();
    const id = Date.now();
    setSearchInput("");
    setUserQueries(prev => [...prev, { id, query, results: null, searching: true, ts: id }]);
    try {
      const { default: recallKnowledge } = await import("@/modules/datasets/recallKnowledge");
      const data = await recallKnowledge(cogniInstance, {
        query,
        scope: "graph" as never,
        datasetIds: selectedDataset ? [selectedDataset.id] : datasets.map(d => d.id),
      });
      const entries = extractDatasetResults(data);
      setUserQueries(prev => prev.map(uq => uq.id === id ? { ...uq, results: entries, searching: false } : uq));
    } catch (e) {
      setUserQueries(prev => prev.map(uq => uq.id === id
        ? { ...uq, error: true, errorMessage: e instanceof Error ? e.message : undefined, searching: false }
        : uq));
    } finally {
      setTimeout(() => logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" }), 80);
    }
  }

  function clearSearches() {
    setUserQueries([]);
  }

  // Sticky autoscroll: follow new content (poll events, demo progress, Q&A
  // lines) only while the user is already at the bottom of the log.
  const qaCount = Object.values(sessionQAs).reduce((n, a) => n + a.length, 0);
  const demoSig = demoEntries.map(e => e.status).join(",");
  useEffect(() => {
    if (dataLoading || !atBottomRef.current) return;
    const el = logRef.current;
    if (!el) return;
    const t = setTimeout(() => el.scrollTo({ top: el.scrollHeight, behavior: "smooth" }), 80);
    return () => clearTimeout(t);
  }, [dataLoading, events.length, qaCount, demoSig]);

  useEffect(() => {
    // Onboarding mode runs a separate, paced reveal effect (below) that consumes
    // the parent-supplied recalls — never fire our own in that mode.
    if (onboardingDemo) return;
    if (!showDemo || !cogniInstance || datasets.length === 0 || demoStartedRef.current) {
      // Don't clear demoEntries — keep them as terminal history when the user starts interacting
      return;
    }
    demoStartedRef.current = true; // never replays within this mount (e.g. after "clear searches")
    let cancelled = false;
    try { localStorage.setItem("cognee-terminal-demo-shown", "1"); } catch {}
    setDemoEntries(DEMO_QUERIES.map(q => ({ query: q, result: null, status: "idle" as const })));
    (async () => {
      const { default: recallKnowledge } = await import("@/modules/datasets/recallKnowledge");
      for (let i = 0; i < DEMO_QUERIES.length; i++) {
        if (cancelled) return;
        if (i > 0) await new Promise(r => setTimeout(r, 700));
        if (cancelled) return;
        setDemoEntries(prev => prev.map((e, j) => j === i ? { ...e, status: "running" } : e));
        try {
          const data = await recallKnowledge(cogniInstance, {
            query: DEMO_QUERIES[i],
            scope: "graph" as never,
            datasetIds: datasets.map(d => d.id),
          });
          if (cancelled) return;
          const entries = extractDatasetResults(data);
          setDemoEntries(prev => prev.map((e, j) => j === i ? { ...e, status: "done", result: entries[0]?.text.slice(0, 220) ?? null } : e));
        } catch {
          if (!cancelled) setDemoEntries(prev => prev.map((e, j) => j === i ? { ...e, status: "error" } : e));
        }
      }
    })();
    return () => {
      cancelled = true;
      // Settle anything in flight so no entry is left blinking "searching…" forever.
      setDemoEntries(prev => prev.map(e => e.status === "running" || e.status === "idle" ? { ...e, status: "error" } : e));
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showDemo, cogniInstance, onboardingDemo]);

  // Onboarding mode: parent has already kicked off the recalls. Reveal each
  // question sequentially — wait at least 2s and until that entry's recall
  // has settled, then type the answer out, then move to the next one. A ref
  // mirror of the prop lets the polling loop see live status updates without
  // restarting the state machine when entries re-render.
  const onboardingDemoRef = useRef(onboardingDemo);
  useEffect(() => { onboardingDemoRef.current = onboardingDemo; }, [onboardingDemo]);
  useEffect(() => {
    if (!onboardingDemo || onboardingDemo.length === 0 || variant !== "onboarding") return;
    if (demoStartedRef.current) return;
    demoStartedRef.current = true;
    try { localStorage.setItem("cognee-terminal-demo-shown", "1"); } catch {}

    let cancelled = false;
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

    (async () => {
      for (let i = 0; i < (onboardingDemoRef.current?.length ?? 0); i++) {
        if (cancelled) return;

        // 1. Show the question with the "searching…" spinner.
        const entry = onboardingDemoRef.current?.[i];
        if (!entry) return;
        const question = entry.query;
        setDemoEntries((prev) => [...prev, { query: question, result: null, status: "running" }]);

        // 2. Wait min 2s AND for the recall to land.
        const startedAt = Date.now();
        while (!cancelled) {
          const cur = onboardingDemoRef.current?.[i];
          const settled = cur && cur.status !== "pending";
          const elapsed = Date.now() - startedAt;
          if (settled && elapsed >= 2000) break;
          await sleep(120);
        }
        if (cancelled) return;

        const final = onboardingDemoRef.current?.[i];
        if (!final) return;

        if (final.status !== "done" || !final.result) {
          // No usable answer — just collapse to the existing "no results" branch.
          setDemoEntries((prev) => prev.map((e, j) => j === i ? { ...e, status: "done", result: null } : e));
          await sleep(700);
          continue;
        }

        // 3. Switch from spinner to a growing string (typewriter).
        const fullText = final.result;
        setDemoEntries((prev) => prev.map((e, j) => j === i ? { ...e, status: "done", result: "" } : e));
        const CHAR_MS = 14;
        for (let k = 1; k <= fullText.length; k++) {
          if (cancelled) return;
          const partial = fullText.slice(0, k);
          setDemoEntries((prev) => prev.map((e, j) => j === i ? { ...e, result: partial } : e));
          await sleep(CHAR_MS);
        }

        // 4. Brief beat between questions.
        await sleep(800);
      }
    })();

    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [!!onboardingDemo, variant]);

  // Fetch the actual questions agents asked (QA entries + trace queries)
  // for the visible sessions, keyed by session_id. Results are merged (not
  // replaced) so a transient fetch failure never erases rendered lines, and
  // sessions whose activity timestamp is unchanged are not refetched.
  const sessionsKey = sessions.map(s => `${s.session_id}:${s.last_activity_at}`).join("|");
  useEffect(() => {
    if (!cogniInstance || sessions.length === 0) return;
    const targets = sessions
      .filter(s => !s.session_id.startsWith(SEARCH_SESSION_PREFIX))
      .slice(0, 20)
      .filter(s => fetchedQAKeys.current[s.session_id] !== String(s.last_activity_at));
    if (targets.length === 0) return;
    let cancelled = false;
    (async () => {
      const { getSessionDetail } = await import("@/modules/sessions/getSessions");
      const details = await Promise.all(
        targets.map(s => getSessionDetail(cogniInstance, s.session_id).catch(() => null))
      );
      if (cancelled) return;
      const map: Record<string, QAItem[]> = {};
      targets.forEach((s, i) => {
        const d = details[i];
        if (!d) return; // failed fetch: keep previous entries, retry next tick
        fetchedQAKeys.current[s.session_id] = String(s.last_activity_at);
        // /recall always writes a trace with memory_query=question; /remember/entry never
        // does. So a QA whose question matches a trace.memory_query came from recall;
        // anything else was saved by the agent via /remember. Heuristic, not authoritative —
        // ponytail: swap for an explicit `source` field on SessionQAEntry if it misfires.
        const recallQuestions = new Set(
          (d.traces ?? []).filter(t => t.memory_query).map(t => String(t.memory_query).trim())
        );
        const qas = (d.qas ?? [])
          .map(qa => {
            const row = qa as { question?: unknown; answer?: unknown; time?: unknown };
            const question = String(row.question ?? "").trim();
            return {
              question,
              answer: row.answer ? String(row.answer) : null,
              time: row.time ? String(row.time) : null,
              source: (recallQuestions.has(question) ? "recall" : "remember") as "recall" | "remember",
            };
          })
          .filter(qa => qa.question);
        // Surface trace-only recalls (the plugin path, where recall fires but agentic_retriever
        // didn't write a QA) — skip any whose question already showed up as a QA above.
        const qaQuestions = new Set(qas.map(qa => qa.question));
        const traceQueries = (d.traces ?? [])
          .filter(t => t.memory_query)
          .map(t => ({
            question: String(t.memory_query).trim(),
            answer: null,
            time: t.time ? String(t.time) : null,
            source: "recall" as const,
          }))
          .filter(qa => qa.question && !qaQuestions.has(qa.question));
        const merged = [...qas, ...traceQueries]
          .sort((a, b) => (a.time ?? "").localeCompare(b.time ?? ""))
          .slice(-12);
        if (merged.length) map[s.session_id] = merged;
      });
      if (Object.keys(map).length) setSessionQAs(prev => ({ ...prev, ...map }));
    })();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cogniInstance, sessionsKey]);

  const isOnboarding = variant === "onboarding";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      <style>{`
        @keyframes term-live { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.8)} }
        @keyframes term-spin { to { transform: rotate(360deg); } }
        @keyframes termBlink2 { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes term-tabin { from { opacity: 0.3; transform: translateY(1px); } to { opacity: 1; transform: none; } }
        @keyframes term-tab-underline { from { transform: scaleX(0); } to { transform: scaleX(1); } }
        .term-tab-active { position: relative; animation: term-tabin 220ms ease; }
        .term-tab-active::after { content: ""; position: absolute; left: 0; right: 0; bottom: -3px; height: 1px; background: #EDECEA; transform-origin: left; animation: term-tab-underline 220ms ease forwards; }
        .term-search-input:focus { outline: none; }
        /* Lock the markdown body + every block element to 12px so nested
           bullets/lists/blockquotes stay aligned with the terminal text.
           Mantine global typography sets font-size on p, so font-size:
           inherit loses the cascade. We set 12px explicitly. h1-h3 + code
           rules come later so their overrides still win. */
        .term-md { color: #CDD6F4; line-height: 1.6; word-break: break-word; font-size: 12px; }
        .term-md p, .term-md ul, .term-md ol, .term-md li, .term-md blockquote, .term-md a, .term-md span { font-size: 12px; line-height: 1.6; }
        .term-md p { margin: 0 0 8px 0; }
        .term-md p:last-child { margin-bottom: 0; }
        .term-md strong { color: #CBA6F7; font-weight: 700; }
        .term-md em { color: #A6E3A1; font-style: normal; }
        .term-md h1, .term-md h2, .term-md h3 { color: #89B4FA; font-weight: 700; margin: 10px 0 4px; font-size: 13px; }
        .term-md ul, .term-md ol { margin: 4px 0 8px; padding-left: 18px; }
        .term-md li { margin-bottom: 3px; }
        .term-md code { background: #313244; color: #F38BA8; padding: 1px 4px; border-radius: 3px; font-size: 11px; font-family: inherit; }
        .term-md pre { background: #181825; border-radius: 6px; padding: 10px 12px; margin: 8px 0; overflow-x: auto; }
        .term-md pre code { background: none; color: #CDD6F4; padding: 0; }
        .term-md a { color: #89B4FA; text-decoration: underline; }
        .term-md blockquote { border-left: 2px solid #45475A; padding-left: 10px; color: #6C7086; margin: 6px 0; }
      `}</style>
      <div style={{ background: "#1E1E2E", borderRadius: 12, overflow: "hidden", boxShadow: "0 4px 32px rgba(0,0,0,0.22)", border: "1px solid #313244" }}>

        {/* Single compact header bar */}
        <div style={{ background: "#181825", borderBottom: "1px solid #2A2A3E", padding: "7px 18px", display: "flex", alignItems: "center", gap: 14, minHeight: 30, boxSizing: "border-box" }}>
          {!isOnboarding && (["all", "mine", "agents", "searches", "errors"] as const).map(f => {
            const active = termFilter === f;
            return (
              <button
                // Remount on activation so the underline/fade animation replays each click.
                key={active ? `${f}-active` : f}
                onClick={() => setTermFilter(f)}
                title={FILTER_HELP[f]}
                aria-label={`${f === "all" ? "all events" : f}: ${FILTER_HELP[f]}`}
                className={active ? "term-tab-active" : undefined}
                style={{
                  background: "none", border: "none", cursor: "pointer",
                  ...FONT, fontSize: 11,
                  color: active ? "#EDECEA" : "rgba(237,236,234,0.55)",
                  paddingBottom: 2, transition: "color 140ms", textDecoration: "underline dotted rgba(237,236,234,0.25)", textUnderlineOffset: 4,
                }}>
                {f === "all" ? "all events" : f}
              </button>
            );
          })}
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 12 }}>
            {userQueries.length > 0 && (
              <button
                onClick={clearSearches}
                title="Remove only the searches you typed here; saved session history stays intact."
                style={{ background: "none", border: "none", cursor: "pointer", ...FONT, fontSize: 10, color: "rgba(237,236,234,0.55)", padding: 0 }}
              >
                clear searches
              </button>
            )}
            {hasRunning && (
              <div title="At least one memory job is still running." style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#A6E3A1", display: "inline-block", animation: "term-live 1.8s ease-in-out infinite" }} />
                <span style={{ ...FONT, fontSize: 10, color: "#A6E3A1" }}>live</span>
              </div>
            )}
            {!isOnboarding && (
              <button
                onClick={() => onNavigate("/integrations")}
                title="Connect your own agent to see its searches here"
                style={{ display: "flex", alignItems: "center", gap: 5, background: "rgba(203,166,247,0.12)", border: "1px solid rgba(203,166,247,0.35)", borderRadius: 6, padding: "3px 9px", cursor: "pointer", ...FONT, fontSize: 10, color: "#CBA6F7" }}
              >
                <span style={{ fontSize: 12, lineHeight: "10px" }}>+</span> Connect an agent
              </button>
            )}
            {!isOnboarding && (
              <button
                onClick={() => onNavigate("/sessions")}
                title="Open the full session log with transcripts and metadata."
                style={{ background: "none", border: "none", ...FONT, fontSize: 10, color: "rgba(237,236,234,0.55)", cursor: "pointer", padding: 0 }}
              >
                sessions ↗
              </button>
            )}
          </div>
        </div>

        {/* Terminal body — clicking anywhere focuses the one real prompt */}
        <div
          ref={logRef}
          onScroll={e => {
            const el = e.currentTarget;
            atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
          }}
          onClick={() => {
            const sel = window.getSelection();
            if (!sel || sel.isCollapsed) inputRef.current?.focus();
          }}
          style={{ padding: "16px 20px 10px", minHeight: 260, maxHeight: 440, overflowY: "auto", ...FONT }}
        >

          {dataLoading ? (
            <div style={{ display: "flex", alignItems: "center", gap: 10, color: "#585B70" }}>
              <div style={{ width: 13, height: 13, borderRadius: "50%", border: "1.5px solid #313244", borderTopColor: "#CBA6F7", animation: "term-spin 0.8s linear infinite", flexShrink: 0 }} />
              <span>Connecting to agent stream…</span>
            </div>
          ) : (
            <>
              {/* One-shot demo entries render as ordinary log history */}
              {demoEntries.filter(e => e.status !== "idle").map((entry, i) => (
                <div key={`demo-${i}`} style={{ marginBottom: 16 }}>
                  <div style={{ display: "flex", alignItems: "baseline", flexWrap: "wrap", gap: 0 }}>
                    <span style={{ color: "#585B70", flexShrink: 0 }}>$&nbsp;</span>
                    <span style={{ color: "#CBA6F7", fontWeight: 700, flexShrink: 0 }}>Claude Code Demo Agent</span>
                    <span style={{ color: "#585B70" }}>&nbsp;❯&nbsp;&ldquo;</span>
                    <span style={{ color: "#A6E3A1" }}>{entry.query}</span>
                    <span style={{ color: "#585B70" }}>&rdquo;</span>
                  </div>
                  <div style={{ paddingLeft: "2ch", marginTop: 4 }}>
                    {entry.status === "running" ? (
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <div style={{ width: 10, height: 10, borderRadius: "50%", border: "1.5px solid #313244", borderTopColor: "#CBA6F7", animation: "term-spin 0.8s linear infinite", flexShrink: 0 }} />
                        <span style={{ color: "#F9E2AF" }}>searching…</span>
                      </div>
                    ) : entry.result ? (
                      <div className="term-md"><ReactMarkdown>{entry.result}</ReactMarkdown></div>
                    ) : entry.status === "error" ? (
                      <span style={{ color: "#6C7086" }}>✗ interrupted</span>
                    ) : (
                      <span style={{ color: "#45475A" }}>no results</span>
                    )}
                  </div>
                </div>
              ))}

              {events.length === 0 && demoEntries.length === 0 && (
                showDemo ? (
                  <div style={{ display: "flex", alignItems: "center", gap: 10, color: "#585B70" }}>
                    <div style={{ width: 13, height: 13, borderRadius: "50%", border: "1.5px solid #313244", borderTopColor: "#CBA6F7", animation: "term-spin 0.8s linear infinite", flexShrink: 0 }} />
                    <span>Claude Code Demo connecting…</span>
                  </div>
                ) : (
                  isOnboarding ? (
                    <div style={{ color: "#45475A", lineHeight: "22px" }}>Type a question below to query your memory.</div>
                  ) : termFilter === "errors" ? (
                    <div style={{ color: "#45475A", lineHeight: "22px" }}>No errors in this time range.</div>
                  ) : termFilter === "mine" ? (
                    <div style={{ color: "#45475A", lineHeight: "22px" }}>You haven&apos;t searched yet — type a query below to start.</div>
                  ) : (
                    // Default / "agents" / "searches": teach how to populate the log.
                    <div style={{ display: "flex", flexDirection: "column", gap: 10, padding: "14px 2px" }}>
                      <div style={{ color: "#A6ADC8" }}>No agent activity yet.</div>
                      <div style={{ color: "#585B70", lineHeight: "20px", maxWidth: 520 }}>
                        These rows appear when an agent searches your memory. Connect your own
                        agent (Claude Code, Codex, Cursor, …) and its recalls will stream here
                        with the evidence behind each answer.
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 2 }}>
                        <span style={{ color: "#45475A" }}>Type a search below to query your memory.</span>
                      </div>
                    </div>
                  )
                )
              )}

            {events.map(ev => {
              // ── Operational pipeline run: a compact, non-expandable marker ──
              if (ev.kind === "run") {
                const r = ev.r;
                const isError = r.status.includes("ERRORED");
                const isRunning = r.status.includes("STARTED") || r.status.includes("INITIATED");
                const outcome: Outcome = isRunning ? "running" : isError ? "error" : "done";
                const label = pipelineLabel(r.pipeline_name);
                const actor = ownerDisplayName(r.owner_email);
                return (
                  <div key={`run-${r.pipeline_run_id ?? r.id}`} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
                    <span style={{ width: 9 }} />
                    <ActorDot name={actor} live={isRunning} />
                    <span style={{ color: "#CDD6F4" }}>{actor}</span>
                    <span style={{ color: "#585B70" }}>ran</span>
                    <span style={{ color: "#CBA6F7", fontWeight: 700 }}>{label}</span>
                    {r.dataset_name && <span style={{ color: "#A6E3A1" }}>· {r.dataset_name}</span>}
                    <OutcomeBadge outcome={outcome} />
                    {r.created_at && <span style={{ color: "#45475A" }}>· {timeAgo(r.created_at)}</span>}
                  </div>
                );
              }

              // ── Session lifecycle: an actor connected to memory ──
              if (ev.kind === "session") {
                const s = ev.s;
                const agentType = ev.agent?.agent_type ?? (s.user_id.includes("@") ? ownerDisplayName(s.user_id) : null);
                const actorName = agentType ?? s.session_id;
                const isRunning = s.effective_status === "running";
                const isFailed = s.effective_status === "failed" || s.error_count > 0;
                const statusLabel = isRunning ? "active" : isFailed ? "errored" : "idle";
                const statusColor = isFailed ? "#F38BA8" : isRunning ? "#A6E3A1" : "#585B70";
                return (
                  <div key={`session-${s.session_id}`} style={{ display: "flex", alignItems: "center", gap: 8, margin: "8px 0 12px", fontSize: 11, whiteSpace: "nowrap" }}>
                    <ActorDot name={actorName} live={isRunning} />
                    <span style={{ color: "#6C7086", flexShrink: 0 }}>{actorName}</span>
                    <span style={{ color: statusColor, flexShrink: 0 }}>{statusLabel}</span>
                    {s.error_count > 0 && <span style={{ color: "#F38BA8", flexShrink: 0 }}>{s.error_count} error{s.error_count !== 1 ? "s" : ""}</span>}
                    {(s.tokens_in + s.tokens_out) > 0 && <span style={{ color: "#45475A", flexShrink: 0 }}>{(s.tokens_in + s.tokens_out).toLocaleString()} tok</span>}
                    {s.last_activity_at && <span style={{ color: "#45475A", flexShrink: 0 }}>{timeAgo(s.last_activity_at)}</span>}
                    <span style={{ flex: 1, height: 1, background: "#2A2A3E", minWidth: 24 }} />
                  </div>
                );
              }

              // ── Evidence cards: an actor (agent or you) searched memory ──
              // Normalize agentQuery + userQuery into one card shape so the
              // you-vs-agent distinction is purely the actor, never the layout.
              const isUser = ev.kind === "userQuery";
              const key = ev.kind === "userQuery" ? `uq-${ev.id}` : `aq-${ev.sessionId}-${ev.ts}-${ev.question.slice(0, 24)}`;
              const actor = ev.kind === "userQuery" ? "You" : ev.label;
              const query = ev.kind === "userQuery" ? ev.query : ev.question;
              const ts = ev.ts;

              let outcome: Outcome;
              let reason = "";
              let body: React.ReactNode = null;
              if (ev.kind === "userQuery") {
                const scope = selectedDataset ? selectedDataset.name : `${datasets.length} ${datasets.length === 1 ? "brain" : "brains"}`;
                if (ev.searching) { outcome = "running"; }
                else if (ev.error) { outcome = "error"; reason = ev.errorMessage || "check your connection"; }
                else if (ev.results && ev.results.length > 0) {
                  outcome = "hit";
                  reason = `${ev.results.length} ${ev.results.length === 1 ? "brain" : "brains"} answered · searched ${scope}`;
                  body = (
                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                      {ev.results.map((r, i) => (
                        <div key={i}>
                          {r.dataset && <div style={{ color: "#89B4FA", marginBottom: 2 }}>[{r.dataset}]</div>}
                          <div className="term-md"><ReactMarkdown>{r.text}</ReactMarkdown></div>
                        </div>
                      ))}
                    </div>
                  );
                } else { outcome = "empty"; reason = `no brain returned relevant knowledge · searched ${scope}`; }
              } else {
                const isRemember = ev.source === "remember";
                if (ev.answer && !isNoAnswer(ev.answer)) {
                  outcome = "hit";
                  reason = isRemember ? "agent saved this to memory" : "agent assembled context from memory";
                  body = <div className="term-md"><ReactMarkdown>{ev.answer}</ReactMarkdown></div>;
                } else if (ev.answer) {
                  outcome = "empty";
                  reason = isRemember ? "saved with no usable content" : "recall returned no usable context";
                  body = <span style={{ color: "#6C7086" }}>{ev.answer.length > 200 ? `${ev.answer.slice(0, 200)}…` : ev.answer}</span>;
                } else {
                  outcome = "done";
                  reason = isRemember ? "entry recorded · no answer attached" : "query recorded · answer not captured in trace";
                }
              }

              const expandable = outcome !== "running";
              // Default-open the latest result; once the user clicks any card,
              // honour that explicit selection — including collapsing the
              // auto-expanded card by clicking it again.
              const open = userSelected ? expandedKey === key : key === lastExpandableKey;
              return (
                <div key={key} style={{ marginBottom: 10 }}>
                  <div
                    onClick={() => { if (expandable) { setUserSelected(true); setExpandedKey(open ? null : key); } }}
                    style={{ display: "flex", alignItems: "center", gap: 7, flexWrap: "wrap", cursor: expandable ? "pointer" : "default", borderRadius: 6, padding: "2px 4px", marginLeft: -4 }}
                    onMouseEnter={e => { if (expandable) e.currentTarget.style.background = "rgba(255,255,255,0.03)"; }}
                    onMouseLeave={e => { e.currentTarget.style.background = "transparent"; }}
                  >
                    {expandable ? <Chevron open={open} /> : <span style={{ width: 9 }} />}
                    <ActorDot name={actor} live={outcome === "running"} />
                    <span style={{ color: isUser ? "#CBA6F7" : "#CDD6F4", fontWeight: 700, wordBreak: "break-word" }}>{actor}</span>
                    <span style={{ color: "#585B70" }}>{ev.kind === "agentQuery" ? ev.source : "recall"}</span>
                    <span style={{ color: "#A6E3A1", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 360 }}>&ldquo;{query}&rdquo;</span>
                    <OutcomeBadge outcome={outcome} />
                    <span style={{ color: "#45475A", flexShrink: 0 }}>· {timeAgo(new Date(ts).toISOString())}</span>
                  </div>
                  {/* Collapsed: one-line reason (the evidence summary). Expanded: full body. */}
                  {!open && reason && outcome !== "running" && (
                    <div style={{ paddingLeft: "calc(9px + 7px)", marginTop: 2, color: "#585B70", fontSize: 11 }}>{reason}</div>
                  )}
                  {open && (
                    <div style={{ marginLeft: 16, marginTop: 6, paddingLeft: 12, borderLeft: "2px solid #2A2A3E", display: "flex", flexDirection: "column", gap: 8 }}>
                      <div style={{ color: "#6C7086", fontSize: 11 }}>
                        <span style={{ color: "#45475A" }}>evidence:&nbsp;</span>{reason}
                      </div>
                      {body ?? <span style={{ color: "#45475A" }}>No relevant knowledge found.</span>}
                    </div>
                  )}
                </div>
              );
            })}
            </>
          )}

          {/* Search input prompt — the one and only cursor in this panel */}
          <div style={{
            marginTop: 14,
            borderTop: "1px solid #2A2A3E",
            paddingTop: 10,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#6C7086" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
              <span style={{ ...FONT, fontSize: 10, color: "#6C7086", letterSpacing: "0.06em", textTransform: "uppercase" }}>
                Search your memory
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center" }}>
              <span style={{ color: "#585B70", flexShrink: 0 }}>$&nbsp;❯&nbsp;</span>
              <div style={{ position: "relative", flex: 1, display: "flex", alignItems: "center" }}>
                {!searchInput && (
                  <span style={{ position: "absolute", left: 0, top: "50%", transform: "translateY(-50%)", display: "flex", alignItems: "center", gap: 0, pointerEvents: "none", whiteSpace: "nowrap" }}>
                    <span style={{ color: "#45475A" }}>
                      {cogniInstance ? typedPlaceholder : "Connect an agent to search your memory…"}
                    </span>
                    {/* Cursor only while the typewriter is actively writing/erasing — no
                        idle blinking. Solid block (no animation): typing motion conveys life. */}
                    {cogniInstance && typingActive && (
                      <span style={{ display: "inline-block", width: 7, height: "0.85em", background: "#CBA6F7", marginLeft: 1 }} />
                    )}
                  </span>
                )}
                <input
                  ref={inputRef}
                  className="term-search-input"
                  type="text"
                  value={searchInput}
                  onChange={e => setSearchInput(e.target.value)}
                  onFocus={() => setInputFocused(true)}
                  onBlur={() => setInputFocused(false)}
                  onKeyDown={e => {
                    if (e.key === "Enter") handleSearch(searchInput);
                    if (e.key === "Escape") setSearchInput("");
                  }}
                  disabled={!cogniInstance}
                  style={{
                    flex: 1, background: "transparent", border: "none",
                    ...FONT, color: "#CDD6F4", caretColor: "#CBA6F7",
                    cursor: !cogniInstance ? "not-allowed" : "text",
                  }}
                />
              </div>
            </div>
          </div>

          <div style={{ height: 4 }} />
        </div>
      </div>
    </div>
  );
}
