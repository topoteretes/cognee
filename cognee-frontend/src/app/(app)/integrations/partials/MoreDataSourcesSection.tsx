"use client";

import { useState, type ReactElement } from "react";
import { trackEvent } from "@/modules/analytics";

// Mirrors the native integrations offered in Claude Desktop's connector
// directory, so the false-door list measures demand against a set users already
// recognize from Claude. Ordered by expected demand for a team knowledge base:
// shared comms and docs first, niche/vertical tools last.
// `logo` is a vendored Simple Icons (CC0) glyph rendered white on the brand
// tile; entries without one fall back to their initials.
//
// Slack is deliberately absent — it's real (see DataSourceSection, rendered
// above this section) — so it isn't shown twice.
interface DataSource {
  name: string;
  description: string;
  initials: string;
  color: string;
  logo?: string;
}

const INTEGRATIONS: DataSource[] = [
  { name: "Notion",       description: "Pages and databases",         initials: "No", color: "#000000", logo: "notion" },
  { name: "Google Drive", description: "Docs, Sheets, and Slides",    initials: "GD", color: "#1A73E8", logo: "googledrive" },
  { name: "Confluence",   description: "Spaces and wikis",            initials: "Cf", color: "#172B4D", logo: "confluence" },
  { name: "GitHub",       description: "Issues, PRs, and docs",       initials: "GH", color: "#181717", logo: "github" },
  { name: "Gmail",        description: "Email threads and files",     initials: "Gm", color: "#EA4335", logo: "gmail" },
  { name: "Jira",         description: "Tickets and epics",           initials: "Jr", color: "#0052CC", logo: "jira" },
  { name: "Linear",       description: "Issues and project context",  initials: "Li", color: "#5E6AD2", logo: "linear" },
  { name: "Granola",      description: "Meeting notes and transcripts",initials: "Gr", color: "#C2410C", logo: "granola" },
  { name: "Asana",        description: "Tasks and projects",          initials: "As", color: "#F06A6A", logo: "asana" },
  { name: "monday.com",   description: "Boards, items, and updates",  initials: "Mo", color: "#FF3D57", logo: "monday" },
  { name: "Figma",        description: "Design files and comments",   initials: "Fg", color: "#F24E1E", logo: "figma" },
  { name: "HubSpot",      description: "Contacts, deals, and notes",  initials: "Hs", color: "#FF7A59", logo: "hubspot" },
  { name: "Intercom",     description: "Conversations and articles",  initials: "In", color: "#1F8DED", logo: "intercom" },
  { name: "Box",          description: "Documents and files",         initials: "Bx", color: "#0061D5", logo: "box" },
  { name: "Canva",        description: "Designs and brand assets",    initials: "Cv", color: "#00C4CC", logo: "canva" },
  { name: "PostHog",      description: "Product analytics",           initials: "Ph", color: "#F54E00", logo: "posthog" },
  { name: "Stripe",       description: "Customers and invoices",      initials: "St", color: "#635BFF", logo: "stripe" },
  { name: "Vercel",       description: "Projects and deployments",    initials: "Ve", color: "#000000", logo: "vercel" },
  { name: "MotherDuck",   description: "Your data warehouse",         initials: "Md", color: "#F9A825", logo: "motherduck" },
  { name: "Xero",         description: "Accounting and invoices",     initials: "Xe", color: "#13B5EA", logo: "xero" },
  { name: "lemlist",      description: "Outreach campaigns",          initials: "Ll", color: "#4F46E5", logo: "lemlist" },
  { name: "Workable",     description: "Candidates and pipelines",    initials: "Wk", color: "#16A34A", logo: "workable" },
];

// These data sources aren't built yet. Each card presents as a real, available
// connector (no "coming soon" telegraph) so a Connect click measures genuine
// intent; the click opens a Coming Soon modal and fires analytics so we can
// decide what to build next. Both interactions are tracked (see tracking-plan).

function ComingSoonModal({
  source,
  onClose,
}: {
  source: DataSource;
  onClose: () => void;
}): ReactElement {
  const [notified, setNotified] = useState(false);

  function requestNotify() {
    trackEvent({
      pageName: "Integrations",
      eventName: "data_source_notify_requested",
      additionalProperties: { data_source: source.name },
    });
    setNotified(true);
  }

  return (
    <div role="dialog" aria-modal="true" onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 }}>
      <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", borderRadius: 14, width: 440, maxWidth: "100%", boxShadow: "0 20px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.1)", overflow: "hidden" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "16px 20px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: source.color, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
            <span style={{ color: "#fff", fontSize: 12, fontWeight: 700 }}>{source.initials}</span>
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#EDECEA", lineHeight: "20px" }}>Coming soon</div>
            <div style={{ fontSize: 12, color: "rgba(237,236,234,0.45)", marginTop: 1 }}>{source.name}</div>
          </div>
          <button onClick={onClose} aria-label="Close" style={{ background: "none", border: "none", color: "rgba(237,236,234,0.65)", cursor: "pointer", padding: 4, borderRadius: 6, lineHeight: 1 }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
          </button>
        </div>
        <div style={{ padding: "18px 20px 20px" }}>
          {notified ? (
            <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
              <svg width="18" height="18" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0, marginTop: 1 }}><circle cx="8" cy="8" r="7" stroke="#22C55E" strokeWidth="1.3" /><path d="M5 8.2L7 10.2L11 5.8" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
              <p style={{ margin: 0, fontSize: 14, color: "rgba(237,236,234,0.75)", lineHeight: 1.6 }}>
                Thanks — we’ll email you the moment the <strong style={{ color: "#EDECEA" }}>{source.name}</strong> integration is live.
              </p>
            </div>
          ) : (
            <>
              <p style={{ margin: "0 0 16px", fontSize: 14, color: "rgba(237,236,234,0.6)", lineHeight: 1.6 }}>
                We can let you know once the <strong style={{ color: "#EDECEA" }}>{source.name}</strong> integration is live to upgrade your Company Brain.
              </p>
              <button onClick={requestNotify} style={{ display: "inline-flex", alignItems: "center", gap: 6, background: "#6510F4", border: "none", borderRadius: 8, padding: "9px 16px", fontSize: 13.5, fontWeight: 600, color: "#fff", cursor: "pointer", fontFamily: "inherit" }}>
                Get notified once live
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default function MoreDataSourcesSection(): ReactElement {
  const [activeName, setActiveName] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const activeSource = INTEGRATIONS.find((it) => it.name === activeName) ?? null;

  const q = query.trim().toLowerCase();
  const filtered = q
    ? INTEGRATIONS.filter((it) => it.name.toLowerCase().includes(q) || it.description.toLowerCase().includes(q))
    : INTEGRATIONS;

  function openSource(source: DataSource) {
    trackEvent({
      pageName: "Integrations",
      eventName: "data_source_connect_clicked",
      additionalProperties: { data_source: source.name },
    });
    setActiveName(source.name);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <style>{`
        .ds-card { transition: transform 150ms ease, border-color 150ms ease, background 150ms ease; }
        .ds-card:hover { transform: translateY(-2px); border-color: rgba(188,155,255,0.35) !important; background: rgba(255,255,255,0.09) !important; }
        .ds-connect { opacity: 0; transform: translateX(4px); transition: opacity 150ms ease, transform 150ms ease; }
        .ds-card:hover .ds-connect, .ds-card:focus-visible .ds-connect { opacity: 1; transform: translateX(0); }
        @media (prefers-reduced-motion: reduce) {
          .ds-card, .ds-connect { transition: none; }
          .ds-card:hover { transform: none; }
        }
        .ds-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(224px, 1fr)); gap: 14px; }
        .ds-search { transition: border-color 150ms ease, background 150ms ease; }
        .ds-search::placeholder { color: rgba(237,236,234,0.4); }
        .ds-search:focus { outline: none; border-color: rgba(188,155,255,0.5); background: rgba(255,255,255,0.09); }
      `}</style>

      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: "0 0 4px", letterSpacing: "-0.01em" }}>More data sources</h2>
          <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Not live yet — tell us which ones to prioritize.</p>
        </div>
        <div style={{ position: "relative", width: 240, maxWidth: "100%" }}>
          <svg width="15" height="15" viewBox="0 0 16 16" fill="none" style={{ position: "absolute", left: 11, top: "50%", marginTop: -7.5, pointerEvents: "none" }}>
            <circle cx="7" cy="7" r="4.5" stroke="rgba(237,236,234,0.45)" strokeWidth="1.4" />
            <path d="M10.5 10.5L14 14" stroke="rgba(237,236,234,0.45)" strokeWidth="1.4" strokeLinecap="round" />
          </svg>
          <input
            className="ds-search"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search data sources"
            aria-label="Search data sources"
            style={{ width: "100%", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 9, padding: "8px 12px 8px 32px", fontSize: 13, color: "#EDECEA", fontFamily: "inherit" }}
          />
        </div>
      </div>

      {filtered.length > 0 ? (
        <div className="ds-grid">
          {filtered.map((it) => (
            <button key={it.name} className="ds-card" onClick={() => openSource(it)} style={{ position: "relative", display: "flex", alignItems: "center", gap: 12, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: "13px 15px", textAlign: "left", cursor: "pointer", fontFamily: "inherit", width: "100%" }}>
              <div style={{ width: 36, height: 36, borderRadius: 10, background: it.color, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.08)" }}>
                {it.logo ? (
                  <span aria-hidden style={{ width: 19, height: 19, background: "#fff", WebkitMaskImage: `url(/visuals/logos/datasources/${it.logo}.svg)`, maskImage: `url(/visuals/logos/datasources/${it.logo}.svg)`, WebkitMaskRepeat: "no-repeat", maskRepeat: "no-repeat", WebkitMaskPosition: "center", maskPosition: "center", WebkitMaskSize: "contain", maskSize: "contain" }} />
                ) : (
                  <span style={{ color: "#fff", fontSize: 13, fontWeight: 700, letterSpacing: "-0.02em" }}>{it.initials}</span>
                )}
              </div>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA", fontFamily: '"TWKLausanne", sans-serif', overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", lineHeight: 1.3 }}>{it.name}</div>
                <div style={{ fontSize: 12, color: "rgba(237,236,234,0.45)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", lineHeight: 1.4 }}>{it.description}</div>
              </div>
              <span className="ds-connect" style={{ position: "absolute", right: 11, top: "50%", marginTop: -12, display: "inline-flex", alignItems: "center", gap: 3, background: "rgba(24,24,27,0.95)", backdropFilter: "blur(8px)", border: "1px solid rgba(188,155,255,0.4)", borderRadius: 6, padding: "3px 8px", fontSize: 11, fontWeight: 600, color: "#BC9BFF" }}>
                Connect
                <svg width="11" height="11" viewBox="0 0 16 16" fill="none"><path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </span>
            </button>
          ))}
        </div>
      ) : (
        <div style={{ textAlign: "center", padding: "28px 16px", fontSize: 13.5, color: "rgba(237,236,234,0.5)", lineHeight: 1.6 }}>
          No data sources match “{query.trim()}”.{" "}
          <a href={`mailto:support@cognee.ai?subject=${encodeURIComponent(`Integration request: ${query.trim()}`)}`} style={{ color: "#6510F4", textDecoration: "underline" }}>Request it</a> and we’ll consider it next.
        </div>
      )}

      <p style={{ fontSize: 13, color: "rgba(237,236,234,0.35)", textAlign: "center", margin: 0 }}>
        More integrations on the way.{" "}
        <a href="mailto:support@cognee.ai?subject=Integration%20request" style={{ color: "#6510F4", textDecoration: "underline" }}>Let us know</a> what to prioritize.
      </p>

      {activeSource && <ComingSoonModal source={activeSource} onClose={() => setActiveName(null)} />}
    </div>
  );
}
