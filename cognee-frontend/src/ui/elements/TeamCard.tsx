"use client";

import type { CSSProperties, ReactElement } from "react";
import { avatarColor } from "@/utils/avatarColor";

export interface TeamInfo {
  /** The cognee role id — also the principal id used to grant dataset access. */
  id: string;
  name: string;
  desc: string;
  members: string[];
  /** Headcount from the roles API, which can exceed the members actually listed. */
  memberCount: number;
  // Card fill — the careers-page role-card purple ramp (cognee.ai/careers).
  color: string;
}

// Card ramp for teams, keyed on role id by the teams module.
export const TEAM_COLORS = ["#BC9BFF", "#A380EA", "#916DD9", "#7E58C8"] as const;

// Card surface — black with white name, grey supporting text; the team's
// purple appears only in the member avatars.
const CARD_BG = "#000000";
const CARD_BORDER = "rgba(255,255,255,0.08)";
const TITLE = "#EDECEA";
const GREY = "rgba(237,236,234,0.55)";
const GREY_FAINT = "rgba(237,236,234,0.4)";

/** A team block: white name, grey description, team-coloured member initials
 *  with the headcount to their right. Renders as a bordered square card by
 *  default; `flat` drops the box (no fill, border or fixed shape) for use
 *  inside list columns that draw their own separators. Share/delete actions
 *  render in the top-right corner when their handlers are provided. */
export default function TeamCard({
  team,
  flat = false,
  hideDesc = false,
  connected = false,
  onShare,
  onDelete,
}: {
  team: TeamInfo;
  flat?: boolean;
  /** Compact variant (dashboard overview): name and members only. */
  hideDesc?: boolean;
  /** Lavender stroke — the card is wired to the memory core in the flow diagram. */
  connected?: boolean;
  onShare?: () => void;
  onDelete?: () => void;
}): ReactElement {
  const boxStyle: CSSProperties = flat
    ? { padding: "8px 14px 10px", display: "flex", flexDirection: "column", gap: 5 }
    : {
        // No fixed shape — stretches to the height of its grid row, so the
        // dashboard cards match the side chip columns.
        minHeight: 0,
        borderRadius: 0,
        background: CARD_BG,
        // Matches the flow-diagram spokes (FlowEdges strokeFor "connected").
        border: connected ? "1px solid rgba(188,155,255,0.55)" : `1px solid ${CARD_BORDER}`,
        padding: "10px 12px",
        display: "flex", flexDirection: "column", justifyContent: "space-between", gap: 6,
      };
  return (
    <div style={boxStyle}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <TeamGlyph name={team.name} color={GREY} size={14} />
        <span style={{ fontSize: 15, fontWeight: 300, letterSpacing: "-0.014em", color: TITLE }}>{team.name}</span>
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 4 }}>
          {onShare && (
            <button
              onClick={onShare}
              className="hover:bg-white/10 cursor-pointer"
              style={{ background: "transparent", border: "none", borderRadius: 0, padding: 4, display: "grid", placeItems: "center", color: GREY }}
              title={`Share with ${team.name}`}
              aria-label={`Share with ${team.name}`}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" />
                <path d="M8.6 13.5l6.8 4M15.4 6.5l-6.8 4" />
              </svg>
            </button>
          )}
          {onDelete && (
            <button
              onClick={onDelete}
              className="hover:bg-white/10 cursor-pointer"
              style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 0, padding: "3px 10px", fontSize: 11, fontWeight: 500, color: "rgba(237,236,234,0.7)" }}
              title={`Remove ${team.name}'s access`}
            >
              Delete
            </button>
          )}
        </span>
      </div>
      {!hideDesc && <div style={{ fontSize: 11, lineHeight: 1.45, fontWeight: 300, color: GREY }}>{team.desc}</div>}
      <MemberRow team={team} flat={flat} />
    </div>
  );
}

// At most this many avatars render; the rest are summarised as "+N people"
// right next to the icons.
const MAX_AVATARS = 4;

function MemberRow({ team, flat }: { team: TeamInfo; flat: boolean }): ReactElement {
  const visible = team.members.slice(0, MAX_AVATARS);
  const overflow = team.memberCount - visible.length;
  const label = overflow > 0 ? `+${overflow} people` : `${team.memberCount} people`;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap", marginTop: flat ? 18 : 0 }}>
      {visible.map((m) => (
        <MemberDot key={m} name={m} color={avatarColor(m)} />
      ))}
      <span style={{ marginLeft: 4, fontSize: 11, fontWeight: 300, color: GREY_FAINT }}>{label}</span>
    </div>
  );
}

/** Tiny initials avatar for a team member — team-coloured chip on the dark fill. */
function MemberDot({ name, color }: { name: string; color: string }): ReactElement {
  const initials = name.split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase();
  return (
    <span
      title={name}
      style={{
        width: 20, height: 20, borderRadius: "50%",
        background: color,
        display: "inline-grid", placeItems: "center",
        fontSize: 8.5, fontWeight: 600, color: "#1e1e1c", letterSpacing: "0.02em",
      }}
    >
      {initials}
    </span>
  );
}

/** Minimal line glyph per team. */
function TeamGlyph({ name, color, size }: { name: string; color: string; size: number }): ReactElement {
  const common = { fill: "none", stroke: color, strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      {name === "Design" && (
        <>
          <path {...common} d="M12 19l7-7 3 3-7 7-3-3z" />
          <path {...common} d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
          <path {...common} d="M2 2l7.586 7.586" />
          <circle {...common} cx="11" cy="11" r="2" />
        </>
      )}
      {name === "Sales" && (
        <>
          <path {...common} d="M3 17l6-6 4 4 8-8" />
          <path {...common} d="M14 7h7v7" />
        </>
      )}
      {name === "Research" && (
        <>
          <circle {...common} cx="11" cy="11" r="7" />
          <path {...common} d="M21 21l-4.35-4.35" />
        </>
      )}
      {name === "Engineers" && (
        <>
          <path {...common} d="M16 18l6-6-6-6" />
          <path {...common} d="M8 6l-6 6 6 6" />
        </>
      )}
      {/* Roles are named by the workspace, so most teams land here. */}
      {!NAMED_GLYPHS.has(name) && (
        <>
          <path {...common} d="M17 20v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
          <circle {...common} cx="9.5" cy="7" r="3.5" />
          <path {...common} d="M22 20v-2a4 4 0 0 0-3-3.87" />
        </>
      )}
    </svg>
  );
}

const NAMED_GLYPHS = new Set(["Design", "Sales", "Research", "Engineers"]);
