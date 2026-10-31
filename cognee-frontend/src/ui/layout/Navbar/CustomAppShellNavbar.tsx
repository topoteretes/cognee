"use client";

import { usePathname } from "next/navigation";
import { useNavbar } from "../NavbarContext";
import NavbarFooter from "./NavbarFooter";
import NavbarNavItem from "./NavbarNavItem";
import NavbarSectionLabel from "./NavbarSectionLabel";
import { useTenant } from "@/modules/tenant/TenantContext";
import type { NavItem } from "./NavbarNavItem";

// Sidebar widths (px). The rail shows icons only; collapsing only applies on
// desktop, matching the Tailwind `sm` breakpoint (640px) used below.
const EXPANDED_WIDTH = 240;
const COLLAPSED_WIDTH = 72;

// -- Icon components for nav items --

function HouseIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#BC9BFF" : "rgba(255,255,255,0.5)"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  );
}

function SessionsIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#BC9BFF" : "rgba(255,255,255,0.5)"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H8l-5 4z" /><line x1="7" y1="8" x2="15" y2="8" /><line x1="7" y1="12" x2="12" y2="12" />
    </svg>
  );
}

function DatabaseIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#BC9BFF" : "rgba(255,255,255,0.5)"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
    </svg>
  );
}

function SearchIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#BC9BFF" : "rgba(255,255,255,0.5)"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function GraphIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#BC9BFF" : "rgba(255,255,255,0.5)"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="6" cy="6" r="3" />
      <circle cx="18" cy="6" r="3" />
      <circle cx="12" cy="18" r="3" />
      <line x1="8.5" y1="7.5" x2="10.5" y2="16" />
      <line x1="15.5" y1="7.5" x2="13.5" y2="16" />
    </svg>
  );
}



function IntegrationsIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#BC9BFF" : "rgba(255,255,255,0.5)"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="2" width="9" height="9" rx="2" /><rect x="13" y="2" width="9" height="9" rx="2" /><rect x="2" y="13" width="9" height="9" rx="2" /><path d="M17.5 13.5v3m0 0v3m0-3h3m-3 0h-3" />
    </svg>
  );
}

function SkillsIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#BC9BFF" : "rgba(255,255,255,0.5)"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18v3h3l6.3-6.3a4 4 0 0 0 5.4-5.4l-2.8 2.8-2.1-2.1z" />
    </svg>
  );
}

function KeyIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#BC9BFF" : "rgba(255,255,255,0.5)"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4" />
    </svg>
  );
}

// -- Navigation data --

// Routes that require the tenant pod — dimmed/locked while it provisions.
const POD_DEPENDENT_LINKS = new Set([
  "/sessions",
  "/datasets",
  "/search",
  "/skills",
  "/knowledge-graph",
]);

interface NavSection {
  label: string;
  items: NavItem[];
}

const NAV_SECTIONS: NavSection[] = [
  {
    label: "General",
    items: [
      { text: "Overview", link: "/dashboard", icon: HouseIcon },
      {
        text: "Datasets",
        link: "/datasets",
        icon: DatabaseIcon,
        // Both live inside datasets: skills are scoped to one, and sessions
        // become dataset data. The submenu only unfolds inside the subtree.
        children: [
          { text: "Sessions", link: "/sessions", icon: SessionsIcon },
          { text: "Skills", link: "/skills", icon: SkillsIcon },
        ],
      },
      { text: "Graph", link: "/knowledge-graph", icon: GraphIcon },
      { text: "Search", link: "/search", icon: SearchIcon },
    ],
  },
  {
    label: "Connect",
    items: [
      { text: "Integrations", link: "/integrations", icon: IntegrationsIcon },
      { text: "API Keys", link: "/api-keys", icon: KeyIcon },
    ],
  },
];

export default function CustomAppShellNavbar() {
  const pathname = usePathname();
  const { isOpen, close, collapsed, toggleCollapsed } = useNavbar();
  const { tenantReady } = useTenant();

  // Collapsing to an icon rail only applies on desktop. On mobile the sidebar
  // is a full-width drawer toggled by the hamburger (isOpen), so whenever the
  // drawer is open it shows full labels. Deriving `railed` from isOpen instead
  // of a media query keeps it deterministic on the server, so the first paint
  // already has the correct width (no flash on hard navigation).
  const railed = collapsed && !isOpen;

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-30 sm:hidden"
          onClick={close}
        />
      )}

      <aside
        className={`
          group flex-shrink-0 flex flex-col
          fixed sm:relative z-40 sm:z-auto h-full sm:h-auto
          transition-[transform,width] sm:translate-x-0
          ${isOpen ? "translate-x-0" : "-translate-x-full sm:translate-x-0"}
        `}
        style={{ width: railed ? COLLAPSED_WIDTH : EXPANDED_WIDTH, maxHeight: "100vh", overflow: "hidden", background: "rgba(0,0,0,0.6)", backdropFilter: "blur(12px)", borderRight: "1px solid rgba(255,255,255,0.08)" }}
      >
        {/* Desktop-only collapse handle: a thin control on the sidebar's right
            edge that fades in on hover (Notion-style). No dedicated header row. */}
        <button
          onClick={toggleCollapsed}
          aria-label={railed ? "Expand sidebar" : "Collapse sidebar"}
          title={railed ? "Expand sidebar" : "Collapse sidebar"}
          className="cursor-pointer hidden sm:flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
          style={{ position: "absolute", top: 8, right: 6, width: 18, height: 44, borderRadius: 6, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)", zIndex: 10 }}
          onMouseEnter={e => ((e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.12)")}
          onMouseLeave={e => ((e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.06)")}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: railed ? "rotate(180deg)" : undefined }}>
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>

        {/* Mobile-only close button (the desktop collapse control is the hover
            handle above, so there is no desktop header row). */}
        <div className="flex sm:hidden items-center justify-end flex-shrink-0 px-3" style={{ height: 40 }}>
          <button
            onClick={close}
            aria-label="Close navigation"
            className="cursor-pointer"
            style={{ background: "none", border: "none", fontSize: 20, color: "rgba(255,255,255,0.6)", padding: 4 }}
          >
            &#10005;
          </button>
        </div>

        {/* Nav sections */}
        <nav className="flex-1 overflow-y-auto px-3 py-2">
          {NAV_SECTIONS.map((section) => (
            <div key={section.label} className="mb-4">
              <NavbarSectionLabel label={section.label} railed={railed} />
              {section.items.map((item) => (
                <NavbarNavItem
                  key={item.link}
                  item={item}
                  pathname={pathname}
                  railed={railed}
                  isLocked={(link) => !tenantReady && POD_DEPENDENT_LINKS.has(link)}
                />
              ))}
            </div>
          ))}
        </nav>
        <NavbarFooter railed={railed} />
      </aside>
    </>
  );
}
