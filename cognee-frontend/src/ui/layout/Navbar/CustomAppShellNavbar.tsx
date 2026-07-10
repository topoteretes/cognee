"use client";

import { usePathname } from "next/navigation";
import { useNavbar } from "../NavbarContext";
import NavbarIconLink from "./NavbarIconLink";
import { ReactNode } from "react";
import { useTenant } from "@/modules/tenant/TenantContext";
import { useFilter } from "../FilterContext";

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

function SchemaIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#BC9BFF" : "rgba(255,255,255,0.5)"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="8.5" y="14" width="7" height="7" rx="1.5" /><path d="M6.5 10v2.5a1.5 1.5 0 0 0 1.5 1.5h4a1.5 1.5 0 0 0 1.5-1.5V10" />
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
  "/schema",
  "/knowledge-graph",
]);

interface NavItem {
  text: string;
  link: string;
  icon: (props: { active: boolean }) => ReactNode;
}

interface NavSection {
  label: string;
  items: NavItem[];
}

const NAV_SECTIONS: NavSection[] = [
  {
    label: "DATA",
    items: [
      { text: "Overview", link: "/dashboard", icon: HouseIcon },
      { text: "Sessions", link: "/sessions", icon: SessionsIcon },
      { text: "Brain", link: "/datasets", icon: DatabaseIcon },
    ],
  },
  {
    label: "EXPLORE",
    items: [
      { text: "Search", link: "/search", icon: SearchIcon },
      { text: "Skills", link: "/skills", icon: SkillsIcon },
      { text: "Memory Schema", link: "/schema", icon: SchemaIcon },
      { text: "Mindmap", link: "/knowledge-graph", icon: GraphIcon },
    ],
  },
  {
    label: "CONNECT",
    items: [
      { text: "Integrations", link: "/integrations", icon: IntegrationsIcon },
      { text: "API Keys", link: "/api-keys", icon: KeyIcon },
    ],
  },
];

export default function CustomAppShellNavbar() {
  const pathname = usePathname();
  const { isOpen, close } = useNavbar();
  const { tenantReady } = useTenant();
  const { datasets } = useFilter();

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
          flex-shrink-0 flex flex-col
          fixed sm:relative z-40 sm:z-auto h-full sm:h-auto
          transition-transform sm:translate-x-0
          ${isOpen ? "translate-x-0" : "-translate-x-full sm:translate-x-0"}
        `}
        style={{ width: 240, maxHeight: "100vh", overflow: "hidden", background: "rgba(0,0,0,0.6)", backdropFilter: "blur(12px)", borderRight: "1px solid rgba(255,255,255,0.08)" }}
      >
        {/* Close button on mobile */}
        <div className="flex sm:hidden justify-end p-2">
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
              <div
                className="px-3 mb-1"
                style={{
                  fontSize: 11,
                  fontWeight: 700,
                  letterSpacing: 0.5,
                  color: "rgba(255,255,255,0.3)",
                  textTransform: "uppercase",
                }}
              >
                {section.label}
              </div>
              {section.items.map((item) => {
                const isActive = pathname === item.link || pathname.startsWith(item.link + "/");
                const locked = !tenantReady && POD_DEPENDENT_LINKS.has(item.link);
                if (locked) {
                  return (
                    <div
                      key={item.link}
                      title="Available once your workspace is ready"
                      className="flex items-center gap-[10px] rounded-[6px] px-3 py-2 text-[14px]"
                      style={{ color: "rgba(237,236,234,0.3)", cursor: "not-allowed", userSelect: "none" }}
                      aria-disabled="true"
                    >
                      {item.icon({ active: false })}
                      {item.text}
                    </div>
                  );
                }
                return (
                  <NavbarIconLink
                    key={item.link}
                    text={item.text}
                    link={item.link}
                    isActive={isActive}
                    icon={item.icon({ active: isActive })}
                  />
                );
              })}
            </div>
          ))}
        </nav>

      </aside>
    </>
  );
}
