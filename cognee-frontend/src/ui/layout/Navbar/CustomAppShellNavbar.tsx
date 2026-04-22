"use client";

import { usePathname, useRouter } from "next/navigation";
import { useNavbar } from "../NavbarContext";
import NavbarIconLink from "./NavbarIconLink";
import { ReactNode } from "react";
import { isCloudEnvironment } from "@/utils";
import { useTenant } from "@/modules/tenant/TenantContext";

const PLAN_LABELS: Record<string, string> = {
  developer: "Developer",
  cloud: "Cloud (Team)",
};

// -- Icon components for nav items --

function HouseIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#7C5CFC" : "#6B7280"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  );
}

function DatabaseIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#7C5CFC" : "#6B7280"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
    </svg>
  );
}

function SearchIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#7C5CFC" : "#6B7280"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function GraphIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#7C5CFC" : "#6B7280"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="6" cy="6" r="3" />
      <circle cx="18" cy="6" r="3" />
      <circle cx="12" cy="18" r="3" />
      <line x1="8.5" y1="7.5" x2="10.5" y2="16" />
      <line x1="15.5" y1="7.5" x2="13.5" y2="16" />
    </svg>
  );
}


function ConnectAgentIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#7C5CFC" : "#6B7280"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="11" width="18" height="10" rx="2" /><circle cx="12" cy="5" r="2" /><path d="M12 7v4" /><line x1="8" y1="16" x2="8" y2="16" /><line x1="16" y1="16" x2="16" y2="16" />
    </svg>
  );
}

function ConnectionsIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#7C5CFC" : "#6B7280"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 8h1a4 4 0 0 1 0 8h-1" />
      <path d="M6 8H5a4 4 0 0 0 0 8h1" />
      <line x1="8" y1="12" x2="16" y2="12" />
    </svg>
  );
}

function KeyIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#7C5CFC" : "#6B7280"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4" />
    </svg>
  );
}

function ActivityIcon({ active }: { active: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={active ? "#7C5CFC" : "#6B7280"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  );
}

// -- Navigation data --

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
      { text: "Datasets", link: "/datasets", icon: DatabaseIcon },
    ],
  },
  {
    label: "EXPLORE",
    items: [
      { text: "Search", link: "/search", icon: SearchIcon },
      { text: "Knowledge Graph", link: "/knowledge-graph", icon: GraphIcon },
    ],
  },
  {
    label: "CONFIGURE",
    items: [
      { text: "Connect Agent", link: "/connect-agent", icon: ConnectAgentIcon },
      { text: "Connections", link: "/connections", icon: ConnectionsIcon },
    ],
  },
  {
    label: "DEVELOP",
    items: [
      { text: "API Keys", link: "/api-keys", icon: KeyIcon },
      { text: "Activity", link: "/activity", icon: ActivityIcon },
    ],
  },
];

export default function CustomAppShellNavbar() {
  const pathname = usePathname();
  const router = useRouter();
  const { isOpen, close } = useNavbar();
  const { planType } = useTenant();
  const planLabel = planType ? PLAN_LABELS[planType] ?? "Developer" : "Free";

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
          flex-shrink-0 bg-white border-r border-cognee-border flex flex-col
          fixed sm:relative z-40 sm:z-auto h-full sm:h-auto
          transition-transform sm:translate-x-0
          ${isOpen ? "translate-x-0" : "-translate-x-full sm:translate-x-0"}
        `}
        style={{ width: 240 }}
      >
        {/* Close button on mobile */}
        <div className="flex sm:hidden justify-end p-2">
          <button
            onClick={close}
            aria-label="Close navigation"
            className="cursor-pointer"
            style={{ background: "none", border: "none", fontSize: 20, color: "#555", padding: 4 }}
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
                  fontWeight: 600,
                  letterSpacing: 0.5,
                  color: "#999999",
                  textTransform: "uppercase",
                }}
              >
                {section.label}
              </div>
              {section.items.map((item) => {
                const isActive = pathname === item.link || pathname.startsWith(item.link + "/");
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

        {/* Bottom upgrade section (cloud only) */}
        {isCloudEnvironment() && (
          <div className="px-3 pb-4 mt-auto">
            <div className="flex items-center justify-center gap-[6px] mb-2">
              <span
                className="rounded-[4px]"
                style={{
                  background: "#F0EDFF",
                  fontSize: 11,
                  fontWeight: 500,
                  color: "#6C47FF",
                  padding: "1px 6px",
                }}
              >
                {planLabel}
              </span>
            </div>
            <button
              onClick={() => router.push("/plan")}
              className="w-full rounded-[6px] text-white cursor-pointer"
              style={{
                background: "#6510F4",
                border: "none",
                padding: "8px",
                fontSize: 13,
                fontWeight: 500,
              }}
            >
              View Plan
            </button>
          </div>
        )}
      </aside>
    </>
  );
}
