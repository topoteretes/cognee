"use client";

import { useCallback } from "react";
import Link from "next/link";
import useBoolean from "@/utils/useBoolean";
import useOutsideClick from "@/utils/useOutsideClick";

function DocsIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#333333" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
    </svg>
  );
}

function EnvelopeIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#333333" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="4" width="20" height="16" rx="2" />
      <polyline points="22,4 12,13 2,4" />
    </svg>
  );
}

function DiscordIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="#333333">
      <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057c.002.022.015.043.032.054a19.9 19.9 0 0 0 5.993 3.03.077.077 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z" />
    </svg>
  );
}

function KeyboardIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#333333" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="4" width="20" height="16" rx="2" ry="2" />
      <line x1="6" y1="8" x2="6.01" y2="8" />
      <line x1="10" y1="8" x2="10.01" y2="8" />
      <line x1="14" y1="8" x2="14.01" y2="8" />
      <line x1="18" y1="8" x2="18.01" y2="8" />
      <line x1="8" y1="12" x2="8.01" y2="12" />
      <line x1="12" y1="12" x2="12.01" y2="12" />
      <line x1="16" y1="12" x2="16.01" y2="12" />
      <line x1="7" y1="16" x2="17" y2="16" />
    </svg>
  );
}

function MeetingIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#333333" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
      <path d="M10 14l2 2 4-4" />
    </svg>
  );
}

function StatusIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#333333" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </svg>
  );
}

const MENU_ITEMS = [
  { label: "Docs", href: "https://docs.cognee.ai", external: true, icon: <DocsIcon /> },
  { label: "Contact us", href: "mailto:social@cognee.ai", external: true, icon: <EnvelopeIcon /> },
  { label: "Talk to us", href: "https://calendly.com/luca-topoteretes/new-meeting", external: true, icon: <MeetingIcon /> },
  { label: "Discord community", href: "https://discord.gg/m63hxKsp4p", external: true, icon: <DiscordIcon /> },
];

const CHANGELOG_ITEMS = [
  { label: "Improved graph visualization", date: "Mar 28" },
  { label: "New dataset import flow", date: "Mar 15" },
];

export default function HelpMenu() {
  const { value: isOpen, toggle, setFalse: close } = useBoolean(false);
  const closeCallback = useCallback(() => close(), [close]);
  const containerRef = useOutsideClick<HTMLDivElement>(closeCallback, isOpen);

  return (
    <div ref={containerRef} className="relative">
      {/* Help trigger button */}
      <button
        onClick={toggle}
        className="flex items-center justify-center rounded-full border-[1.5px] border-cognee-border cursor-pointer bg-white hover:bg-cognee-hover"
        style={{ width: 28, height: 28 }}
      >
        <span style={{ fontSize: 12, fontWeight: 600, color: "#A1A1AA" }}>?</span>
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div
          className="absolute z-50"
          style={{
            top: 40,
            right: 0,
            width: 240,
            background: "#fff",
            borderRadius: 10,
            boxShadow: "0px 8px 30px #0000001F, 0px 0px 0px 1px #0000000F",
            padding: 6,
          }}
        >
          {MENU_ITEMS.map((item) => (
            <Link
              key={item.label}
              href={item.href}
              target="_blank"
              rel="noopener noreferrer"
              onClick={close}
              className="flex items-center gap-[10px] rounded-[6px] px-3 py-[10px] hover:bg-cognee-hover"
              style={{ fontSize: 13, color: "#333333", textDecoration: "none" }}
            >
              {item.icon}
              {item.label}
            </Link>
          ))}

          {/* Separator */}
          <div style={{ height: 1, background: "#EEEEEE", margin: "2px -6px" }} />

          {/* Keyboard shortcuts */}
          <div
            className="flex items-center justify-between gap-[10px] rounded-[6px] px-3 py-[10px] hover:bg-cognee-hover cursor-default"
            style={{ fontSize: 13, color: "#333333" }}
          >
            <div className="flex items-center gap-[10px]">
              <KeyboardIcon />
              Keyboard shortcuts
            </div>
            <kbd
              className="flex items-center justify-center rounded-[4px]"
              style={{
                background: "#F5F5F5",
                border: "1px solid #E5E5E5",
                color: "#999999",
                fontSize: 11,
                padding: "2px 6px",
                fontFamily: "inherit",
              }}
            >
              &#8984; /
            </kbd>
          </div>

          {/* System status */}
          <div
            className="flex items-center gap-[10px] rounded-[6px] px-3 py-[10px] hover:bg-cognee-hover cursor-default"
            style={{ fontSize: 13, color: "#333333" }}
          >
            <StatusIcon />
            System status
            <span
              className="ml-auto rounded-full"
              style={{ width: 10, height: 10, background: "#22C55E", flexShrink: 0 }}
            />
          </div>

          {/* Separator */}
          <div style={{ height: 1, background: "#EEEEEE", margin: "2px -6px" }} />

          {/* What's new */}
          <div style={{ padding: "8px 12px 4px", fontSize: 11, fontWeight: 500, color: "#999999" }}>
            {"What's new"}
          </div>
          {CHANGELOG_ITEMS.map((item) => (
            <div
              key={item.label}
              className="flex items-start gap-[10px] rounded-[6px] px-3 py-[8px] cursor-default"
              style={{ fontSize: 13, color: "#333333" }}
            >
              <span
                className="mt-[5px] rounded-full flex-shrink-0"
                style={{
                  width: 6,
                  height: 6,
                  border: "1.5px solid #6510F4",
                  background: "transparent",
                }}
              />
              <div>
                <div>{item.label}</div>
                <div style={{ fontSize: 11, color: "#999999" }}>{item.date}</div>
              </div>
            </div>
          ))}
          <Link
            href="#"
            className="block px-3 py-[6px] pb-[8px]"
            style={{ fontSize: 12, color: "#6C47FF", textDecoration: "none" }}
          >
            Full changelog
          </Link>
        </div>
      )}
    </div>
  );
}
