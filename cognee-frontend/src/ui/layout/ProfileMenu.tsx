"use client";

import { useCallback } from "react";
import Link from "next/link";
import useBoolean from "@/utils/useBoolean";
import useOutsideClick from "@/utils/useOutsideClick";

function PersonIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#333333" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}

function LogoutIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#CC3333" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
      <polyline points="16 17 21 12 16 7" />
      <line x1="21" y1="12" x2="9" y2="12" />
    </svg>
  );
}

interface ProfileMenuProps {
  userName: string;
  userEmail: string;
  profileHref?: string;
  logoutHref?: string;
}

export default function ProfileMenu({ userName, userEmail, profileHref = "/settings", logoutHref = "/api/signout" }: ProfileMenuProps) {
  const { value: isOpen, toggle, setFalse: close } = useBoolean(false);
  const closeCallback = useCallback(() => close(), [close]);
  const containerRef = useOutsideClick<HTMLDivElement>(closeCallback, isOpen);

  const initial = userName ? userName.charAt(0).toUpperCase() : "U";

  return (
    <div ref={containerRef} className="relative">
      {/* Avatar trigger */}
      <button
        onClick={toggle}
        className="flex items-center justify-center rounded-full cursor-pointer"
        style={{ width: 28, height: 28, background: "#6C47FF", border: "none" }}
      >
        <span style={{ fontSize: 13, fontWeight: 500, color: "#fff" }}>{initial}</span>
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div
          className="absolute z-50"
          style={{
            top: 40,
            right: 0,
            width: 220,
            background: "#fff",
            borderRadius: 10,
            boxShadow: "0px 8px 30px #0000001F, 0px 0px 0px 1px #0000000F",
            padding: 6,
          }}
        >
          {/* User info */}
          <div className="flex items-center gap-[10px] px-3 py-[10px]">
            <div
              className="flex items-center justify-center rounded-full flex-shrink-0"
              style={{ width: 32, height: 32, background: "#6510F4" }}
            >
              <span style={{ fontSize: 14, fontWeight: 500, color: "#fff" }}>{initial}</span>
            </div>
            <div className="min-w-0">
              <div style={{ fontSize: 13, fontWeight: 500, color: "#1A1A1A" }} className="truncate">
                {userName}
              </div>
              <div style={{ fontSize: 12, color: "#999999" }} className="truncate">
                {userEmail}
              </div>
            </div>
          </div>

          <div style={{ height: 1, background: "#EEEEEE", margin: "2px -6px" }} />

          {/* Profile link */}
          <Link
            href={profileHref}
            onClick={close}
            className="flex items-center gap-[10px] rounded-[6px] px-3 py-[10px] hover:bg-cognee-hover"
            style={{ fontSize: 13, color: "#333333", textDecoration: "none" }}
          >
            <PersonIcon />
            Profile
          </Link>

          <div style={{ height: 1, background: "#EEEEEE", margin: "2px -6px" }} />

          {/* Log out — use <a> instead of <Link> to trigger a full page navigation to the API route */}
          <a
            href={logoutHref}
            onClick={close}
            className="flex items-center gap-[10px] rounded-[6px] px-3 py-[10px] hover:bg-cognee-hover"
            style={{ fontSize: 13, color: "#CC3333", textDecoration: "none" }}
          >
            <LogoutIcon />
            Log out
          </a>
        </div>
      )}
    </div>
  );
}
