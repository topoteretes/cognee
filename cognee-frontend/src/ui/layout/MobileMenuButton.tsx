"use client";

import { useNavbar } from "./NavbarContext";

export default function MobileMenuButton() {
  const { isOpen, toggle } = useNavbar();

  if (isOpen) return null;

  return (
    <button
      onClick={toggle}
      aria-label="Open navigation"
      style={{
        position: "fixed",
        bottom: "1.25rem",
        left: "1.25rem",
        zIndex: 200,
        width: "2.75rem",
        height: "2.75rem",
        borderRadius: "50%",
        background: "#6510f4",
        color: "#fff",
        border: "none",
        cursor: "pointer",
        alignItems: "center",
        justifyContent: "center",
        boxShadow: "0 2px 12px rgba(101,16,244,0.35)",
      }}
      className="flex sm:hidden"
    >
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
        <rect y="3" width="20" height="2" rx="1" fill="currentColor" />
        <rect y="9" width="20" height="2" rx="1" fill="currentColor" />
        <rect y="15" width="20" height="2" rx="1" fill="currentColor" />
      </svg>
    </button>
  );
}
