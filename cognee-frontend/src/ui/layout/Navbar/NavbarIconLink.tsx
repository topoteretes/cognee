"use client";

import Link from "next/link";
import classNames from "classnames";
import { ReactNode } from "react";

interface NavbarIconLinkProps {
  text: string;
  icon: ReactNode;
  link: string;
  isActive: boolean;
  external?: boolean;
  collapsed?: boolean;
  onClick?: () => void;
}

export default function NavbarIconLink({
  text,
  icon,
  link,
  isActive,
  external,
  collapsed = false,
  onClick,
}: NavbarIconLinkProps) {
  const classes = classNames(
    "flex items-center rounded-[6px] px-3 py-2 text-[14px] transition-colors",
    // Collapsing only applies to the desktop rail (`sm` and up); the mobile
    // drawer always shows full-width labels.
    collapsed ? "gap-[10px] sm:gap-0 sm:justify-center" : "gap-[10px]",
    {
      "font-medium": isActive,
    }
  );

  // Literal hex on active to bypass Tailwind v4 @theme HMR edge cases that
  // were rendering the active nav item text as black.
  const linkStyle: React.CSSProperties = isActive
    ? { background: "rgba(188,155,255,0.20)", color: "#BC9BFF", textDecoration: "none" }
    : { color: "rgba(237,236,234,0.7)", textDecoration: "none" };

  return (
    <Link
      href={link}
      className={classes}
      style={linkStyle}
      title={collapsed ? text : undefined}
      aria-label={collapsed ? text : undefined}
      {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
      onClick={onClick}
      onMouseEnter={!isActive ? (e) => { (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.06)"; } : undefined}
      onMouseLeave={!isActive ? (e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; } : undefined}
    >
      {icon}
      <span className={classNames("whitespace-nowrap", { "sm:hidden": collapsed })}>{text}</span>
    </Link>
  );
}
