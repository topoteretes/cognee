"use client";

import NavbarIconLink from "./NavbarIconLink";
import type { ReactNode } from "react";

export interface NavItem {
  text: string;
  link: string;
  icon: (props: { active: boolean }) => ReactNode;
  /** Sub-pages rendered indented under this item while its subtree is active. */
  children?: NavItem[];
}

interface NavbarNavItemProps {
  item: NavItem;
  pathname: string;
  railed: boolean;
  isLocked: (link: string) => boolean;
}

function matches(pathname: string, link: string): boolean {
  return pathname === link || pathname.startsWith(link + "/");
}

export default function NavbarNavItem({ item, pathname, railed, isLocked }: NavbarNavItemProps): React.JSX.Element {
  const isActive = matches(pathname, item.link);
  const children = item.children ?? [];
  // A submenu stays collapsed until the user is inside the parent's subtree —
  // its children only add depth once they are one click away anyway.
  const expanded = isActive || children.some((child) => matches(pathname, child.link));

  if (isLocked(item.link)) {
    return (
      <div
        title={railed ? `${item.text} — available once your workspace is ready` : "Available once your workspace is ready"}
        className={`flex items-center gap-[10px] rounded-[6px] px-3 py-2 text-[14px] ${railed ? "justify-center" : ""}`}
        style={{ color: "rgba(237,236,234,0.3)", cursor: "not-allowed", userSelect: "none" }}
        aria-disabled="true"
      >
        {item.icon({ active: false })}
        {!railed && item.text}
      </div>
    );
  }

  return (
    <>
      <NavbarIconLink
        text={item.text}
        link={item.link}
        isActive={isActive}
        collapsed={railed}
        icon={item.icon({ active: isActive })}
      />
      {expanded &&
        children.map((child) => (
          <div key={child.link} style={railed ? undefined : { paddingLeft: 14 }}>
            <NavbarNavItem item={child} pathname={pathname} railed={railed} isLocked={isLocked} />
          </div>
        ))}
    </>
  );
}
