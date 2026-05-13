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
  onClick?: () => void;
}

export default function NavbarIconLink({
  text,
  icon,
  link,
  isActive,
  external,
  onClick,
}: NavbarIconLinkProps) {
  const classes = classNames(
    "flex items-center gap-[10px] rounded-[6px] px-3 py-2 text-[14px] transition-colors",
    {
      "bg-[#F0EDFF] text-[#6C47FF] font-medium": isActive,
      "text-[#333333] hover:bg-[#F4F4F5]": !isActive,
    }
  );

  return (
    <Link
      href={link}
      className={classes}
      style={{ textDecoration: "none" }}
      {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
      onClick={onClick}
    >
      {icon}
      {text}
    </Link>
  );
}
