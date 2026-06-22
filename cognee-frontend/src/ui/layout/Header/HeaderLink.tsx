"use client";

import Link from "next/link";
import { PropsWithChildren } from "react";
import { Text } from "@mantine/core";
import { usePathname } from "next/navigation";

interface HeaderLinkProps extends PropsWithChildren {
  link?: string;
  isActive?: boolean;
}

export default function HeaderLink({
  link,
  isActive,
  children,
}: HeaderLinkProps) {
  const pathname = usePathname();
  const isActiveByPathname = link !== undefined && pathname.includes(link);
  const activeStyles = "bg-cognee-purple text-white text-[1rem]";
  const inactiveStyles = "hover:bg-gray-200";

  return (
    <>
      {link !== undefined ? (
        <Link href={link}>
          <Text
            unstyled
            className={`hover:cursor-pointer px-[0.5rem] py-[0.25rem] rounded-[0.5rem] ${isActive || isActiveByPathname ? activeStyles : inactiveStyles}`}
          >
            {children}
          </Text>
        </Link>
      ) : (
        <Text
          unstyled
          className={`px-[0.5rem] py-[0.25rem]`}
          opacity={`${link !== undefined ? "1" : "0.5"}`}
        >
          {children}
        </Text>
      )}
    </>
  );
}
