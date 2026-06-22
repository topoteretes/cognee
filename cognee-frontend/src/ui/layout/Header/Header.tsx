"use client";

import { AppShellHeader, ColorSchemeScript } from "@mantine/core";
import { usePathname } from "next/navigation";
import HeaderNav from "./HeaderNav";
import HeaderLogo from "./HeaderLogo";

const AUTH_PATHS = ["/sign-in", "/sign-up", "/reset-password", "/forgot-password", "/account", "/plan"];

export default function CustomAppShellHeader() {
  const pathname = usePathname();

  if (!AUTH_PATHS.includes(pathname)) {
    return null;
  }

  return (
    <AppShellHeader>
      <header className="flex flex-row h-14 min-h-14 px-5 items-center justify-between w-full mx-auto gap-[1rem]">
        <HeaderLogo />
        <HeaderNav />
        <ColorSchemeScript />
      </header>
    </AppShellHeader>
  );
}
