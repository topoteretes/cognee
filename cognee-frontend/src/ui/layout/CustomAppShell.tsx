"use client";

import { PropsWithChildren } from "react";
import { usePathname } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import TopBar from "./TopBar";
import CustomAppShellNavbar from "./Navbar/CustomAppShellNavbar";

const SHELL_HIDDEN_PATHS = [
  "/account",
  "/plan",
  "/setup",
  "/sign-in",
  "/sign-up",
  "/reset-password",
  "/forgot-password",
];

export default function CustomAppShell({ children }: PropsWithChildren) {
  const pathname = usePathname();
  const hideShell = SHELL_HIDDEN_PATHS.includes(pathname);
  const { statusMessage, isInitializing } = useCogniInstance();

  if (hideShell) {
    return <>{children}</>;
  }

  return (
    <div className="flex flex-col h-screen">
      <TopBar />
      <div className="flex flex-1 overflow-hidden">
        <CustomAppShellNavbar />
        <main className="flex-1 overflow-auto" style={{ background: "#FAFAF9" }}>
          {isInitializing && statusMessage ? (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", background: "#FFFFFF" }}>
              <video src="/videos/mascot-waiting.mp4" autoPlay loop muted playsInline style={{ width: 200, height: "auto" }} />
            </div>
          ) : children}
        </main>
      </div>
    </div>
  );
}
