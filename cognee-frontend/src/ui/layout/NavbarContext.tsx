"use client";

import { createContext, PropsWithChildren, useContext, useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { getSidebarCollapsed, setSidebarCollapsed } from "@/utils/browserStorage";

interface NavbarContextType {
  /** Mobile drawer open/closed (below the `sm` breakpoint). */
  isOpen: boolean;
  toggle: () => void;
  close: () => void;
  /** Desktop icon-rail collapsed/expanded (`sm` breakpoint and up). */
  isCollapsed: boolean;
  toggleCollapsed: () => void;
}

const NavbarContext = createContext<NavbarContextType>({
  isOpen: false,
  toggle: () => {},
  close: () => {},
  isCollapsed: false,
  toggleCollapsed: () => {},
});

export function NavbarProvider({ children }: PropsWithChildren) {
  const [isOpen, setIsOpen] = useState(false);
  // Desktop collapse preference is read synchronously on the first render via a
  // lazy initializer — not an effect — so a user who collapsed the sidebar
  // doesn't see it flash open on every navigation. localGet is SSR-safe
  // (returns false when localStorage is unavailable).
  const [isCollapsed, setIsCollapsed] = useState(getSidebarCollapsed);
  const pathname = usePathname();

  // Close the mobile drawer whenever the route changes (user tapped a nav
  // link). The desktop collapse state is intentionally left untouched.
  useEffect(() => {
    setIsOpen(false);
  }, [pathname]);

  return (
    <NavbarContext.Provider
      value={{
        isOpen,
        toggle: () => setIsOpen((v) => !v),
        close: () => setIsOpen(false),
        isCollapsed,
        toggleCollapsed: () =>
          setIsCollapsed((collapsed) => {
            const next = !collapsed;
            setSidebarCollapsed(next);
            return next;
          }),
      }}
    >
      {children}
    </NavbarContext.Provider>
  );
}

export const useNavbar = () => useContext(NavbarContext);
