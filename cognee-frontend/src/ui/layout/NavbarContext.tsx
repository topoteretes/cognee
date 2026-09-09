"use client";

import { createContext, PropsWithChildren, useContext, useEffect, useState } from "react";
import { usePathname } from "next/navigation";

// Persisted as a cookie (not localStorage) so the server can read the
// preference and render the correct sidebar width on first paint — this
// avoids the collapsed sidebar visibly animating shut on every hard
// navigation. One year, readable client-side so the toggle can update it.
export const COLLAPSED_COOKIE_KEY = "cognee.sidebar.collapsed";
const COLLAPSED_COOKIE_MAX_AGE = 365 * 24 * 60 * 60;

interface NavbarContextType {
  isOpen: boolean;
  toggle: () => void;
  close: () => void;
  collapsed: boolean;
  toggleCollapsed: () => void;
}

const NavbarContext = createContext<NavbarContextType>({
  isOpen: false,
  toggle: () => {},
  close: () => {},
  collapsed: false,
  toggleCollapsed: () => {},
});

interface NavbarProviderProps {
  initialCollapsed?: boolean;
}

export function NavbarProvider({
  initialCollapsed = false,
  children,
}: PropsWithChildren<NavbarProviderProps>): React.JSX.Element {
  const [isOpen, setIsOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(initialCollapsed);
  const pathname = usePathname();

  // Close mobile drawer whenever the route changes (user tapped a nav link)
  useEffect(() => {
    setIsOpen(false);
  }, [pathname]);

  const toggleCollapsed = (): void => {
    setCollapsed((prev) => {
      const next = !prev;
      document.cookie = `${COLLAPSED_COOKIE_KEY}=${next}; path=/; max-age=${COLLAPSED_COOKIE_MAX_AGE}; samesite=lax`;
      return next;
    });
  };

  return (
    <NavbarContext.Provider
      value={{
        isOpen,
        toggle: () => setIsOpen((v) => !v),
        close: () => setIsOpen(false),
        collapsed,
        toggleCollapsed,
      }}
    >
      {children}
    </NavbarContext.Provider>
  );
}

export const useNavbar = (): NavbarContextType => useContext(NavbarContext);
