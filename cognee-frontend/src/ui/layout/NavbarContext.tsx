"use client";

import { createContext, PropsWithChildren, useContext, useEffect, useState } from "react";
import { usePathname } from "next/navigation";

interface NavbarContextType {
  isOpen: boolean;
  toggle: () => void;
  close: () => void;
}

const NavbarContext = createContext<NavbarContextType>({
  isOpen: false,
  toggle: () => {},
  close: () => {},
});

export function NavbarProvider({ children }: PropsWithChildren) {
  const [isOpen, setIsOpen] = useState(false);
  const pathname = usePathname();

  // Close sidebar whenever the route changes (user tapped a nav link)
  useEffect(() => {
    setIsOpen(false);
  }, [pathname]);

  return (
    <NavbarContext.Provider
      value={{
        isOpen,
        toggle: () => setIsOpen((v) => !v),
        close: () => setIsOpen(false),
      }}
    >
      {children}
    </NavbarContext.Provider>
  );
}

export const useNavbar = () => useContext(NavbarContext);
