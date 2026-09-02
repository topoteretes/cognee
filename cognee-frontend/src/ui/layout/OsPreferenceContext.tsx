"use client";

import { createContext, PropsWithChildren, useContext, useState } from "react";

export type PreferredOs = "mac" | "windows";

interface OsPreferenceContextType {
  os: PreferredOs;
  setOs: (os: PreferredOs) => void;
}

const OsPreferenceContext = createContext<OsPreferenceContextType | undefined>(undefined);

export function OsPreferenceProvider({ children }: PropsWithChildren): React.JSX.Element {
  const [os, setOs] = useState<PreferredOs>("mac");

  return (
    <OsPreferenceContext.Provider value={{ os, setOs }}>
      {children}
    </OsPreferenceContext.Provider>
  );
}

export function useOsPreference(): OsPreferenceContextType {
  const context = useContext(OsPreferenceContext);
  if (!context) {
    throw new Error("useOsPreference must be used within an OsPreferenceProvider");
  }
  return context;
}
