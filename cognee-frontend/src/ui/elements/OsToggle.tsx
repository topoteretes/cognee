"use client";

import { SegmentedControl } from "@mantine/core";
import { PreferredOs, useOsPreference } from "@/ui/layout/OsPreferenceContext";

// Matches the purple accent + dark glass card styling used throughout the
// connection cards (e.g. AgentConnectionSection.tsx, integrations/page.tsx)
// instead of Mantine's light-theme default (white track, black text).
const ACCENT_PURPLE = "#6510F4";
const INACTIVE_LABEL = "rgba(237,236,234,0.65)";

function label(text: string, active: boolean): React.ReactNode {
  return <span style={{ color: active ? "#fff" : INACTIVE_LABEL }}>{text}</span>;
}

export function OsToggle(): React.JSX.Element {
  const { os, setOs } = useOsPreference();

  return (
    <SegmentedControl
      size="xs"
      aria-label="Operating system"
      color={ACCENT_PURPLE}
      value={os}
      onChange={(value) => setOs(value as PreferredOs)}
      data={[
        { label: label("Mac", os === "mac"), value: "mac" },
        { label: label("Windows", os === "windows"), value: "windows" },
      ]}
      styles={{ root: { background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)" } }}
    />
  );
}
