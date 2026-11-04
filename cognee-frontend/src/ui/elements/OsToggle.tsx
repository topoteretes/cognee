"use client";

import { SegmentedControl } from "@mantine/core";
import { PreferredOs, useOsPreference } from "@/ui/layout/OsPreferenceContext";

// Matches the purple accent + dark glass card styling used throughout the
// connection cards (e.g. AgentConnectionSection.tsx, integrations/page.tsx)
// instead of Mantine's light-theme default (white track, black text).
const ACCENT_PURPLE = "var(--color-cognee-lavender)";
const INACTIVE_LABEL = "rgba(237,236,234,0.65)";
// The selected segment is a filled lavender pill, so it takes the same near-black
// label the app's other lavender-filled controls use (Continue, Connect agent,
// Upload data). White-on-lavender was the odd one out and read washed out.
const ACTIVE_LABEL = "#1e1e1c";

function label(text: string, active: boolean): React.ReactNode {
  return <span style={{ color: active ? ACTIVE_LABEL : INACTIVE_LABEL }}>{text}</span>;
}

export function OsToggle(): React.JSX.Element {
  const { os, setOs } = useOsPreference();

  return (
    <SegmentedControl
      size="xs"
      aria-label="Operating system"
      color={ACCENT_PURPLE}
      // Square, like every other surface in onboarding and the dashboard cards.
      radius={0}
      value={os}
      onChange={(value) => setOs(value as PreferredOs)}
      data={[
        { label: label("Mac", os === "mac"), value: "mac" },
        { label: label("Windows", os === "windows"), value: "windows" },
      ]}
      styles={{
        root: { background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 0 },
        // Mantine rounds the sliding pill and the label independently of the
        // root's radius, so both need squaring explicitly.
        indicator: { borderRadius: 0 },
        label: { borderRadius: 0 },
      }}
    />
  );
}
