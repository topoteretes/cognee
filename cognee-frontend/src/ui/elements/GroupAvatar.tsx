"use client";

import type { ReactElement } from "react";
import { WEIGHT_BOLD } from "@/ui/elements/panelTokens";

// The careers-page role-card purple ramp, also used by TeamCard.
const GROUP_COLORS = ["#BC9BFF", "#A380EA", "#916DD9", "#7E58C8"] as const;
const INK = "#1e1e1c";

function initials(name: string): string {
  const words = name.trim().split(/[\s-]+/).filter(Boolean);
  if (words.length === 0) return "?";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[1][0]).toUpperCase();
}

export default function GroupAvatar({ name, size = 36 }: { name: string; size?: number }): ReactElement {
  let hash = 0;
  for (let index = 0; index < name.length; index++) hash = ((hash << 5) - hash + name.charCodeAt(index)) | 0;
  const background = GROUP_COLORS[Math.abs(hash) % GROUP_COLORS.length];

  return (
    <div
      className="flex items-center justify-center flex-shrink-0"
      style={{ width: size, height: size, background, borderRadius: size * 0.28 }}
    >
      <span style={{ color: INK, fontSize: size * 0.36, fontWeight: WEIGHT_BOLD }}>
        {initials(name)}
      </span>
    </div>
  );
}
