"use client";

import type { ReactElement } from "react";
import { ACCESS_LEVELS, type AccessLevel } from "@/modules/permissions/accessLevels";
import { selectStyle } from "@/ui/elements/panelTokens";

interface AccessLevelSelectProps {
  value: AccessLevel | null;
  disabled: boolean;
  label: string;
  onChange: (level: AccessLevel) => void;
}

function isAccessLevel(value: string): value is AccessLevel {
  return ACCESS_LEVELS.some((option) => option.value === value);
}

/** The level a group holds on one dataset: the same for every member. */
export default function AccessLevelSelect({ value, disabled, label, onChange }: AccessLevelSelectProps): ReactElement {
  return (
    <select
      aria-label={label}
      value={value ?? ""}
      disabled={disabled}
      style={{ ...selectStyle, width: 132, cursor: disabled ? "default" : "pointer" }}
      onChange={(event) => { if (isAccessLevel(event.target.value)) onChange(event.target.value); }}
    >
      {value === null && <option value="" disabled>Custom</option>}
      {ACCESS_LEVELS.map((option) => (
        <option key={option.value} value={option.value}>{option.label}</option>
      ))}
    </select>
  );
}
