import { MantineThemeOverride } from "@mantine/core";

const customColors: MantineThemeOverride = {
  colors: {
    // Zinc scale — primary neutral palette
    primary1: [
      "#FAFAFA",   // zinc-50
      "#F4F4F5",   // zinc-100
      "#E4E4E7",   // zinc-200
      "#D4D4D8",   // zinc-300
      "#A1A1AA",   // zinc-400
      "#71717A",   // zinc-500
      "#52525B",   // zinc-600
      "#3F3F46",   // zinc-700
      "#27272A",   // zinc-800
      "#18181B",   // zinc-900
    ],
    // Purple scale — brand accent
    primary2: [
      "#F0EDFF",   // selected bg
      "#E8E2FD",   // selected+hover bg
      "#E4DEFF",   // pressed help bg
      "#BEA0FC",
      "#9A6BF8",
      "#7C3FF5",
      "#6510F4",   // brand purple
      "#5A0ED6",   // hover
      "#4A0BAF",   // pressed
      "#3A0090",
    ],
    // Green scale — accent / signal
    primary3: [
      "#e6ffe5",
      "#cfffcd",
      "#a0ff9b",
      "#6cff64",
      "#42ff37",
      "#28ff1c",
      "#0dff00",
      "#00e300",
      "#00ca00",
      "#00ae00",
    ],
    // Gray scale — secondary neutral (for borders, muted elements)
    secondary1: [
      "#F9FAFB",
      "#F3F4F6",
      "#E5E7EB",
      "#D1D5DB",
      "#9CA3AF",
      "#6B7280",
      "#4B5563",
      "#374151",
      "#1F2937",
      "#111827",
    ],
  },
};

export default customColors;
