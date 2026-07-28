"use client";

import { Stack, Text } from "@mantine/core";

export default function OrganizationWidget() {
  return (
    <Stack
      className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0] flex-1"
      style={{
        background: "rgba(255,255,255,0.06)",
        backdropFilter: "blur(12px)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 12,
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: "1.375rem" }}>
        <h2 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif' }}>
          Organization
        </h2>
        <Text size="sm" style={{ color: "rgba(237,236,234,0.55)" }}>
          Team and organization management is not available in the open-source UI.
        </Text>
      </div>
    </Stack>
  );
}
