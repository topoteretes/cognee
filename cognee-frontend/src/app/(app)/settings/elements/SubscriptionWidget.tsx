"use client";

import { Stack, Text, Title } from "@mantine/core";
import { tokens } from "@/ui/theme/tokens";

export default function SubscriptionWidget() {
  return (
    <Stack
      className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0] flex-1"
      bg="white"
    >
      <Stack className="!gap-[0] mb-[1.375rem]">
        <Title size="h2" mb="0.125rem">
          Subscription
        </Title>
        <Text c={tokens.textMuted} size="lg">
          Managed cloud billing is not available in the open-source UI.
        </Text>
      </Stack>
    </Stack>
  );
}
