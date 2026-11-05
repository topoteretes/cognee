"use client";

import { Button, Divider, Stack, Text, Title } from "@mantine/core";
import { tokens } from "@/ui/theme/tokens";
import { trackPageEvent } from "@/modules/analytics";

export default function SecurityWidget() {
  return (
    <Stack
      className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0] flex-1"
      bg="white"
    >
      <Stack className="!gap-[0] mb-[1.375rem]">
        <Title size="h2" mb="0.125rem">
          Security
        </Title>
        <Text c={tokens.textMuted} size="lg">
          Manage your account security and session
        </Text>
      </Stack>
      <Stack gap="1.25rem">
        <Stack gap="0.25rem">
          <Text size="sm" fw={500}>
            Password
          </Text>
          <Text size="sm" c={tokens.textMuted}>
            Your password is managed by your authentication provider.
          </Text>
        </Stack>
        <Divider />
        <Stack gap="0.25rem">
          <Text size="sm" fw={500}>
            Session
          </Text>
          <Text size="sm" c={tokens.textMuted} mb="0.5rem">
            Sign out of your account on this device.
          </Text>
          <Button
            component="a"
            href="/api/signout"
            variant="outline"
            color="gray"
            size="sm"
            radius="0.5rem"
            className="self-start"
            onClick={() => trackPageEvent({ pageName: "Settings", eventName: "sign_out" })}
          >
            Log out
          </Button>
        </Stack>
      </Stack>
    </Stack>
  );
}
