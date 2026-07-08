"use client";

import { Button, Divider, Stack, Text } from "@mantine/core";
import { trackPageEvent } from "@/modules/analytics";
import isCloudEnvironment from "@/utils/isCloudEnvironment";

export default function SecurityWidget() {
  const logoutHref = isCloudEnvironment() ? "/api/signout" : "/api/local-signout";

  return (
    <Stack
      className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0] min-w-[25rem] max-w-[29.5rem]"
      style={{
        background: "rgba(255,255,255,0.06)",
        backdropFilter: "blur(12px)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 12,
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: "1.375rem" }}>
        <h2 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif' }}>Security</h2>
        <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Manage your account security and session</p>
      </div>
      <Stack gap="1.25rem">
        <Stack gap="0.25rem">
          <Text size="sm" fw={500} style={{ color: "#EDECEA" }}>
            Password
          </Text>
          <Text size="sm" style={{ color: "rgba(237,236,234,0.55)" }}>
            Your password is managed by your authentication provider.
          </Text>
        </Stack>
        <Divider style={{ borderColor: "rgba(255,255,255,0.08)" }} />
        <Stack gap="0.25rem">
          <Text size="sm" fw={500} style={{ color: "#EDECEA" }}>
            Session
          </Text>
          <Text size="sm" style={{ color: "rgba(237,236,234,0.55)" }} mb="0.5rem">
            Sign out of your account on this device.
          </Text>
          <Button
            component="a"
            href={logoutHref}
            variant="outline"
            color="gray"
            size="sm"
            radius="0.5rem"
            className="self-start"
            styles={{
              root: {
                borderColor: "rgba(255,255,255,0.15)",
                color: "rgba(237,236,234,0.8)",
              },
            }}
            onClick={() => trackPageEvent({ pageName: "Settings", eventName: "sign_out" })}
          >
            Log out
          </Button>
        </Stack>
      </Stack>
    </Stack>
  );
}
