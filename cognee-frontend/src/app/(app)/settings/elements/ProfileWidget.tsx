"use client";

import getUser from "@/modules/users/getUser";
import type CogneeUser from "@/modules/users/CogneeUser";
import { Avatar, Divider, Flex, Stack, Text, TextInput } from "@mantine/core";
import Image from "next/image";
import { useEffect, useState } from "react";
import { tokens } from "@/ui/theme/tokens";

export default function ProfileWidget() {
  const [user, setUser] = useState<CogneeUser | null>(null);

  useEffect(() => {
    getUser().then(setUser);
  }, []);

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
        <h2 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif' }}>Profile</h2>
        <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Your personal account information</p>
      </div>
      <Flex align="center" gap="1rem" mb="1.5rem">
        <Avatar size="lg">
          <Image
            width={54}
            height={54}
            alt=""
            src="/images/icons/avatar.svg"
          />
        </Avatar>
        <Stack gap="0">
          <Text size="sm" fw={700} style={{ color: "#EDECEA" }}>{user?.name ?? "—"}</Text>
          <Text size="xs" style={{ color: "rgba(237,236,234,0.35)" }}>{user?.email ?? "—"}</Text>
        </Stack>
      </Flex>
      <Divider mb="1rem" style={{ borderColor: "rgba(255,255,255,0.08)" }} />
      <Stack gap="0.75rem">
        <TextInput
          label="Name"
          value={user?.name ?? ""}
          disabled
          classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
          radius="0.5rem"
          styles={{
            label: { color: "rgba(237,236,234,0.7)" },
            input: {
              background: "rgba(255,255,255,0.06)",
              borderColor: "rgba(255,255,255,0.12)",
              color: "#EDECEA",
            },
          }}
        />
        <TextInput
          label="Email"
          value={user?.email ?? ""}
          disabled
          classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
          radius="0.5rem"
          styles={{
            label: { color: "rgba(237,236,234,0.7)" },
            input: {
              background: "rgba(255,255,255,0.06)",
              borderColor: "rgba(255,255,255,0.12)",
              color: "#EDECEA",
            },
          }}
        />
      </Stack>
      <Text size="sm" style={{ color: "rgba(237,236,234,0.35)" }} mt="1rem">
        Profile information is managed by your authentication provider.
      </Text>
    </Stack>
  );
}
