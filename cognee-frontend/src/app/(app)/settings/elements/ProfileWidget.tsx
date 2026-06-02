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
      bg="white"
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: "1.375rem" }}>
        <h2 style={{ fontSize: 20, fontWeight: 300, color: "#18181B", margin: 0, fontFamily: '"TWK Lausanne", system-ui, sans-serif' }}>Profile</h2>
        <p style={{ fontSize: 14, color: "#71717A", margin: 0 }}>Your personal account information</p>
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
          <Text size="sm" fw={600}>{user?.name ?? "—"}</Text>
          <Text size="xs" c={tokens.textMuted}>{user?.email ?? "—"}</Text>
        </Stack>
      </Flex>
      <Divider mb="1rem" />
      <Stack gap="0.75rem">
        <TextInput
          label="Name"
          value={user?.name ?? ""}
          disabled
          classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
          radius="0.5rem"
        />
        <TextInput
          label="Email"
          value={user?.email ?? ""}
          disabled
          classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
          radius="0.5rem"
        />
      </Stack>
      <Text size="sm" c={tokens.textMuted} mt="1rem">
        Profile information is managed by your authentication provider.
      </Text>
    </Stack>
  );
}
