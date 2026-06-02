"use client";

import { Flex, Text, Button } from "@mantine/core";
import { useRouter } from "next/navigation";
import { tokens } from "@/ui/theme/tokens";

export default function UpgradeBanner() {
  const router = useRouter();

  return (
    <Flex
      align="center"
      justify="space-between"
      className="rounded-[0.5rem] px-[2rem] py-[1rem]"
      style={{
        background: `linear-gradient(135deg, ${tokens.purple}12 0%, #ffffff 50%, ${tokens.purple}08 100%)`,
        border: `1px solid ${tokens.purple}30`,
      }}
    >
      <div>
        <Text fw={600} size="sm" c={tokens.textDark}>
          No active subscription
        </Text>
        <Text size="sm" c={tokens.textSecondary}>
          Subscribe to unlock data uploads, search, and all features.
        </Text>
      </div>
      <Button
        color="primary2.6"
        size="sm"
        onClick={() => router.push("/setup")}
      >
        Upgrade
      </Button>
    </Flex>
  );
}
