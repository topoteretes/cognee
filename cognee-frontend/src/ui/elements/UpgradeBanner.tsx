"use client";

import { Flex, Text, Button } from "@mantine/core";
import { useRouter } from "next/navigation";

export default function UpgradeBanner() {
  const router = useRouter();

  return (
    <Flex
      align="center"
      justify="space-between"
      className="rounded-[0.5rem] px-[2rem] py-[1rem]"
      style={{
        background: "rgba(255,255,255,0.06)",
        backdropFilter: "blur(12px)",
        border: "1px solid rgba(188,155,255,0.35)",
      }}
    >
      <div>
        <Text fw={700} size="sm" c="#EDECEA">
          No active subscription
        </Text>
        <Text size="sm" c="rgba(237,236,234,0.7)">
          Subscribe to unlock data uploads, search, and all features.
        </Text>
      </div>
      <Button
        size="sm"
        styles={{ root: { backgroundColor: "#6510F4", color: "#fff" } }}
        onClick={() => router.push("/setup")}
      >
        Upgrade
      </Button>
    </Flex>
  );
}
