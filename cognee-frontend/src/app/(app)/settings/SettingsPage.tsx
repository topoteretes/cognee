"use client";

import { Flex, Stack } from "@mantine/core";
import ProfileWidget from "./elements/ProfileWidget";
import SecurityWidget from "./elements/SecurityWidget";
import { TrackPageView } from "@/modules/analytics";

export default function SettingsPage() {
  return (
    <Stack className="!gap-[0.625rem] h-full">
      <TrackPageView page="Settings" />
      <Flex gap={"0.625rem"} className="flex-col xl:flex-row">
        <ProfileWidget />
        <SecurityWidget />
      </Flex>
    </Stack>
  );
}
