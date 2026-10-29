"use client";

import { Flex, Stack } from "@mantine/core";
import ProfileWidget from "./elements/ProfileWidget";
import SecurityWidget from "./elements/SecurityWidget";
import { TrackPageView } from "@/modules/analytics";

// Open-source override — the SaaS settings page adds cloud-only widgets;
// local mode has profile and security only. (OrganizationWidget was removed
// from the SaaS repo; do not re-add it here.)
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
