"use client";

import getUser from "@/modules/users/getUser";
import { useEffect, useState } from "react";
import CogneeUser from "@/modules/users/CogneeUser";
import { Avatar, Flex, Popover, Stack, Text } from "@mantine/core";
import Image from "next/image";
import Link from "next/link";
import { tokens } from "@/ui/theme/tokens";
import trackEvent from "@/modules/analytics/trackEvent";

function SettingsIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function AccessManagementIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  );
}

function BillingIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="1" y="4" width="22" height="16" rx="2" ry="2" />
      <line x1="1" y1="10" x2="23" y2="10" />
    </svg>
  );
}

function DocsIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  );
}

function DiscordIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057c.002.022.015.043.032.054a19.9 19.9 0 0 0 5.993 3.03.077.077 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z" />
    </svg>
  );
}

function HelpIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

const menuItems = [
  { label: "Settings", href: "/settings", external: false, icon: <SettingsIcon />, track: null },
  { label: "Access Management", href: "/access-management", external: false, icon: <AccessManagementIcon />, track: null },
  { label: "Billing", href: "/plan", external: false, icon: <BillingIcon />, track: null },
  { label: "Discord Community", href: "https://discord.gg/m63hxKsp4p", external: true, icon: <DiscordIcon />, track: "https://discord.gg/m63hxKsp4p" },
  { label: "Help", href: "mailto:social@cognee.ai", external: true, icon: <HelpIcon />, track: "mailto:social@cognee.ai" },
];

export default function UserProfile() {
  const [user, setUser] = useState<CogneeUser>();
  const [opened, setOpened] = useState(false);

  useEffect(() => {
    getUser().then((respUser) => {
      setUser(respUser);
    });
  }, []);

  if (!user) {
    return null;
  }

  return (
    <Popover
      opened={opened}
      onChange={setOpened}
      position="top"
      width="target"
      offset={4}
      withArrow={false}
    >
      <Popover.Target>
        <button
          onClick={() => setOpened((o) => !o)}
          className="w-full hover:bg-cognee-hover rounded-[0.25rem] text-left group"
          style={{ background: "none", border: "none", cursor: "pointer", padding: 0 }}
        >
          <Flex className="items-center justify-between p-[0.625rem]">
            <Flex gap={"0.625rem"} wrap={"wrap"}>
              <Avatar>
                <Image
                  width={"38"}
                  height={"38"}
                  alt=""
                  src={"/images/icons/avatar.svg"}
                />
              </Avatar>
              <Stack gap={"0"} className="flex-wrap">
                <Text size="sm" className="!font-semibold">
                  {user.name}
                </Text>
                <Text size="xs" c={tokens.textMuted}>
                  {user.email}
                </Text>
              </Stack>
            </Flex>
            <span
              className="flex items-center justify-center w-[1.375rem] h-[1.375rem] rounded-[0.25rem] group-hover:bg-cognee-border transition-colors"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{
                  color: tokens.textMuted,
                  flexShrink: 0,
                  transform: opened ? "rotate(-90deg)" : "rotate(90deg)",
                  transition: "transform 200ms ease",
                }}
              >
                <polyline points="9 18 15 12 9 6" />
              </svg>
            </span>
          </Flex>
        </button>
      </Popover.Target>
      <Popover.Dropdown p={0}>
        <Stack gap={0}>
          {menuItems.map((item) => (
            <Link
              key={item.label}
              href={item.href}
              {...(item.external
                ? { target: "_blank", rel: "noopener noreferrer" }
                : {})}
              onClick={() => {
                setOpened(false);
                if (item.track) {
                  trackEvent({ pageName: "Navbar", eventName: "click_out", additionalProperties: { target_url: item.track } });
                }
              }}
              className="flex items-center justify-between px-[1rem] py-[0.625rem] hover:bg-cognee-hover text-sm"
              style={{ color: tokens.textBody }}
            >
              <span className="flex items-center gap-[0.25rem]">
                {item.label}
                {item.external && (
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                    <line x1="7" y1="17" x2="17" y2="7" />
                    <polyline points="7 7 17 7 17 17" />
                  </svg>
                )}
              </span>
              <span style={{ color: tokens.textMuted }}>{item.icon}</span>
            </Link>
          ))}
        </Stack>
      </Popover.Dropdown>
    </Popover>
  );
}
