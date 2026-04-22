"use client";

import { Flex, Button, Text, Title, Divider, TextInput } from "@mantine/core";
import Link from "next/link";
import { useState } from "react";

interface OAuthLoginButtonsProps {
  type: "login" | "signup";
}

export default function OAuthLoginButtons({ type }: OAuthLoginButtonsProps) {
  const [email, setEmail] = useState("");

  const handleEmailSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;

    const params = new URLSearchParams();
    if (type === "signup") params.set("screen_hint", "signup");
    params.set("connection", "Username-Password-Authentication");
    params.set("login_hint", email);

    window.location.href = `/auth/login?${params.toString()}`;
  };

  const oauthHrefBase = type === "signup" ? "screen_hint=signup&" : "";

  return (
    <Flex className="flex-col gap-[1.5rem] items-center w-full max-w-[22rem]">
      <Flex className="flex-col gap-[0.5rem] items-center">
        <Title order={2} className="!text-[1.75rem] !font-semibold">
          {type === "login" ? "Welcome back" : "Get started"}
        </Title>
        <Text size="sm" className="!text-cognee-muted">
          {type === "login"
            ? "Sign in to your Cognee account"
            : "Create your Cognee account"}
        </Text>
      </Flex>

      <Flex className="flex-col gap-[0.75rem] w-full">
        <a
          href={`/auth/login?${oauthHrefBase}connection=google-oauth2`}
          className="w-full"
        >
          <Button
            h="2.75rem"
            radius="md"
            color="#ffffff"
            className="!text-black !w-full !border !border-solid !border-cognee-border hover:!bg-cognee-bg !transition-colors"
            leftSection={
              <svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
                <path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844a4.14 4.14 0 0 1-1.796 2.716v2.259h2.908c1.702-1.567 2.684-3.875 2.684-6.615Z" fill="#4285F4"/>
                <path d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18Z" fill="#34A853"/>
                <path d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.997 8.997 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332Z" fill="#FBBC05"/>
                <path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 7.29C4.672 5.163 6.656 3.58 9 3.58Z" fill="#EA4335"/>
              </svg>
            }
          >
            <Text size="sm" fw={500}>Continue with Google</Text>
          </Button>
        </a>

        <a
          href={`/auth/login?${oauthHrefBase}connection=github`}
          className="w-full"
        >
          <Button
            h="2.75rem"
            radius="md"
            color="#ffffff"
            className="!text-black !w-full !border !border-solid !border-cognee-border hover:!bg-cognee-bg !transition-colors"
            leftSection={
              <svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
                <path
                  d="M9 0C4.03 0 0 4.03 0 9c0 3.98 2.58 7.35 6.16 8.54.45.08.62-.2.62-.43 0-.21-.01-.78-.01-1.53-2.51.55-3.04-1.21-3.04-1.21-.41-1.04-1-1.32-1-1.32-.82-.56.06-.55.06-.55.9.06 1.38.93 1.38.93.8 1.37 2.1.98 2.62.75.08-.58.31-.98.57-1.2-2-.23-4.1-1-4.1-4.46 0-.99.35-1.79.93-2.42-.09-.23-.4-1.15.09-2.39 0 0 .76-.24 2.49.93a8.68 8.68 0 0 1 4.54 0c1.73-1.17 2.49-.93 2.49-.93.49 1.24.18 2.16.09 2.39.58.63.93 1.44.93 2.42 0 3.47-2.11 4.23-4.12 4.45.32.28.61.83.61 1.67 0 1.21-.01 2.18-.01 2.48 0 .24.16.52.62.43A9.01 9.01 0 0 0 18 9c0-4.97-4.03-9-9-9Z"
                  fill="#24292f"
                />
              </svg>
            }
          >
            <Text size="sm" fw={500}>Continue with GitHub</Text>
          </Button>
        </a>
      </Flex>

      <Divider
        label="or"
        labelPosition="center"
        className="!w-full"
        classNames={{ label: "!text-cognee-muted !text-xs" }}
      />

      <form onSubmit={handleEmailSubmit} className="w-full flex flex-col gap-[0.75rem]">
        <TextInput
          value={email}
          onChange={(e) => setEmail(e.currentTarget.value)}
          type="email"
          placeholder="name@example.com"
          label="Email"
          required
          classNames={{
            input: "!h-[40px]",
            label: "!text-[16px] mb-[8px] !font-normal",
          }}
        />
        <Button
          type="submit"
          h="2.75rem"
          radius="md"
          color="primary2.6"
          className="!w-full"
        >
          <Text size="sm" fw={500}>
            {type === "login" ? "Continue with email" : "Sign up with email"}
          </Text>
        </Button>
      </form>

      <Text size="sm" className="!text-cognee-muted">
        {type === "login" ? (
          <>
            Don&apos;t have an account?{" "}
            <Link href="/sign-up" className="text-[var(--mantine-color-primary2-6)] hover:underline font-medium">
              Sign up
            </Link>
          </>
        ) : (
          <>
            Already have an account?{" "}
            <Link href="/sign-in" className="text-[var(--mantine-color-primary2-6)] hover:underline font-medium">
              Sign in
            </Link>
          </>
        )}
      </Text>
    </Flex>
  );
}
