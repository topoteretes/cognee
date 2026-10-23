"use client";

import { useState } from "react";
import { Flex, Text, Title, TextInput, PasswordInput, Button } from "@mantine/core";

const localApiUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL || "http://localhost:8000";

const DEFAULT_EMAIL = "default_user@example.com";
const DEFAULT_PASSWORD = "default_password";

const INPUT_CLASSNAMES = {
  input: "!h-[44px] !bg-white/[0.06] !border-white/15 !rounded-xl !text-[#EDECEA] placeholder:!text-white/35 focus:!border-[#BC9BFF] !transition-colors",
  innerInput: "!text-[#EDECEA] placeholder:!text-white/35",
  label: "!text-[15px] mb-[8px] !font-light !text-[#EDECEA]",
  visibilityToggle: "!text-white/45 hover:!text-white/70",
};

export default function LocalSignInForm() {
  const [email, setEmail] = useState(DEFAULT_EMAIL);
  const [password, setPassword] = useState(DEFAULT_PASSWORD);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      const formData = new URLSearchParams();
      formData.append("username", email);
      formData.append("password", password);

      const response = await global.fetch(`${localApiUrl}/api/v1/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: formData.toString(),
        credentials: "include",
      });

      if (!response.ok) {
        const data = await response.json().catch((err) => {
          console.warn("Failed to parse login error response:", err);
          return null;
        });
        const detail = data?.detail;
        if (detail === "LOGIN_BAD_CREDENTIALS") {
          setError("Invalid email or password.");
        } else if (detail === "LOGIN_USER_NOT_VERIFIED") {
          setError("Please verify your email before signing in.");
        } else {
          setError(typeof detail === "string" ? detail : "Login failed. Please try again.");
        }
        return;
      }

      window.location.href = "/";
    } catch (err) {
      if (err instanceof TypeError) {
        setError(
          "Cannot connect to local backend at " + localApiUrl + ". Is it running?"
        );
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <Flex className="flex-col gap-[1.5rem] items-center w-full max-w-[24rem]">
      <Flex className="flex-col gap-[0.75rem] items-center">
        <Title
          order={2}
          className="!text-[2.5rem] !font-light !leading-[1.1] !tracking-[-0.04em] !text-[#EDECEA]"
          style={{ fontFamily: '"TWKLausanne", sans-serif' }}
        >
          Welcome back
        </Title>
        <Text size="sm" className="!text-[#EDECEA]/85 !font-light !text-center">
          Sign in to your local Cognee backend
        </Text>
      </Flex>

      {error && (
        <Flex
          role="alert"
          className="w-full px-4 py-3 rounded-xl gap-2 items-start"
          style={{ backgroundColor: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.35)" }}
        >
          <Text size="sm" style={{ color: "#FCA5A5" }}>
            {error}
          </Text>
        </Flex>
      )}

      <form onSubmit={handleSubmit} className="w-full flex flex-col gap-[0.75rem]">
        <TextInput
          label="Email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.currentTarget.value)}
          required
          autoComplete="email"
          classNames={INPUT_CLASSNAMES}
        />

        <PasswordInput
          label="Password"
          value={password}
          onChange={(e) => setPassword(e.currentTarget.value)}
          required
          autoComplete="current-password"
          classNames={INPUT_CLASSNAMES}
        />

        <Text size="xs" className="!text-white/55 !font-light" mt={-4}>
          Default credentials are pre-filled for local development
        </Text>

        <Button
          type="submit"
          loading={isLoading}
          disabled={isLoading}
          h="3rem"
          className="!w-full !rounded-full !bg-[#BC9BFF] !text-[#1e1e1c] hover:!bg-[#A87CFF] !transition-colors !border-none"
          mt="xs"
        >
          <Text size="sm" fw={500}>Sign in</Text>
        </Button>
      </form>
    </Flex>
  );
}
