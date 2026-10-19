"use client";

import { useState } from "react";
import { Flex, Text, Title, TextInput, PasswordInput, Button } from "@mantine/core";
import AuthCard from "@/ui/elements/Auth/AuthCard";
import { getLocalApiUrl } from "@/modules/users/getLocalApiUrl";

export default function LocalSignInForm() {
  const localApiUrl = getLocalApiUrl();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [register, setRegister] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      if (register) {
        const created = await global.fetch(`${localApiUrl}/api/v1/auth/register`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }), credentials: "include",
        });
        if (!created.ok) {
          const data = await created.json().catch(() => null);
          setError(typeof data?.detail === "string" ? data.detail : "Could not create your account.");
          return;
        }
      }
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
    <AuthCard>
      <Flex className="flex-col gap-[0.75rem] items-center">
        <Title
          order={2}
          className="!text-[2.5rem] !font-light !leading-[1.1] !tracking-[-0.04em] !text-[#EDECEA]"
          style={{ fontFamily: '"TWKLausanne", sans-serif' }}
        >
          Your Cognee server
        </Title>
        <Text size="sm" className="!text-[#EDECEA]/85 !font-light !text-center">
          {register ? "Create your own account to accept a team invitation" : "Sign in with your own account"}
        </Text>
      </Flex>

      {error && (
        <Flex
          className="w-full px-4 py-3 rounded-lg gap-2 items-start"
          style={{ backgroundColor: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.35)" }}
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
          size="md"
          radius="md"
          classNames={{
            label: "!text-[#EDECEA]/85 !font-light",
            input:
              "!bg-white/[0.06] !border-white/15 !text-[#EDECEA] focus:!border-[#BC9BFF] focus:!border-2",
          }}
        />

        <PasswordInput
          label="Password"
          value={password}
          onChange={(e) => setPassword(e.currentTarget.value)}
          required
          autoComplete={register ? "new-password" : "current-password"}
          size="md"
          radius="md"
          classNames={{
            label: "!text-[#EDECEA]/85 !font-light",
            input:
              "!bg-white/[0.06] !border-white/15 !text-[#EDECEA] focus:!border-[#BC9BFF] focus:!border-2",
            innerInput: "!text-[#EDECEA]",
          }}
        />

        <Text size="xs" className="!text-[#EDECEA]/60 !font-light" mt={-4}>
          Team access uses your own account and dataset permissions.
        </Text>

        <Button
          type="submit"
          loading={isLoading}
          fullWidth
          h="2.75rem"
          radius="md"
          mt="xs"
          className="!bg-[#BC9BFF] !text-[#1e1e1c] hover:!bg-[#A87CFF] !transition-colors !border-none"
        >
          <Text size="sm" fw={500}>
            {register ? "Create account and sign in" : "Sign in"}
          </Text>
        </Button>
        <Button variant="subtle" disabled={isLoading} onClick={() => { setRegister(!register); setError(null); }}>
          {register ? "Already have an account? Sign in" : "Create an account"}
        </Button>
      </form>
    </AuthCard>
  );
}
