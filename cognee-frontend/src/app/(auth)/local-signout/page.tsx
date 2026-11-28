"use client";

import { useEffect, useState } from "react";
import { Button, Flex, Text, Title } from "@mantine/core";

import AuthCard from "@/ui/elements/Auth/AuthCard";
import AuthPageContainer from "@/ui/elements/Auth/AuthPageContainer";
import { getLocalApiUrl } from "@/modules/users/getLocalApiUrl";
import { navigateTo } from "@/utils/navigateTo";

/** Ask the backend to end the login. Resolves to whether it did. */
function signOut(): Promise<boolean> {
  return global
    .fetch(`${getLocalApiUrl()}/api/v1/auth/logout`, { method: "POST", credentials: "include" })
    .then((response) => response.ok)
    // No answer at all: the backend is down or cannot be reached.
    .catch(() => false);
}

/**
 * Sign out of the local backend, then go to the login page.
 *
 * The request goes from the browser to the backend: the refresh cookie is sent
 * to the backend's auth routes only, and the backend answers with the deletion
 * of its cookies, which only it knows the names and attributes of.
 *
 * So only the backend can end the login. Unless it answered that it did, the
 * user is still signed in, and the page says so instead of opening the login
 * page as if the sign-out had worked.
 */
export default function LocalSignOutPage() {
  const [attempt, setAttempt] = useState(1);
  const [failedAttempt, setFailedAttempt] = useState(0);

  useEffect(() => {
    signOut().then((signedOut) => (signedOut ? navigateTo("/local-login") : setFailedAttempt(attempt)));
  }, [attempt]);

  // Nothing to show while the first request is on its way.
  if (failedAttempt === 0) return null;

  return (
    <AuthPageContainer>
      <Flex className="flex-1 flex-col items-center justify-center px-6">
        <AuthCard>
          <Title order={2} className="!text-[1.75rem] !font-light !text-[#EDECEA]">
            Sign-out failed
          </Title>
          <Text role="alert" size="sm" className="!text-center" style={{ color: "#FCA5A5" }}>
            You are still signed in: the backend at {getLocalApiUrl()} did not confirm the sign-out.
          </Text>
          <Button
            loading={failedAttempt < attempt}
            onClick={() => setAttempt(attempt + 1)}
            h="3rem"
            className="!w-full !rounded-full !bg-[#BC9BFF] !text-[#1e1e1c] hover:!bg-[#A87CFF] !transition-colors !border-none"
          >
            <Text size="sm" fw={500}>Try again</Text>
          </Button>
        </AuthCard>
      </Flex>
    </AuthPageContainer>
  );
}
