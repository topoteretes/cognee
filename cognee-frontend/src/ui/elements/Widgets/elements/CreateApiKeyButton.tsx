"use client";

import { startTransition, useActionState } from "react";
import { LoadingIndicator } from "@/ui/app";
import createApiKey from "@/modules/apiKeys/createAPIKey";
import { Box, Button } from "@mantine/core";
import { trackEvent } from "@/modules/analytics";

export default function CreateApiKeyButton({
  isDisabled,
  children,
  className,
  buttonClassName,
  onCreated,
}: {
  isDisabled: boolean;
  children: React.ReactNode;
  className?: string;
  buttonClassName?: string;
  onCreated?: () => void;
}) {
  async function handleApiKeyCreate() {
    try {
      await createApiKey();
      trackEvent({ pageName: "API Keys", eventName: "api_key_created" });
      onCreated?.();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (error: any) {
      return error.message;
    }
  }

  const [errorMessage, createApiKeyAction, isLoading] = useActionState(
    handleApiKeyCreate,
    null,
  );

  const handleClick = () => {
    startTransition(() => {
      createApiKeyAction();
    });
  };

  return (
    <div className={className}>
      <Button
        disabled={isDisabled || isLoading}
        onClick={handleClick}
        color="primary2.6"
        className={buttonClassName}
      >
        {children}
        {isLoading && (
          <Box ml="0.25rem">
            <LoadingIndicator />
          </Box>
        )}
      </Button>
      <div className="text-red-500">{errorMessage}</div>
    </div>
  );
}
