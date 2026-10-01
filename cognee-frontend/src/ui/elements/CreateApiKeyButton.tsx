"use client";

import { startTransition, useActionState } from "react";
import { IconButton } from "@/ui/elements";
import { LoadingIndicator } from "@/ui/app";
import createApiKey from "@/modules/apiKeys/createAPIKey";


export default function CreateApiKeyButton({ isDisabled, children, className }: { isDisabled: boolean; children: React.ReactNode, className: string }) {
  async function handleApiKeyCreate(): Promise<string | null> {
    const result = await createApiKey();
    return result.ok ? null : result.error;
  }

  const [errorMessage, createApiKeyAction, isLoading] = useActionState(handleApiKeyCreate, null);

  const handleClick = () => {
    startTransition(() => {
      createApiKeyAction();
    });
  };

  return (
    <div className={className}>
      <IconButton disabled={isDisabled || isLoading} onClick={handleClick}>
        {children}
        {isLoading && <LoadingIndicator />}
      </IconButton>
      <div className="text-red-500">
        {errorMessage}
      </div>
    </div>
  );
}
