"use client";

import { startTransition, useActionState } from "react";
import { IconButton } from "@/ui/elements";
import { LoadingIndicator } from "@/ui/app";
import deleteApiKey from "@/modules/apiKeys/deleteAPIKey";
import Image from "next/image";
import { trackEvent } from "@/modules/analytics";

export default function DeleteApiKeyButton({
  isDisabled,
  apiKey,
  onDeleted,
}: {
  isDisabled?: boolean;
  apiKey: { id: string };
  onDeleted?: (id: string) => void;
}) {
  async function handleApiKeyDelete() {
    try {
      await deleteApiKey(apiKey.id);
      trackEvent({ pageName: "API Keys", eventName: "api_key_deleted" });
      onDeleted?.(apiKey.id);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (error: any) {
      return error.message;
    }
  }

  const [errorMessage, deleteApiKeyAction, isLoading] = useActionState(
    handleApiKeyDelete,
    null,
  );

  const handleClick = () => {
    startTransition(() => {
      deleteApiKeyAction();
    });
  };

  return (
    <>
      <div className="text-red-500 mb-4">{errorMessage}</div>
      <IconButton disabled={isDisabled || isLoading} onClick={handleClick}>
        {isLoading ? (
          <LoadingIndicator />
        ) : (
          <Image width={28} height={28} src={"/images/icons/x.svg"} alt={"X"} />
        )}
      </IconButton>
    </>
  );
}
