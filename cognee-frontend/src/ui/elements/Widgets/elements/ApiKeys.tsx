"use server";

import getApiKeys from "@/modules/apiKeys/getApiKeys";
import CopyApiKeyButton from "../../CopyApiKeyButton";
import DeleteApiKeyButton from "../../DeleteApiKeyButton";
import { Box, Flex } from "@mantine/core";

interface ApiKeysProps {
  onApiKeysLoad?: (val: number) => void;
}

export default async function ApiKeys({
  onApiKeysLoad = () => {},
}: ApiKeysProps) {
  const apiKeys = await getApiKeys().then((resp) => {
    onApiKeysLoad(resp.length);
    return resp;
  });

  return (
    <>
      {apiKeys.length === 0 && (
        <Box className="mt-[1rem]">No active API keys</Box>
      )}
      {apiKeys.length > 0 && (
        <Box className="flex flex-col gap-4 mt-8 pb-4 w-full">
          {apiKeys.map((apiKey) => (
            <Flex
              key={apiKey.id}
              className="flex flex-row gap-4 items-center justify-between"
            >
              <Box>{apiKey.name || apiKey.label.slice(0, 15)}</Box>
              <Flex gap="0.25rem">
                <CopyApiKeyButton apiKey={apiKey} />
                <DeleteApiKeyButton apiKey={apiKey} />
              </Flex>
            </Flex>
          ))}
        </Box>
      )}
    </>
  );
}
