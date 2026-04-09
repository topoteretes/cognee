import { Stack, Title, Text, Flex, Center } from "@mantine/core";
import ApiKeys from "./elements/ApiKeys";
import { Suspense } from "react";
import { LoadingIndicator } from "@/ui/app";
import CreateApiKeyButton from "./elements/CreateApiKeyButton";

export default function ManageApiKeysWidget() {
  // Use useRef to store a value that doesn't trigger a re-render when it changes
  //   const numberOfApiKeys = useRef(0);
  //   const onApiKeysLoad = (val: number) => {
  //     numberOfApiKeys.current = val;
  //   };

  return (
    <Stack
      className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0] min-w-[20rem] max-w-[29.5rem] max-h-[49.625rem] min-h-[40rem] !justify-between"
      bg="white"
    >
      <Stack unstyled>
        <Title size="h2" mb="0.125rem">
          Manage API keys
        </Title>
        <Flex mb="1rem">
          <Suspense
            fallback={
              <Center className="mt-[1.5rem]">
                <LoadingIndicator />
              </Center>
            }
          >
            <ApiKeys />
          </Suspense>
        </Flex>
      </Stack>
      <CreateApiKeyButton isDisabled={false} buttonClassName="!w-full">
        <Text size="md">Add API key</Text>
      </CreateApiKeyButton>
    </Stack>
  );
}
