import { AppShellFooter, Flex, Text } from "@mantine/core";

export default function CustomAppShellFooter() {
  return (
    <AppShellFooter>
      <Flex gap="1.75rem">
        <Text size="xs">Local: cognee.db</Text>
        <Text size="xs">Sync: Active</Text>
      </Flex>
    </AppShellFooter>
  );
}
