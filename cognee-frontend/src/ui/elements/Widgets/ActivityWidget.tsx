import { Stack, Title, Text, Flex, Button } from "@mantine/core";
import LabelInfoItem from "./elements/LabelInfoItem";
import LogSection from "./elements/LogSection";
import { tokens } from "@/ui/theme/tokens";

export default function ActivityWidget() {
  return (
    <Stack
      className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0]"
      bg="white"
    >
      <Title size="h2" mb="0.125rem">
        Activity
      </Title>
      <Text c={tokens.textMuted} size="lg" mb="1rem">
        processing...
      </Text>
      <Flex wrap="wrap" gap="0.375rem" mb={"1.5rem"}>
        <LabelInfoItem text="12 Documents" />
        <LabelInfoItem text="1,847 Entities" />
        <LabelInfoItem text="3,293 Relations" />
        <LabelInfoItem text="4,562 Embeddings" />
      </Flex>
      <Flex mb={"1.5rem"}>
        <LogSection
          items={[
            "14:32:36 Ready for queries",
            "14:32:35 Memory consolidation complete",
            "14:32:28 Running memify algorithms...",
            "14:32:25 Graph stored in Neo4j",
            "14:32:22 Created 412 relationships",
            "14:32:18 Extracted 234 entities",
            "14:32:15 Building knowledge graph...",
          ]}
        />
      </Flex>
      <Button color="primary2.6">
        <Text size="md">Visualize</Text>
      </Button>
    </Stack>
  );
}
