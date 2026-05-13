import { Box, Stack, Text } from "@mantine/core";

interface LogSectionProps {
  items: Array<string> | undefined;
}

export default function LogSection({ items }: LogSectionProps) {
  return (
    <Stack className="w-full h-full bg-[#F4F4F4] rounded-[0.5rem] px-[0.75rem] py-[0.625rem] !gap-[0] !min-h-[10.625rem] !max-h-[10.75rem] overflow-y-auto">
      {items &&
        items.map((item) => (
          <Box key={item}>
            <LogItem text={item} />
          </Box>
        ))}
    </Stack>
  );
}

function LogItem({ text }: { text: string }) {
  return (
    <Text size="sm" className="!text-[#ADB5BD]">
      {text}
    </Text>
  );
}
