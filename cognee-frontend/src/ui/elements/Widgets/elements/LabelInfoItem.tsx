import { Box, Text } from "@mantine/core";

interface LabelInfoItemProps {
  text: string;
  onClick?: (val: string) => void;
}

export default function LabelInfoItem({ text, onClick }: LabelInfoItemProps) {
  return (
    <Box
      bg="#F4F4F4"
      bdrs={"0.5rem"}
      pl="1.75rem"
      pr="1.25rem"
      py={"0.25rem"}
      className="overflow-auto whitespace-nowrap hover:cursor-pointer"
      onClick={
        onClick !== undefined
          ? () => {
              onClick(text);
            }
          : function () {}
      }
    >
      <Text size="xs" className="!font-semibold">
        {text}
      </Text>
    </Box>
  );
}
