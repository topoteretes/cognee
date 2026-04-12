import { Flex } from "@mantine/core";
import { PropsWithChildren } from "react";

export default function AuthFormContainer({ children }: PropsWithChildren) {
  return (
    <Flex
      mb={"0.625rem"}
      mt={"0.625rem"}
      p={"1.5rem"}
      bg="white"
      className="flex-col w-[24.5rem] rounded-[0.5rem] !gap-[1.5rem]"
    >
      {children}
    </Flex>
  );
}
