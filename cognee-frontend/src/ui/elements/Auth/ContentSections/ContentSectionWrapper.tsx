import { Flex } from "@mantine/core";
import { PropsWithChildren } from "react";

interface ContentSectionWrapperProps extends PropsWithChildren {
  isCenter?: boolean;
}

export default function ContentSectionWrapper({
  isCenter,
  children,
}: ContentSectionWrapperProps) {
  const centerContentStyles =
    "content-center ml-[-164px] align-center items-center";

  return (
    <Flex
      className={`flex-1 justify-center ${isCenter ? centerContentStyles : "xl:justify-start"}`}
    >
      {children}
    </Flex>
  );
}
