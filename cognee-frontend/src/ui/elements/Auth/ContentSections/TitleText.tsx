import { Title } from "@mantine/core";
import { PropsWithChildren } from "react";

export default function TitleText({ children }: PropsWithChildren) {
  return (
    <Title size="2rem" className="text-cognee-purple !mb-[3.625rem]">
      {children}
    </Title>
  );
}
