import { CogneeIcon } from "@/ui/icons";
import { Flex, Title } from "@mantine/core";
import Link from "next/link";

export default function HeaderLogo() {
  return (
    <Link href={"/"}>
      <Flex className="gap-[0.375rem] items-center justify-center">
        <CogneeIcon />
        <Title size="h5" c="primary2.6">
          Cognee
        </Title>
      </Flex>
    </Link>
  );
}
