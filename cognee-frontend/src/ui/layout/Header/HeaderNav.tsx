import HeaderAuthNav from "./HeadeAuthNav";
import { Flex } from "@mantine/core";
import HeaderLink from "./HeaderLink";

export default function HeaderNav() {
  return (
    <Flex className="gap-[1rem] flex-wrap items-center">
      <HeaderLink>Products</HeaderLink>
      <HeaderLink>Solutions</HeaderLink>
      <HeaderLink>Community</HeaderLink>
      <HeaderLink>Resource</HeaderLink>
      <HeaderLink>Pricing</HeaderLink>
      <HeaderLink>Contact</HeaderLink>
      <HeaderAuthNav />
    </Flex>
  );
}
