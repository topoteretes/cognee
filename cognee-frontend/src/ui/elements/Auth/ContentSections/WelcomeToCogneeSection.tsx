import { Flex, Title, Avatar, Text } from "@mantine/core";
import Image from "next/image";
import ContentSectionWrapper from "./ContentSectionWrapper";
import TitleText from "./TitleText";

export default function WelcomeToCogneeSection() {
  return (
    <ContentSectionWrapper>
      <Flex className="w-[24.5rem] flex-col">
        <TitleText>Welcome to cognee</TitleText>
        <Title size="2rem" className="!mb-[0.875rem]">
          Cognee helped us enrich the data!
        </Title>
        <Text size="md" className="!mb-[2rem]">
          The cognee team built and deployed the entire solution within a month.
          Cognee helped us enrich the data for thousands of our customers and
          provide them with personalized support better suited to their needs.
        </Text>
        <Flex className="gap-[0.75rem] items-center mb-[3.75rem]">
          <Avatar
            size="2.125rem"
            alt="it's me"
            src="/images/avatars/orr-kowarsky-avatar.png"
          />
          <Flex className="flex-col">
            <Text size="md">Orr Kowarsky</Text>
            <Text size="xs">CEO at Dynamo</Text>
          </Flex>
        </Flex>
        <Flex className="gap-[0.3125rem] flex-col">
          <Text size="xs">Used by</Text>
          <Flex className="gap-[1.5rem]">
            <Image
              src={"/images/logos/redis-logo.png"}
              alt={"Redis"}
              width={54}
              height={18}
              objectFit="cover"
              className="object-contain opacity-[0.5]"
            />
            <Image
              src={"/images/logos/bayer-logo.png"}
              alt={"Bayer"}
              width={30}
              height={30}
              objectFit="cover"
              className="object-contain opacity-[0.5]"
            />
            <Image
              src={"/images/logos/atlassian-logo.png"}
              alt={"Atlassian"}
              width={90}
              height={12}
              objectFit="cover"
              className="object-contain opacity-[0.5]"
            />
            <Image
              src={"/images/logos/aws-logo.png"}
              alt={"AWS"}
              width={32}
              height={20}
              objectFit="cover"
              className="object-contain opacity-[0.5]"
            />
          </Flex>
        </Flex>
      </Flex>
    </ContentSectionWrapper>
  );
}
