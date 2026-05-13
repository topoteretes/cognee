import { Box, Flex, Text } from "@mantine/core";
import ContentSectionWrapper from "./ContentSectionWrapper";
import TitleText from "./TitleText";
import Image from "next/image";

export default function BuildSmarterAgentsYourWaySection() {
  return (
    <ContentSectionWrapper>
      <Box className="flex flex-row 2xl:flex-row ml-[1rem]">
        <Flex className="flex-col md:mb-[5rem]">
          <TitleText>Build smarter agents your way.</TitleText>
          <Flex className="flex-col gap-[1rem]">
            <Text>Custom Ontologies & Reasoners for Domain-awareness</Text>
            <Text>Memory Layers for Agent-Scoped Context</Text>
            <Text>CodeGraph Context for Coding Copilots</Text>
          </Flex>
        </Flex>
        <Image
          src={
            "/images/backgrounds/build-smarter-agents-your-way-background.svg"
          }
          alt={"Atlassian"}
          width={404}
          height={400}
          objectFit="cover"
          className="object-contain"
        />
      </Box>
    </ContentSectionWrapper>
  );
}
