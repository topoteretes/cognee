"use client";
import { Flex } from "@mantine/core";
import CogneeMascotWaveAnimation from "./CogneeMascotWaveAnimation";

export default function AuthContentSectionCarousel() {
  return (
    <Flex w={"calc(100vw * 1/2)"} className={"!bg-white hidden lg:flex"}>
      <CogneeMascotWaveAnimation />
    </Flex>
  );
}
