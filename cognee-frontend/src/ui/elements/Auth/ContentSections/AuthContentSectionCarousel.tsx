"use client";
import { EmblaOptionsType } from "embla-carousel";
import { ReactElement } from "react";
import Carousel from "../../Carousel";
import WelcomeToCogneeSection from "./WelcomeToCogneeSection";
import TrustedByDevelopersSection from "./TrustedByDevelopersSection";
import { Flex } from "@mantine/core";
import CogneeMascotWaveAnimation from "./CogneeMascotWaveAnimation";

export default function AuthContentSectionCarousel() {
  const carouselOptions: EmblaOptionsType = {
    loop: true,
  };
  const carouselSlides = new Array<ReactElement>();
  carouselSlides.push(
    <WelcomeToCogneeSection />,
    <CogneeMascotWaveAnimation />,
    <TrustedByDevelopersSection />,
  );

  return (
    <Flex w={"calc(100vw * 1/2)"} className={"!bg-white hidden lg:flex"}>
      <Carousel
        slides={carouselSlides}
        options={carouselOptions}
        autoplay={{ useAutoplay: true, delay: 10000 }}
      />
    </Flex>
  );
}
