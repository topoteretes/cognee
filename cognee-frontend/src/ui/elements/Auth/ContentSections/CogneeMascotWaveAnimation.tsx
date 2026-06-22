import { AspectRatio } from "@mantine/core";
import ContentSectionWrapper from "./ContentSectionWrapper";

export default function CogneeMascotWaveAnimation() {
  return (
    <ContentSectionWrapper isCenter>
      <AspectRatio ratio={1080 / 720} mah={374} maw={374}>
        <video autoPlay loop muted playsInline>
          <source src="/videos/cognee-mascot-wave.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      </AspectRatio>
    </ContentSectionWrapper>
  );
}
