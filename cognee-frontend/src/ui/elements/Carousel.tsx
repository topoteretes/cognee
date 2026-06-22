import { EmblaOptionsType, EmblaPluginType } from "embla-carousel";
import useEmblaCarousel from "embla-carousel-react";
import Fade from "embla-carousel-fade";
import Autoplay from "embla-carousel-autoplay";
import { Box, Flex } from "@mantine/core";

interface CarouselProps {
  slides: React.ReactNode[];
  options?: EmblaOptionsType;
  autoplay?: { useAutoplay: boolean; delay?: number };
  fadeTransition?: boolean;
}

export default function Carousel({
  slides,
  options,
  autoplay,
  fadeTransition = true,
}: CarouselProps) {
  const plugins: Array<EmblaPluginType> = [];
  if (fadeTransition) {
    plugins.push(Fade());
  }
  if (autoplay) {
    plugins.push(Autoplay({ delay: autoplay.delay ?? 10000 }));
  }

  const [emblaRef] = useEmblaCarousel(options, plugins);

  return (
    <Flex className="flex-1 flex-col justify-center xl:items-start xl:pl-[10.3125rem]">
      <Box style={styles.embla}>
        <Box style={styles.emblaViewport} ref={emblaRef}>
          <Box style={styles.emblaContainer} className="items-center">
            {slides.map((content, index) => (
              <Box style={styles.emblaSlide} key={index}>
                {content}
              </Box>
            ))}
          </Box>
        </Box>
      </Box>
    </Flex>
  );
}

const styles = {
  embla: {
    margin: "0",
    ["--slide-size"]: "100%",
  },
  emblaViewport: {
    overflow: "hidden",
  },
  emblaContainer: {
    display: "flex",
    touchAction: "pan-y pinch-zoom",
    marginLeft: "calc(var(--slide-spacing) * -1)",
  },
  emblaSlide: {
    transform: "translate3d(0, 0, 0)",
    flex: "0 0 var(--slide-size)",
    minWidth: 0,
    paddingLeft: "var(--slide-spacing)",
  },
} as const;
