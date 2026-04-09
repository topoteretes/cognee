import { Box, Container } from "@mantine/core";
import { PropsWithChildren } from "react";

export default function AuthPageContainer({ children }: PropsWithChildren) {
  return (
    <Box className="h-screen overflow-hidden" m="-10px">
      <Container
        fluid
        className="h-screen w-full max-w-none !p-0 flex flex-row"
      >
        {children}
      </Container>
    </Box>
  );
}
