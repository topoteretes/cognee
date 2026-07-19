import AuthContentSectionCarousel from "@/ui/elements/Auth/ContentSections/AuthContentSectionCarousel";
import AuthPageContainer from "@/ui/elements/Auth/AuthPageContainer";
import LocalSignInForm from "./partials/LocalSignInForm";
import { Center, Flex } from "@mantine/core";

export default function LocalLoginPage() {
  return (
    <AuthPageContainer>
      <Center className="flex-1 flex-col" bg={"primary1.0"}>
        <Flex className="flex-col items-center w-full px-6 lg:w-[50vw] lg:px-0">
          <LocalSignInForm />
        </Flex>
      </Center>
      <AuthContentSectionCarousel />
    </AuthPageContainer>
  );
}
