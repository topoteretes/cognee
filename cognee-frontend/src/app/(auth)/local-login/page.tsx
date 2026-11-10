import AuthCard from "@/ui/elements/Auth/AuthCard";
import AuthPageContainer from "@/ui/elements/Auth/AuthPageContainer";
import LocalSignInForm from "./partials/LocalSignInForm";
import { Center, Flex } from "@mantine/core";

export default function LocalLoginPage() {
  return (
    <AuthPageContainer>
      <Center className="flex-1 flex-col">
        <Flex className="flex-col items-center w-full px-6">
          <AuthCard>
            <LocalSignInForm />
          </AuthCard>
        </Flex>
      </Center>
    </AuthPageContainer>
  );
}
