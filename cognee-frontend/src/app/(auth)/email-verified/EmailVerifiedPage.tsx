"use client";

import AuthCard from "@/ui/elements/Auth/AuthCard";
import { Button, Center, Flex, Text, Title } from "@mantine/core";
import Link from "next/link";


export default function EmailVerifiedPage() {
  return (
    <Center className="flex-1 flex-col">
      <Flex className="flex-col items-center w-full px-6">
        <AuthCard>
          <Flex className="flex-col gap-[1.5rem] items-center w-full max-w-[24rem]">
            <Flex className="flex-col gap-[0.75rem] items-center">
              <Title
                order={2}
                className="!text-[2.5rem] !font-light !leading-[1.1] !tracking-[-0.04em] !text-[#EDECEA]"
                style={{ fontFamily: '"TWKLausanne", sans-serif' }}
              >
                Email verified!
              </Title>
              <Text size="sm" className="!text-[#EDECEA]/85 !font-light !text-center">
                Your account is ready. Sign in to get started.
              </Text>
            </Flex>

            <Link href="/sign-in" className="w-full">
              <Button
                h="3rem"
                className="!w-full !rounded-full !bg-[#BC9BFF] !text-[#1e1e1c] hover:!bg-[#A87CFF] !transition-colors !border-none"
              >
                <Text size="sm" fw={500}>
                  Sign in
                </Text>
              </Button>
            </Link>
          </Flex>
        </AuthCard>
      </Flex>
    </Center>
  );
}
