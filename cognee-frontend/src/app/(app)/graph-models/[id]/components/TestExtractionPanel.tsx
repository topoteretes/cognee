"use client";

import { useState } from "react";
import {
  Button,
  Code,
  Divider,
  Flex,
  ScrollArea,
  Stack,
  Text,
  Textarea,
  Title,
} from "@mantine/core";
import { tokens } from "@/ui/theme/tokens";
import type { GraphSchema } from "@/modules/graphModels/types";
import { toCleanSchema } from "@/modules/graphModels/types";

interface TestExtractionPanelProps {
  schema: GraphSchema;
}

export default function TestExtractionPanel({ schema }: TestExtractionPanelProps) {
  const [inputText, setInputText] = useState("");
  const [hasRun, setHasRun] = useState(false);

  const requestPayload = {
    text: inputText || "<your input text>",
    schema: toCleanSchema(schema),
  };

  function handleRun() {
    if (!inputText.trim()) return;
    setHasRun(true);
  }

  return (
    <Stack className="h-full !gap-[0] overflow-hidden" style={{ padding: "1rem" }}>
      <Stack gap="0.25rem" mb="1rem">
        <Title size="h4">Test Extraction</Title>
        <Text size="sm" c={tokens.textMuted}>
          Paste sample text and run extraction against your schema definition.
        </Text>
      </Stack>

      <Textarea
        label="Input text"
        placeholder="Paste a sample text to extract entities and relations from…"
        value={inputText}
        onChange={(e) => {
          setInputText(e.currentTarget.value);
          setHasRun(false);
        }}
        minRows={5}
        maxRows={10}
        autosize
        classNames={{ input: "!border-cognee-border" }}
        radius="0.5rem"
        mb="0.75rem"
      />

      <Flex gap="0.5rem" mb="1rem">
        <Button
          onClick={handleRun}
          disabled={!inputText.trim()}
          style={{ backgroundColor: tokens.purple }}
          radius="0.5rem"
        >
          Run test
        </Button>
        <Text size="xs" c={tokens.textMuted} className="self-center">
          Backend not connected — shows request payload below
        </Text>
      </Flex>

      <Divider mb="1rem" />

      {hasRun ? (
        <ScrollArea className="flex-1">
          <Stack gap="1rem">
            <Stack gap="0.5rem">
              <Text size="sm" fw={600}>
                Extracted Entities
              </Text>
              <Flex
                className="px-[1rem] py-[0.75rem] rounded-[0.5rem]"
                style={{ backgroundColor: tokens.bgHover, border: `1px solid ${tokens.borderLight}` }}
              >
                <Text size="xs" c={tokens.textMuted}>
                  Backend not wired — no extraction results available.
                </Text>
              </Flex>
            </Stack>

            <Stack gap="0.5rem">
              <Text size="sm" fw={600}>
                Extracted Relations
              </Text>
              <Flex
                className="px-[1rem] py-[0.75rem] rounded-[0.5rem]"
                style={{ backgroundColor: tokens.bgHover, border: `1px solid ${tokens.borderLight}` }}
              >
                <Text size="xs" c={tokens.textMuted}>
                  Backend not wired — no extraction results available.
                </Text>
              </Flex>
            </Stack>

            <Stack gap="0.5rem">
              <Text size="sm" fw={600}>
                Suggestions
              </Text>
              <Flex
                className="px-[1rem] py-[0.75rem] rounded-[0.5rem]"
                style={{ backgroundColor: tokens.bgHover, border: `1px solid ${tokens.borderLight}` }}
              >
                <Text size="xs" c={tokens.textMuted}>
                  Backend not wired — no suggestions available.
                </Text>
              </Flex>
            </Stack>
          </Stack>
        </ScrollArea>
      ) : (
        <Stack gap="0.5rem" className="flex-1">
          <Text size="sm" fw={600} c={tokens.textMuted}>
            Request payload that would be sent:
          </Text>
          <ScrollArea className="flex-1">
            <Code
              block
              className="!text-[0.75rem] !leading-relaxed"
              style={{
                backgroundColor: "#1e1e2e",
                color: "#cdd6f4",
                borderRadius: "0.5rem",
                padding: "1rem",
              }}
            >
              {JSON.stringify(requestPayload, null, 2)}
            </Code>
          </ScrollArea>
        </Stack>
      )}
    </Stack>
  );
}
