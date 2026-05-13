"use client";

import {
  Badge,
  Button,
  Divider,
  Drawer,
  Flex,
  ScrollArea,
  Stack,
  Text,
} from "@mantine/core";
import { tokens } from "@/ui/theme/tokens";
import type { ValidationIssue } from "@/modules/graphModels/validator";

interface ValidationIssuesPanelProps {
  opened: boolean;
  onClose: () => void;
  issues: ValidationIssue[];
  onJumpTo: (entityId?: string, fieldId?: string) => void;
}

export default function ValidationIssuesPanel({
  opened,
  onClose,
  issues,
  onJumpTo,
}: ValidationIssuesPanelProps) {
  const errors = issues.filter((i) => i.severity === "error");
  const warnings = issues.filter((i) => i.severity === "warn");

  return (
    <Drawer
      opened={opened}
      onClose={onClose}
      title={
        <Flex align="center" gap="0.5rem">
          <Text fw={600}>Validation Issues</Text>
          {errors.length > 0 && (
            <Badge color="red" variant="filled" size="sm">
              {errors.length} error{errors.length > 1 ? "s" : ""}
            </Badge>
          )}
          {warnings.length > 0 && (
            <Badge color="yellow" variant="filled" size="sm">
              {warnings.length} warning{warnings.length > 1 ? "s" : ""}
            </Badge>
          )}
        </Flex>
      }
      position="right"
      size="md"
      padding="1.5rem"
    >
      {issues.length === 0 ? (
        <Flex direction="column" align="center" justify="center" className="mt-[3rem]">
          <Text size="xl" mb="0.5rem">
            ✓
          </Text>
          <Text size="sm" c={tokens.textMuted}>
            No issues found — schema looks valid!
          </Text>
        </Flex>
      ) : (
        <ScrollArea>
          <Stack gap="0.75rem">
            {errors.length > 0 && (
              <>
                <Text size="xs" fw={700} c={tokens.textMuted} tt="uppercase">
                  Errors ({errors.length})
                </Text>
                {errors.map((issue, i) => (
                  <IssueCard key={i} issue={issue} onJump={onJumpTo} />
                ))}
              </>
            )}

            {errors.length > 0 && warnings.length > 0 && <Divider />}

            {warnings.length > 0 && (
              <>
                <Text size="xs" fw={700} c={tokens.textMuted} tt="uppercase">
                  Warnings ({warnings.length})
                </Text>
                {warnings.map((issue, i) => (
                  <IssueCard key={i} issue={issue} onJump={onJumpTo} />
                ))}
              </>
            )}
          </Stack>
        </ScrollArea>
      )}
    </Drawer>
  );
}

function IssueCard({
  issue,
  onJump,
}: {
  issue: ValidationIssue;
  onJump: (entityId?: string, fieldId?: string) => void;
}) {
  return (
    <Flex
      gap="0.75rem"
      className="px-[0.875rem] py-[0.75rem] rounded-[0.5rem]"
      style={{
        backgroundColor: issue.severity === "error" ? "#fff5f5" : "#fffbeb",
        border: `1px solid ${issue.severity === "error" ? "#fecaca" : "#fde68a"}`,
      }}
    >
      <Text size="sm" style={{ flexShrink: 0 }}>
        {issue.severity === "error" ? "✕" : "⚠"}
      </Text>
      <Stack gap="0.25rem" className="flex-1 min-w-0">
        <Text size="xs" c={tokens.textMuted} className="font-mono">
          {issue.path}
        </Text>
        <Text size="sm">{issue.message}</Text>
        {(issue.entityId || issue.fieldId) && (
          <Button
            size="xs"
            variant="subtle"
            color={issue.severity === "error" ? "red" : "yellow"}
            onClick={() => onJump(issue.entityId, issue.fieldId)}
            className="self-start"
            style={{ padding: "0 0.25rem", height: "1.5rem" }}
          >
            Jump to →
          </Button>
        )}
      </Stack>
    </Flex>
  );
}
