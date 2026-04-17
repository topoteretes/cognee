"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import getApiKeys, { type ApiKey } from "@/modules/apiKeys/getApiKeys";
import CopyApiKeyButton from "@/ui/elements/CopyApiKeyButton";
import DeleteApiKeyButton from "@/ui/elements/DeleteApiKeyButton";
import { Box, Center, Flex, Text } from "@mantine/core";
import { LoadingIndicator } from "@/ui/app";
import { tokens } from "@/ui/theme/tokens";

const ROW_EXIT_MS = 220;
const ROW_ENTER_MS = 220;

export default function ApiKeysList({
  refreshTrigger = 0,
  onKeyDeleted,
  onCountChange,
}: {
  refreshTrigger?: number;
  onKeyDeleted?: () => void;
  onCountChange?: (count: number) => void;
} = {}) {
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [removingIds, setRemovingIds] = useState<Set<string>>(new Set());
  const [enteringIds, setEnteringIds] = useState<Set<string>>(new Set());
  const prevKeysRef = useRef<ApiKey[]>([]);
  const isInitialLoad = useRef(true);

  const fetchKeys = useCallback(async (silent = false) => {
    if (!silent) setIsLoading(true);
    try {
      const keys = await getApiKeys();
      if (silent) {
        const prevIds = new Set(prevKeysRef.current.map((k) => k.id));
        const newIds = keys.filter((k) => !prevIds.has(k.id)).map((k) => k.id);
        setApiKeys(keys);
        prevKeysRef.current = keys;
        onCountChange?.(keys.length);
        if (newIds.length > 0) {
          setEnteringIds(new Set(newIds));
          setTimeout(() => setEnteringIds(new Set()), ROW_ENTER_MS);
        }
      } else {
        setApiKeys(keys);
        prevKeysRef.current = keys;
        onCountChange?.(keys.length);
      }
    } catch (err) {
      console.error("Failed to fetch API keys:", err);
    } finally {
      setIsLoading(false);
    }
  }, [onCountChange]);

  useEffect(() => {
    if (isInitialLoad.current) {
      isInitialLoad.current = false;
      fetchKeys(false);
    } else {
      fetchKeys(true);
    }
  }, [fetchKeys, refreshTrigger]);

  const handleDeleted = useCallback(
    (id: string) => {
      setRemovingIds((s) => new Set(s).add(id));
      setTimeout(() => {
        setApiKeys((prev) => prev.filter((k) => k.id !== id));
        setRemovingIds((s) => {
          const next = new Set(s);
          next.delete(id);
          return next;
        });
        fetchKeys(true);
        onKeyDeleted?.();
      }, ROW_EXIT_MS);
    },
    [fetchKeys, onKeyDeleted],
  );

  if (isLoading && apiKeys.length === 0) {
    return (
      <Center className="mt-[1.5rem]">
        <LoadingIndicator />
      </Center>
    );
  }

  if (apiKeys.length === 0 && removingIds.size === 0) {
    return (
      <Center py="2rem">
        <Text c={tokens.textMuted} size="sm">No active API keys</Text>
      </Center>
    );
  }

  return (
    <Box className="flex flex-col w-full api-keys-list">
      {apiKeys.map((apiKey) => (
        <Flex
          key={apiKey.id}
          align="center"
          justify="space-between"
          p="0.625rem"
          className={
            removingIds.has(apiKey.id)
              ? "api-key-row-exit"
              : enteringIds.has(apiKey.id)
                ? "api-key-row-enter"
                : "api-key-row"
          }
          style={{
            backgroundColor: tokens.bgHover,
            borderRadius: "0.5rem",
            border: `1px solid ${tokens.borderLight}`,
          }}
        >
          <Text
            size="sm"
            ff="monospace"
            style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
          >
            {apiKey.name || apiKey.label.slice(0, 15)}
          </Text>
          <Flex gap="0.25rem" style={{ flexShrink: 0 }}>
            <CopyApiKeyButton apiKey={apiKey} />
            <DeleteApiKeyButton
              apiKey={apiKey}
              onDeleted={handleDeleted}
              isDisabled={removingIds.has(apiKey.id)}
            />
          </Flex>
        </Flex>
      ))}
      <style>{`
        .api-keys-list .api-key-row,
        .api-keys-list .api-key-row-enter,
        .api-keys-list .api-key-row-exit {
          margin-bottom: 0.75rem;
        }
        .api-keys-list .api-key-row:last-child,
        .api-keys-list .api-key-row-enter:last-child,
        .api-keys-list .api-key-row-exit:last-child {
          margin-bottom: 0;
        }
        .api-key-row-enter {
          animation: apiKeyEnter ${ROW_ENTER_MS}ms ease-out forwards;
        }
        .api-key-row-exit {
          animation: apiKeyExit ${ROW_EXIT_MS}ms ease-in forwards;
        }
        @keyframes apiKeyEnter {
          from { opacity: 0; transform: translateY(-6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes apiKeyExit {
          from { opacity: 1; height: auto; min-height: auto; padding: 0.625rem; margin-bottom: 0.75rem; border-width: 1px; }
          to { opacity: 0; height: 0; min-height: 0; padding-top: 0; padding-bottom: 0; margin-bottom: 0; overflow: hidden; border-width: 0; }
        }
      `}</style>
    </Box>
  );
}
