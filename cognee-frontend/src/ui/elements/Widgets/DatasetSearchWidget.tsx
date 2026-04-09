import {
  Box,
  Flex,
  Stack,
  TextInput,
  Title,
  Text,
} from "@mantine/core";
import Image from "next/image";
import { useCallback, useEffect, useRef, useState, KeyboardEvent, ChangeEvent } from "react";
import { SearchResponse } from "@/modules/datasets/searchDataset";
import { useToggle } from "@mantine/hooks";
import SearchSuggestions from "./elements/SearchSuggestions";
import { DataFile } from "@/modules/ingestion/useData";
import { trackEvent } from "@/modules/analytics";
import { tokens } from "@/ui/theme/tokens";

interface DatasetSearchWidgetProps {
  selectedDatasetId: string | null;
  searchDataset: (datasetId: string, query: string) => Promise<unknown>;
  getDatasetData: (datasetId: string) => Promise<DataFile[]>;
  dataVersion?: number;
  onReady?: () => void;
}

export default function DatasetSearchWidget({
  selectedDatasetId,
  searchDataset,
  getDatasetData,
  dataVersion = 0,
  onReady,
}: DatasetSearchWidgetProps) {
  const [searchInputVal, setSearchInputVal] = useState<string>("");
  const [searchResponse, setSearchResponse] = useState<string>("");
  const [displayedResponse, setDisplayedResponse] = useState<string>("");
  const [fullResponse, setFullResponse] = useState<string>("");
  const [answerBoxVisible, setAnswerBoxVisible] = useState(false);
  const [answerBoxFirstRender, setAnswerBoxFirstRender] = useState(true);
  const [loadingSearch, toggleLoadingSearch] = useToggle();
  const [datasetHasData, setDatasetHasData] = useState(false);
  const prevCheckedId = useRef<string | null>(null);
  const prevVersion = useRef(dataVersion);
  const typingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!fullResponse) {
      if (typingRef.current) clearInterval(typingRef.current);
      return;
    }
    let charIndex = 0;
    setDisplayedResponse("");
    typingRef.current = setInterval(() => {
      charIndex++;
      setDisplayedResponse(fullResponse.slice(0, charIndex));
      if (charIndex >= fullResponse.length) {
        if (typingRef.current) clearInterval(typingRef.current);
      }
    }, 5);
    return () => {
      if (typingRef.current) clearInterval(typingRef.current);
    };
  }, [fullResponse]);

  useEffect(() => {
    if (!selectedDatasetId) {
      setDatasetHasData(false);
      setSearchInputVal("");
      setSearchResponse("");
      setFullResponse("");
      setDisplayedResponse("");
      setAnswerBoxVisible(false);
      setAnswerBoxFirstRender(true);
      prevCheckedId.current = null;
      prevVersion.current = dataVersion;
      return;
    }

    const datasetChanged = selectedDatasetId !== prevCheckedId.current;
    const versionChanged = dataVersion !== prevVersion.current;

    if (datasetChanged || versionChanged) {
      prevCheckedId.current = selectedDatasetId;
      prevVersion.current = dataVersion;

      if (datasetChanged) {
        setSearchInputVal("");
        setSearchResponse("");
        setFullResponse("");
        setDisplayedResponse("");
        setAnswerBoxVisible(false);
        setAnswerBoxFirstRender(true);
        setDatasetHasData(false);
      }

      getDatasetData(selectedDatasetId)
        .then((data) => {
          setDatasetHasData(Array.isArray(data) && data.length > 0);
          onReady?.();
        })
        .catch(() => {
          setDatasetHasData(false);
          onReady?.();
        });
    }
  }, [selectedDatasetId, getDatasetData, dataVersion, onReady]);

  const executeSearch = useCallback((query: string) => {
    if (query && query !== "" && selectedDatasetId && !loadingSearch) {
      setFullResponse("");
      setDisplayedResponse("");
      setAnswerBoxVisible(true);
      toggleLoadingSearch();
      searchDataset(selectedDatasetId!, query).then((resp) => {
        const respTest = resp as SearchResponse;
        const hasResults = Boolean(respTest?.[0]?.search_result?.[0]);
        trackEvent({ pageName: "Dashboard", eventName: "search_executed", additionalProperties: { dataset_id: selectedDatasetId!, query_length: String(query.length), has_results: String(hasResults) } });
        if (respTest?.[0]?.search_result?.[0]) {
          setSearchResponse(respTest[0].search_result[0]);
          setFullResponse(respTest[0].search_result[0]);
          setAnswerBoxFirstRender(false);
        } else {
          setSearchResponse("");
          setFullResponse("");
          setDisplayedResponse("");
        }
        toggleLoadingSearch();
      }).catch(() => {
        setSearchResponse("");
        setFullResponse("");
        setDisplayedResponse("");
        toggleLoadingSearch();
      });
    }
  }, [selectedDatasetId, searchDataset, toggleLoadingSearch, loadingSearch]);

  const onSearchSuggestionClick = (val: string) => {
    if (loadingSearch) return;
    setSearchInputVal(val);
    trackEvent({
      pageName: "Dashboard",
      eventName: "search_initiated",
      additionalProperties: { question_type: "suggested", dataset_id: selectedDatasetId ?? "" },
    });
    executeSearch(val);
  };

  const performSearch = () => {
    trackEvent({
      pageName: "Dashboard",
      eventName: "search_initiated",
      additionalProperties: { question_type: "user-defined", dataset_id: selectedDatasetId ?? "" },
    });
    executeSearch(searchInputVal);
  };

  const searchDatasetOnEnterKeyDown = (
    event: KeyboardEvent<HTMLInputElement>,
  ) => {
    if (event.key === "Enter") {
      performSearch();
    }
  };
  const searchDatasetOnClick = () => {
    performSearch();
  };

  return (
    <Stack
      className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0]"
      bg="white"
    >
      <Flex className="justify-between" mb={"1.25rem"}>
        <Stack className="!gap-[0]">
          <Title size="h2" mb="0.125rem">
            Search
          </Title>
          <Text c={tokens.textMuted} size="lg">Ask questions to your cognified data</Text>
        </Stack>
      </Flex>
      <TextInput
        value={searchInputVal}
        disabled={!selectedDatasetId || !datasetHasData || loadingSearch}
        onChange={(e: ChangeEvent<HTMLInputElement>) => {
          setSearchInputVal(e.target.value);
        }}
        onKeyDown={searchDatasetOnEnterKeyDown}
        placeholder={datasetHasData ? "Ask questions about your data" : "No data in this dataset yet"}
        rightSection={
          <Box
            onClick={() => {
              searchDatasetOnClick();
            }}
            className={"hover:cursor-pointer"}
          >
            <Image
              width={"32"}
              height={"32"}
              alt=""
              src={"/images/icons/arrow-right.svg"}
            />
          </Box>
        }
        classNames={{
          input:
            "!text-[1.25rem] !h-[3.75rem] !pl-[1.5rem] !pr-[5rem] !border-cognee-purple",
          section: "mr-[1rem]",
        }}
        radius={"2rem"}
        mb="0.75rem"
        styles={{
          input: {
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          },
        }}
      />
      {!datasetHasData && selectedDatasetId && (
        <Text size="sm" c={tokens.textPlaceholder} mt="0.5rem">
          Upload documents below and cognify to start searching your knowledge graph
        </Text>
      )}
      {datasetHasData && (
        <SearchSuggestions
          onSuggestionClick={onSearchSuggestionClick}
          searchDataset={searchDataset}
          datasetId={selectedDatasetId}
          dataVersion={dataVersion}
          disabled={loadingSearch}
        />
      )}
      {answerBoxVisible && (
        <div
          style={{
            marginTop: "1rem",
            padding: "1.25rem 1.5rem",
            borderRadius: "0.75rem",
            background: "linear-gradient(135deg, #f8f5ff 0%, #ffffff 50%, #f0fff0 100%)",
            border: "1px solid rgba(101, 16, 244, 0.15)",
            boxShadow: "0 4px 20px rgba(101, 16, 244, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04)",
            animation: answerBoxFirstRender ? "answerSlideIn 0.4s ease-out forwards" : undefined,
          }}
        >
          <Text
            size="xs"
            fw={600}
            c={tokens.purple}
            mb="0.5rem"
            style={{ textTransform: "uppercase", letterSpacing: "0.06em" }}
          >
            Answer
          </Text>
          {loadingSearch ? (
            <Flex align="center" gap="0.25rem" py="0.25rem">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  style={{
                    display: "inline-block",
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    backgroundColor: tokens.purple,
                    animation: `dotBounce 1.4s ease-in-out ${i * 0.16}s infinite both`,
                  }}
                />
              ))}
            </Flex>
          ) : (
            <Text
              size="md"
              c="dark.8"
              fw={400}
              style={{ lineHeight: 1.7 }}
            >
              {displayedResponse}
            </Text>
          )}
          <style>{`
            @keyframes answerSlideIn {
              from { opacity: 0; transform: translateY(12px) scale(0.98); }
              to { opacity: 1; transform: translateY(0) scale(1); }
            }
            @keyframes dotBounce {
              0%, 80%, 100% { transform: scale(0.4); opacity: 0.3; }
              40% { transform: scale(1); opacity: 1; }
            }
          `}</style>
        </div>
      )}
    </Stack>
  );
}
