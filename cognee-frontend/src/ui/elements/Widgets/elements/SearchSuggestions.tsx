import { Box, Flex } from "@mantine/core";
import LabelInfoItem from "./LabelInfoItem";
import { useCallback, useEffect, useRef, useState } from "react";
import { SearchResponse } from "@/modules/datasets/searchDataset";
import { tokens } from "@/ui/theme/tokens";
const VISIBLE_COUNT = 5;
const ANIM_DURATION = 300;

const CRAFTING_PHRASES = [
  "Convincing neurons to form interesting questions...",
  "Negotiating with embeddings for the best prompts...",
  "Deploying AGI to craft the perfect questions...",
  "Teaching attention heads where to look...",
  "Letting the latent space dream up questions...",
];

function CraftingText() {
  const [index, setIndex] = useState(() => Math.floor(Math.random() * CRAFTING_PHRASES.length));
  const [fade, setFade] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setFade(false);
      setTimeout(() => {
        setIndex((prev) => {
          let next;
          do { next = Math.floor(Math.random() * CRAFTING_PHRASES.length); } while (next === prev && CRAFTING_PHRASES.length > 1);
          return next;
        });
        setFade(true);
      }, 300);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <span
      style={{
        fontSize: "0.8125rem",
        color: tokens.textSecondary,
        fontWeight: 500,
        opacity: fade ? 1 : 0,
        transition: "opacity 0.3s ease",
      }}
    >
      {CRAFTING_PHRASES[index]}
    </span>
  );
}

interface SuggestionSlot {
  text: string;
  state: "visible" | "fading-out" | "fading-in";
}

interface SearchSuggestionsProps {
  searchDataset: (datasetId: string, query: string) => Promise<unknown>;
  datasetId: string | null;
  onSuggestionClick: (value: string) => void;
  dataVersion?: number;
  disabled?: boolean;
}

export default function SearchSuggestions({
  searchDataset,
  datasetId,
  onSuggestionClick,
  dataVersion = 0,
  disabled = false,
}: SearchSuggestionsProps) {
  const allSuggestionsRef = useRef<string[]>([]);
  const nextIndexRef = useRef(VISIBLE_COUNT);
  const [slots, setSlots] = useState<SuggestionSlot[]>([]);
  const [loading, setLoading] = useState(false);
  const prevDatasetId = useRef<string | null>(null);
  const prevVersion = useRef(dataVersion);

  useEffect(() => {
    const datasetChanged = datasetId !== prevDatasetId.current;
    const versionChanged = dataVersion !== prevVersion.current;
    if (datasetId && (datasetChanged || versionChanged)) {
      prevDatasetId.current = datasetId;
      prevVersion.current = dataVersion;
      setLoading(true);
      setSlots([]);
      searchDataset(
        datasetId,
        "List 10 likely questions a user would ask about this data?",
      ).then((resp) => {
        const respTest = resp as SearchResponse;
        if (respTest?.[0]?.search_result?.[0]) {
          const parsed = respTest[0].search_result[0]
            .split("?")
            .filter((val) => val)
            .map((val) => val.trim().concat("?").slice(3, val.length));
          allSuggestionsRef.current = parsed;
          nextIndexRef.current = VISIBLE_COUNT;
          setSlots(
            parsed.slice(0, VISIBLE_COUNT).map((text) => ({
              text,
              state: "fading-in" as const,
            })),
          );
          setTimeout(() => {
            setSlots((prev) =>
              prev.map((s) => (s.state === "fading-in" ? { ...s, state: "visible" } : s)),
            );
          }, ANIM_DURATION);
        } else {
          allSuggestionsRef.current = [];
          setSlots([]);
        }
        setLoading(false);
      }).catch(() => {
        allSuggestionsRef.current = [];
        setSlots([]);
        setLoading(false);
      });
    } else if (!datasetId) {
      prevDatasetId.current = null;
      prevVersion.current = dataVersion;
      allSuggestionsRef.current = [];
      setSlots([]);
      setLoading(false);
    }
  }, [datasetId, searchDataset, dataVersion]);

  const disabledRef = useRef(disabled);
  disabledRef.current = disabled;

  const handleClick = useCallback(
    (val: string) => {
      if (disabledRef.current) return;
      onSuggestionClick(val);

      setSlots((prev) => {
        const idx = prev.findIndex((s) => s.text === val);
        if (idx === -1) return prev;
        const next = [...prev];
        next[idx] = { ...next[idx], state: "fading-out" };
        return next;
      });

      setTimeout(() => {
        const all = allSuggestionsRef.current;
        setSlots((prev) => {
          const idx = prev.findIndex((s) => s.text === val && s.state === "fading-out");
          if (idx === -1) return prev;

          if (nextIndexRef.current < all.length) {
            const replacement = all[nextIndexRef.current];
            nextIndexRef.current += 1;
            const next = [...prev];
            next[idx] = { text: replacement, state: "fading-in" };
            return next;
          } else {
            return prev.filter((_, i) => i !== idx);
          }
        });

        setTimeout(() => {
          setSlots((p) =>
            p.map((s) => (s.state === "fading-in" ? { ...s, state: "visible" } : s)),
          );
        }, ANIM_DURATION);
      }, ANIM_DURATION);
    },
    [onSuggestionClick],
  );

  return (
    <>
      {loading && slots.length === 0 ? (
        <Flex
          align="center"
          gap="0.5rem"
          mb="0.625rem"
          style={{ minHeight: 32 }}
        >
          <Flex align="center" gap="0.2rem">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                style={{
                  display: "inline-block",
                  width: 5,
                  height: 5,
                  borderRadius: "50%",
                  backgroundColor: tokens.purple,
                  animation: `suggestDotBounce 1.4s ease-in-out ${i * 0.16}s infinite both`,
                }}
              />
            ))}
          </Flex>
          <CraftingText />
        </Flex>
      ) : (
        <Flex
          wrap="wrap"
          gap="0.375rem"
          mb={"0.625rem"}
          style={{
            opacity: disabled ? 0.5 : 1,
            pointerEvents: disabled ? "none" : "auto",
            transition: "opacity 0.2s ease",
          }}
        >
          {slots.map((slot) => (
            <Box
              key={slot.text}
              style={{
                animation:
                  slot.state === "fading-in"
                    ? `suggestIn ${ANIM_DURATION}ms ease-out forwards`
                    : slot.state === "fading-out"
                      ? `suggestOut ${ANIM_DURATION}ms ease-in forwards`
                      : undefined,
              }}
            >
              <LabelInfoItem text={slot.text} onClick={handleClick} />
            </Box>
          ))}
        </Flex>
      )}
      <style>{`
        @keyframes suggestIn {
          from { opacity: 0; transform: translateY(6px) scale(0.95); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes suggestOut {
          from { opacity: 1; transform: translateY(0) scale(1); }
          to { opacity: 0; transform: translateY(-6px) scale(0.95); }
        }
        @keyframes suggestDotBounce {
          0%, 80%, 100% { transform: scale(0.4); opacity: 0.3; }
          40% { transform: scale(1); opacity: 1; }
        }
        @keyframes suggestPulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </>
  );
}
