"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Stack, Title, Text, Box } from "@mantine/core";
import { tokens } from "@/ui/theme/tokens";

interface LogEntry {
  text: string;
  delay: number;
}

// Each step has multiple variants — one is picked at random per activation.
// Steps are shown in order; within each step a random variant is chosen.
interface LogStep {
  variants: string[];
  delay: number;
}

const LOG_STEPS: LogStep[] = [
  {
    delay: 1200,
    variants: [
      "Waking up the knowledge elves...",
      "Booting neural pathways...",
      "Warming up the cognification engine...",
      "Stretching our inference muscles...",
    ],
  },
  {
    delay: 2200,
    variants: [
      "Reading your documents. No judgement.",
      "Ingesting your files. We promise not to peek... much.",
      "Scanning documents. Pretending we haven\u2019t seen everything.",
      "Loading your data. It\u2019s heavier than it looks.",
    ],
  },
  {
    delay: 1800,
    variants: [
      "Splitting text into digestible chunks...",
      "Slicing content into bite-sized pieces...",
      "Chopping text into perfectly sized morsels...",
      "Breaking things down so the AI can keep up...",
    ],
  },
  {
    delay: 2800,
    variants: [
      "Feeding chunks to the embedding machine...",
      "Converting words into vectors. Math is beautiful.",
      "Turning prose into high-dimensional poetry...",
      "Embedding everything. Your words are now coordinates.",
    ],
  },
  {
    delay: 1000,
    variants: [
      "Embeddings computed. Your data has opinions now.",
      "Vectors locked in. Every word knows its place.",
      "Embedding complete. The numbers have spoken.",
      "Done embedding. Your text is now very geometric.",
    ],
  },
  {
    delay: 2400,
    variants: [
      "Hunting for entities hiding in your text...",
      "Extracting entities. They can run but they can\u2019t hide.",
      "Finding the who, what, and where in your data...",
      "Entity extraction underway. Names are being named.",
    ],
  },
  {
    delay: 800,
    variants: [
      "Found some. They weren\u2019t hiding very well.",
      "Got them. Resistance was futile.",
      "Entities captured. None escaped.",
      "All accounted for. That was almost too easy.",
    ],
  },
  {
    delay: 2000,
    variants: [
      'Resolving co-references... "he", "she", "it" \u2014 we see you.',
      "Figuring out who \u201Cthey\u201D actually refers to...",
      "Untangling pronouns. It\u2019s like detective work but nerdier.",
      "Connecting the dots between mentions...",
    ],
  },
  {
    delay: 3000,
    variants: [
      "Building knowledge graph. Nodes are making friends.",
      "Constructing the graph. It\u2019s like social networking for concepts.",
      "Assembling knowledge graph. Relationships are forming.",
      "Wiring up the graph. Every idea gets a seat at the table.",
    ],
  },
  {
    delay: 1200,
    variants: [
      "Edges formed. It\u2019s a small world after all.",
      "Connections made. Six degrees of your data.",
      "Links established. Everything is related to everything.",
      "Graph wired up. The nodes are networking.",
    ],
  },
  {
    delay: 2600,
    variants: [
      "Running community detection \u2014 finding who hangs out together",
      "Detecting clusters. Some concepts are clearly best friends.",
      "Finding topic neighborhoods. Birds of a feather...",
      "Grouping related ideas. The cliques are forming.",
    ],
  },
  {
    delay: 2000,
    variants: [
      "Scoring entity importance (sorry, not everyone can be a main character)",
      "Ranking nodes. Popularity contest in progress.",
      "Calculating centrality. Some entities are just more important.",
      "Determining who matters most. It\u2019s not personal, it\u2019s math.",
    ],
  },
  {
    delay: 1800,
    variants: [
      "Storing graph in the database. The database is thrilled.",
      "Persisting everything. The database was getting lonely.",
      "Saving to storage. Your knowledge is now immortal.",
      "Writing to disk. Commitment issues? Not us.",
    ],
  },
  {
    delay: 2400,
    variants: [
      "Consolidating memory layers...",
      "Merging knowledge layers into one unified brain...",
      "Stacking memory layers like a knowledge lasagna...",
      "Fusing everything together. Almost there...",
    ],
  },
];

function pickRandom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

/** Build a concrete sequence by picking one random variant per step. */
function generateSequence(): LogEntry[] {
  return LOG_STEPS.map((step) => ({
    text: pickRandom(step.variants),
    delay: step.delay,
  }));
}

const DONE_LINE = "\u2713 Done. Your knowledge graph is ready to answer questions.";

const TERMINAL_FIXED_HEIGHT = "12rem";
const MAX_VISIBLE_LINES = 8;

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

interface TimestampedLine {
  text: string;
  timestamp: string;
  done?: boolean;
}

interface CognifyActivityPanelProps {
  isActive: boolean;
}

export default function CognifyActivityPanel({ isActive }: CognifyActivityPanelProps) {
  const [visibleLines, setVisibleLines] = useState<TimestampedLine[]>([]);
  const [showPanel, setShowPanel] = useState(false);
  const [progress, setProgress] = useState(0);
  const lineIndexRef = useRef(0);
  const sequenceRef = useRef<LogEntry[]>(generateSequence());
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const progressRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<Date>(new Date());
  const contentRef = useRef<HTMLDivElement>(null);
  const wasActiveRef = useRef(false);
  const collapseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const addLine = useCallback((text: string, done?: boolean) => {
    const now = new Date();
    setVisibleLines((prev) => [
      ...prev,
      { text, timestamp: formatTime(now), done },
    ]);
  }, []);

  const scheduleNextLine = useCallback(() => {
    const seq = sequenceRef.current;
    // When we exhaust the sequence, generate a fresh random one
    if (lineIndexRef.current >= seq.length) {
      sequenceRef.current = generateSequence();
      lineIndexRef.current = 0;
    }
    const entry = sequenceRef.current[lineIndexRef.current];
    timerRef.current = setTimeout(() => {
      addLine(entry.text);
      lineIndexRef.current += 1;
      scheduleNextLine();
    }, entry.delay);
  }, [addLine]);

  // Start progress bar animation
  const startProgress = useCallback(() => {
    setProgress(0);
    // Slowly fill to ~90% over time, then pause
    progressRef.current = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) return prev;
        // Slow down as we approach 90
        const increment = Math.max(0.2, (90 - prev) * 0.02);
        return Math.min(90, prev + increment);
      });
    }, 200);
  }, []);

  const stopProgress = useCallback(() => {
    if (progressRef.current) {
      clearInterval(progressRef.current);
      progressRef.current = null;
    }
  }, []);

  // Handle activation / deactivation
  useEffect(() => {
    if (isActive && !wasActiveRef.current) {
      // Clear any pending collapse
      if (collapseTimerRef.current) {
        clearTimeout(collapseTimerRef.current);
        collapseTimerRef.current = null;
      }
      // Fresh activation — generate a new random sequence
      sequenceRef.current = generateSequence();
      startTimeRef.current = new Date();
      lineIndexRef.current = 0;
      setVisibleLines([]);
      setShowPanel(true);
      setProgress(0);

      // First line immediately
      const first = sequenceRef.current[0];
      addLine(first.text);
      lineIndexRef.current = 1;
      scheduleNextLine();
      startProgress();
    }

    if (!isActive && wasActiveRef.current) {
      // Completion: stop log lines, show done, fill progress, then collapse
      if (timerRef.current) clearTimeout(timerRef.current);
      stopProgress();
      setProgress(100);
      addLine(DONE_LINE, true);

      collapseTimerRef.current = setTimeout(() => {
        setShowPanel(false);
        setVisibleLines([]);
        lineIndexRef.current = 0;
        setProgress(0);
      }, 2000);
    }

    wasActiveRef.current = isActive;

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      stopProgress();
    };
  }, [isActive, addLine, scheduleNextLine, startProgress, stopProgress]);

  const displayLines = visibleLines.slice(-MAX_VISIBLE_LINES);
  const totalLines = visibleLines.length;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateRows: showPanel ? "1fr" : "0fr",
        opacity: showPanel ? 1 : 0,
        transition: "grid-template-rows 0.5s ease, opacity 0.4s ease",
      }}
    >
      <div style={{ overflow: "hidden" }} ref={contentRef}>
        <Stack
          className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0]"
          bg="white"
        >
          <Title size="h2" mb="0.125rem">
            Activity
          </Title>
          <Text c={tokens.textMuted} size="lg" mb="1.5rem">
            processing...
          </Text>

          <Box
            style={{
              background: tokens.textDark,
              borderRadius: "0.5rem",
              overflow: "hidden",
              position: "relative",
            }}
          >
            {/* Progress bar */}
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                height: "2px",
                width: `${progress}%`,
                background: progress >= 100
                  ? tokens.green
                  : `linear-gradient(90deg, ${tokens.purple}, ${tokens.green})`,
                transition: progress >= 100
                  ? "width 0.4s ease-out, background 0.3s"
                  : "width 0.3s linear",
                zIndex: 1,
              }}
            />

            <div
              style={{
                padding: "0.75rem 1rem",
                height: TERMINAL_FIXED_HEIGHT,
                overflow: "hidden",
                fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
                fontSize: "0.8125rem",
                lineHeight: 1.7,
              }}
            >
              {displayLines.map((line, i) => {
                const globalIndex = totalLines - displayLines.length + i;
                const isLast = i === displayLines.length - 1;
                // Dim older lines, newest is brightest
                const age = displayLines.length - 1 - i;
                const dimOpacity = line.done ? 1 : Math.max(0.4, 1 - age * 0.08);

                return (
                  <div
                    key={globalIndex}
                    style={{
                      color: line.done ? tokens.green : tokens.green,
                      opacity: dimOpacity,
                      animation: "lineSlideIn 0.35s ease-out",
                    }}
                  >
                    <span style={{ color: "#5a5a7a", marginRight: "0.75rem" }}>
                      {line.timestamp}
                    </span>
                    {line.done ? (
                      <span style={{ color: tokens.green, fontWeight: 600 }}>{line.text}</span>
                    ) : (
                      <span>{line.text}</span>
                    )}
                    {isLast && !line.done && (
                      <span
                        style={{
                          display: "inline-block",
                          width: "0.5rem",
                          height: "1em",
                          background: tokens.green,
                          marginLeft: "0.25rem",
                          verticalAlign: "text-bottom",
                          animation: "cursorBlink 1s step-end infinite",
                        }}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </Box>

          <style>{`
            @keyframes cursorBlink {
              0%, 100% { opacity: 1; }
              50% { opacity: 0; }
            }
            @keyframes lineSlideIn {
              from {
                opacity: 0;
                transform: translateY(8px);
              }
              to {
                opacity: 1;
                transform: translateY(0);
              }
            }
          `}</style>
        </Stack>
      </div>
    </div>
  );
}
