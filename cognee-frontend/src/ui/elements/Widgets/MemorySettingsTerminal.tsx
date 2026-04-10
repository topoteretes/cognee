"use client";

import { useCallback, useRef, useState, useEffect, KeyboardEvent } from "react";
import { Stack, Title, Text, Box, Button } from "@mantine/core";
import {
  graphModelPresets,
  promptPresets,
  llmPresets,
  DEFAULT_GRAPH_PROMPT,
} from "./elements/memorySettingsPresets";
import { tokens } from "@/ui/theme/tokens";
import { syncMemorySettings, loadMemorySettingsFromBackend } from "@/modules/configuration/userConfiguration";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";

export interface MemorySettings {
  graphModel: object | null;
  customPrompt: string | null;
  llmModel: string | null;
}

interface MemorySettingsTerminalProps {
  onSettingsChange: (settings: MemorySettings) => void;
  onCognify?: () => void;
  cognifyDisabled?: boolean;
}

type Step =
  | "menu"
  | "help"
  | "graph-model"
  | "graph-model-edit"
  | "prompt"
  | "prompt-edit"
  | "llm"
  | "llm-edit";

interface TerminalLine {
  type: "text" | "options" | "editor" | "success";
  content: string;
  options?: { label: string; action: () => void }[];
}

const TERMINAL_FONT = "'SF Mono', 'Fira Code', 'Cascadia Code', monospace";

/* Company palette */
const C = {
  purple:      tokens.purple,
  purpleLight: tokens.purpleLight,
  green:       tokens.green,
  bgLight:     tokens.bgPage,
  border:      tokens.border,
  muted:       tokens.textMuted,
  dark:        tokens.textBody,
} as const;

/* ── Breadcrumb labels ── */
const STEP_BREADCRUMBS: Record<Step, string> = {
  menu:              "~/settings",
  help:              "~/settings/help",
  "graph-model":     "~/settings/graph-model",
  "graph-model-edit":"~/settings/graph-model/edit",
  prompt:            "~/settings/prompt",
  "prompt-edit":     "~/settings/prompt/edit",
  llm:               "~/settings/llm",
  "llm-edit":        "~/settings/llm/edit",
};

/* ── localStorage key ── */
const STORAGE_KEY = "cognee-memory-settings";

/* ── Typewriter sub-component (#7) ── */
function TypewriterText({ text, delay = 0 }: { text: string; delay?: number }) {
  const [count, setCount] = useState(0);
  const [started, setStarted] = useState(delay === 0);

  useEffect(() => {
    if (delay === 0) return;
    const t = setTimeout(() => setStarted(true), delay);
    return () => clearTimeout(t);
  }, [delay]);

  useEffect(() => {
    if (!started || count >= text.length) return;
    const t = setTimeout(() => setCount((c) => c + 1), 14);
    return () => clearTimeout(t);
  }, [started, count, text.length]);

  if (!started) return <span style={{ visibility: "hidden" }}>{text}</span>;
  return (
    <>
      {text.slice(0, count)}
      {count < text.length && (
        <span
          style={{
            display: "inline-block",
            width: "0.45em",
            height: "1em",
            background: C.green,
            verticalAlign: "text-bottom",
            marginLeft: 1,
            animation: "cursorBlink 0.6s step-end infinite",
          }}
        />
      )}
    </>
  );
}

export default function MemorySettingsTerminal({
  onSettingsChange,
  onCognify,
  cognifyDisabled = false,
}: MemorySettingsTerminalProps) {
  const { cogniInstance } = useCogniInstance();
  const [step, setStep] = useState<Step>("menu");
  const [lines, setLines] = useState<TerminalLine[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [editorValue, setEditorValue] = useState("");
  const [settings, setSettings] = useState<MemorySettings>({
    graphModel: null,
    customPrompt: null,
    llmModel: null,
  });
  const [flashActive, setFlashActive] = useState(false);
  const [comingSoon, setComingSoon] = useState(false);
  const [settingsChanged, setSettingsChanged] = useState(false);

  const terminalRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  /* ── Restore settings from localStorage on mount ── */
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed: MemorySettings = JSON.parse(stored);
        setSettings(parsed);
        onSettingsChange(parsed);
      }
    } catch {
      // corrupted data — ignore and start fresh
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ── Sync from backend once cogniInstance is ready ── */
  useEffect(() => {
    if (!cogniInstance) return;
    loadMemorySettingsFromBackend(cogniInstance)
      .then((data) => {
        if (data) {
          const fromBackend = data as MemorySettings;
          setSettings(fromBackend);
          onSettingsChange(fromBackend);
          try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(fromBackend));
          } catch {
            // storage unavailable — ignore
          }
        }
      })
      .catch(() => {
        // backend unavailable — keep localStorage data
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cogniInstance]);

  const scrollToBottom = useCallback(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [lines, step, scrollToBottom]);

  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  /* (#8) Save flash trigger */
  const triggerFlash = useCallback(() => {
    setFlashActive(true);
    setTimeout(() => setFlashActive(false), 600);
  }, []);

  const updateSettings = useCallback(
    (patch: Partial<MemorySettings>) => {
      const next = { ...settingsRef.current, ...patch };
      setSettings(next);
      onSettingsChange(next);
      setSettingsChanged(true);
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        // storage full or unavailable — ignore
      }
      if (cogniInstance) {
        syncMemorySettings(cogniInstance, next).catch(() => {
          // backend sync failed — localStorage still has the data
        });
      }
    },
    [onSettingsChange],
  );

  const handleCognifyClick = useCallback(() => {
    setSettingsChanged(false);
    onCognify?.();
  }, [onCognify]);

  /* (#2) Configured count for status bar */
  const configuredCount = [
    // settings.graphModel,
    settings.customPrompt,
    settings.llmModel,
  ].filter(Boolean).length;

  // ──────────────── Build lines for each step ────────────────

  const buildMenu = useCallback((): TerminalLine[] => {
    const statusLine = (label: string, val: unknown) =>
      val ? `${label}: configured` : `${label}: (default)`;

    return [
      { type: "text", content: "> cognee settings v1.0" },
      { type: "text", content: "> Type a number or click an option to configure:" },
      {
        type: "options",
        content: "",
        options: [
          // {
          //   label: `[1] Graph Model — ${statusLine("schema", settings.graphModel).split(": ")[1]}`,
          //   action: () => goTo("graph-model"),
          // },
          {
            label: `[1] Cognify Prompt — ${statusLine("prompt", settings.customPrompt).split(": ")[1]}`,
            action: () => goTo("prompt"),
          },
          {
            label: `[2] LLM Settings — ${statusLine("model", settings.llmModel).split(": ")[1]}`,
            action: () => goTo("llm"),
          },
        ],
      },
    ];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings]);

  const goTo = useCallback((next: Step, initialEditorValue?: string) => {
    setStep(next);
    setInputValue("");
    setEditorValue(initialEditorValue ?? "");
    setComingSoon(false);
  }, []);

  const buildHelpLines = useCallback((): TerminalLine[] => {
    return [
      { type: "text", content: "> Available commands:" },
      { type: "text", content: "  1, prompt    — Configure the cognify extraction prompt" },
      { type: "text", content: "  2, llm       — Configure the LLM provider and model" },
      { type: "text", content: "  back         — Go back one level" },
      { type: "text", content: "  menu         — Return to main menu" },
      { type: "text", content: "  help         — Show this help message" },
      { type: "text", content: "" },
      { type: "text", content: "> You can also click the buttons to navigate." },
      {
        type: "options",
        content: "",
        options: [{ label: "[Back to Menu]", action: () => goTo("menu") }],
      },
    ];
  }, [goTo]);

  const buildGraphModelLines = useCallback((): TerminalLine[] => {
    return [
      { type: "text", content: "> Select a domain template or start from scratch:" },
      {
        type: "options",
        content: "",
        options: [
          ...graphModelPresets.map((p) => ({
            label: `[${p.label}]`,
            action: () => {
              goTo("graph-model-edit", JSON.stringify(p.schema, null, 2));
            },
          })),
          {
            label: "[Custom]",
            action: () => {
              goTo(
                "graph-model-edit",
                JSON.stringify(
                  { name: "CustomEntity", fields: [{ name: "name", type: "str" }] },
                  null,
                  2,
                ),
              );
            },
          },
        ],
      },
      {
        type: "options",
        content: "",
        options: [{ label: "[Back]", action: () => goTo("menu") }],
      },
    ];
  }, [goTo]);

  const buildGraphModelEditLines = useCallback((): TerminalLine[] => {
    return [
      { type: "text", content: "> Template loaded. Edit below, then click Save to confirm:" },
      { type: "editor", content: editorValue },
      {
        type: "options",
        content: "",
        options: [
          {
            label: "[Save]",
            action: () => {
              try {
                const parsed = JSON.parse(editorValue);
                updateSettings({ graphModel: parsed });
                triggerFlash();
                goTo("menu");
              } catch {
                // invalid JSON — user can keep editing
              }
            },
          },
          {
            label: "[Reset]",
            action: () => {
              setEditorValue(
                JSON.stringify(
                  { name: "CustomEntity", fields: [{ name: "name", type: "str" }] },
                  null,
                  2,
                ),
              );
            },
          },
          { label: "[Back]", action: () => goTo("graph-model") },
        ],
      },
    ];
  }, [editorValue, goTo, updateSettings, triggerFlash]);

  const buildPromptLines = useCallback((): TerminalLine[] => {
    return [
      { type: "text", content: `> Current prompt: ${settings.customPrompt ? "(custom)" : "(default)"}` },
      { type: "text", content: "> Pick a template or write your own:" },
      { type: "text", content: "> Templates: Technical (algorithms, specs), Business (companies, metrics), Biographical (people, events), Legal (laws, regulations)" },
      {
        type: "options",
        content: "",
        options: [
          ...promptPresets.map((p) => ({
            label: `[${p.label}]`,
            action: () => {
              goTo("prompt-edit", p.prompt);
            },
          })),
          {
            label: "[Custom]",
            action: () => {
              goTo("prompt-edit", settings.customPrompt ?? "");
            },
          },
        ],
      },
      {
        type: "options",
        content: "",
        options: [{ label: "[Back]", action: () => goTo("menu") }],
      },
    ];
  }, [settings.customPrompt, goTo]);

  const buildPromptEditLines = useCallback((): TerminalLine[] => {
    return [
      { type: "text", content: "> Edit your extraction prompt:" },
      { type: "editor", content: editorValue },
      {
        type: "options",
        content: "",
        options: [
          {
            label: "[Save]",
            action: () => {
              updateSettings({ customPrompt: editorValue || null });
              triggerFlash();
              goTo("menu");
            },
          },
          {
            label: "[Reset to Default]",
            action: () => {
              setEditorValue(DEFAULT_GRAPH_PROMPT);
            },
          },
          {
            label: "[Clear]",
            action: () => {
              updateSettings({ customPrompt: null });
              goTo("menu");
            },
          },
          { label: "[Back]", action: () => goTo("prompt") },
        ],
      },
    ];
  }, [editorValue, goTo, updateSettings, triggerFlash]);

  const buildLLMLines = useCallback((): TerminalLine[] => {
    return [
      { type: "text", content: `> Current model: ${settings.llmModel ?? "(default)"}` },
      { type: "text", content: "> Select LLM provider and model:" },
      { type: "text", content: "> The LLM processes your documents during cognification. Different models vary in speed and accuracy." },
      {
        type: "options",
        content: "",
        options: [
          ...llmPresets.map((p) => ({
            label: `[${p.label}]`,
            action: () => {
              goTo("llm-edit", p.model);
            },
          })),
          {
            label: "[Custom]",
            action: () => {
              goTo("llm-edit", settings.llmModel ?? "");
            },
          },
        ],
      },
      {
        type: "options",
        content: "",
        options: [{ label: "[Back]", action: () => goTo("menu") }],
      },
    ];
  }, [settings.llmModel, goTo]);

  const buildLLMEditLines = useCallback((): TerminalLine[] => {
    return [
      { type: "text", content: `> Model: ${editorValue || "(type a model identifier)"}` },
      { type: "editor", content: editorValue },
      {
        type: "options",
        content: "",
        options: [
          {
            label: "[Save]",
            action: () => {
              updateSettings({ llmModel: editorValue || null });
              triggerFlash();
              goTo("menu");
            },
          },
          {
            label: "[Clear]",
            action: () => {
              updateSettings({ llmModel: null });
              goTo("menu");
            },
          },
          { label: "[Back]", action: () => goTo("llm") },
        ],
      },
    ];
  }, [editorValue, goTo, updateSettings, triggerFlash]);

  // ──────────────── Derive lines from step ────────────────

  useEffect(() => {
    switch (step) {
      case "menu":             setLines(buildMenu()); break;
      case "help":             setLines(buildHelpLines()); break;
      case "graph-model":      setLines(buildGraphModelLines()); break;
      case "graph-model-edit": setLines(buildGraphModelEditLines()); break;
      case "prompt":           setLines(buildPromptLines()); break;
      case "prompt-edit":      setLines(buildPromptEditLines()); break;
      case "llm":              setLines(buildLLMLines()); break;
      case "llm-edit":         setLines(buildLLMEditLines()); break;
    }
  }, [
    step,
    buildMenu,
    buildHelpLines,
    buildGraphModelLines,
    buildGraphModelEditLines,
    buildPromptLines,
    buildPromptEditLines,
    buildLLMLines,
    buildLLMEditLines,
  ]);

  // ──────────────── Keyboard handler ────────────────

  const handleInputKey = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key !== "Enter") return;
      const val = inputValue.trim().toLowerCase();
      setInputValue("");

      if (!val) return;

      let handled = false;

      if (step === "menu") {
        // if (val === "1" || val === "graph") { goTo("graph-model"); handled = true; }
        if (val === "1" || val === "prompt") { goTo("prompt"); handled = true; }
        else if (val === "2" || val === "llm") { goTo("llm"); handled = true; }
      }

      if (val === "back" || val === "menu") {
        if (step.includes("edit")) {
          goTo(step.replace("-edit", "") as Step);
        } else if (step !== "menu") {
          goTo("menu");
        }
        handled = true;
      }

      if (val === "help") {
        goTo("help");
        handled = true;
      }

      if (!handled) {
        setComingSoon(true);
        setTimeout(() => setComingSoon(false), 3000);
      }
    },
    [inputValue, step, goTo],
  );

  // ──────────────── Render a single line ────────────────

  const renderLine = (line: TerminalLine, i: number) => {
    switch (line.type) {
      /* (#3) Green text glow + (#7) typewriter */
      case "text":
        return (
          <div
            key={`${step}-text-${i}`}
            style={{
              color: C.green,
              textShadow: "0 0 6px rgba(13,255,0,0.5), 0 0 14px rgba(13,255,0,0.18)",
              lineHeight: 1.8,
            }}
          >
            <TypewriterText text={line.content} delay={i * 130} />
          </div>
        );

      case "success":
        return (
          <div
            key={`${step}-success-${i}`}
            style={{
              color: C.green,
              textShadow: "0 0 8px rgba(13,255,0,0.65), 0 0 20px rgba(13,255,0,0.25)",
              fontWeight: 600,
              lineHeight: 1.8,
            }}
          >
            <TypewriterText text={line.content} delay={i * 130} />
          </div>
        );

      /* (#4) Purple glow on hover */
      case "options":
        return (
          <div
            key={`${step}-opts-${i}`}
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "0.5rem",
              margin: "0.375rem 0",
              animation: `termFadeIn 0.35s ease-out ${i * 0.1}s both`,
            }}
          >
            {line.options?.map((opt, j) => (
              <button
                key={j}
                onClick={opt.action}
                style={{
                  background: "rgba(92, 16, 244, 0.12)",
                  border: "1px solid rgba(165, 80, 255, 0.35)",
                  borderRadius: "0.25rem",
                  color: C.purpleLight,
                  fontFamily: TERMINAL_FONT,
                  fontSize: "0.8125rem",
                  padding: "0.25rem 0.75rem",
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                  boxShadow: "none",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "rgba(165, 80, 255, 0.22)";
                  e.currentTarget.style.borderColor = "rgba(165, 80, 255, 0.7)";
                  e.currentTarget.style.color = C.bgLight;
                  e.currentTarget.style.boxShadow =
                    "0 0 10px rgba(165,80,255,0.45), 0 0 24px rgba(92,16,244,0.2)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "rgba(92, 16, 244, 0.12)";
                  e.currentTarget.style.borderColor = "rgba(165, 80, 255, 0.35)";
                  e.currentTarget.style.color = C.purpleLight;
                  e.currentTarget.style.boxShadow = "none";
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        );

      case "editor":
        return (
          <div
            key={`${step}-editor-${i}`}
            style={{
              margin: "0.375rem 0",
              animation: "termFadeIn 0.3s ease-out",
            }}
          >
            <textarea
              value={editorValue}
              onChange={(e) => setEditorValue(e.target.value)}
              style={{
                width: "100%",
                minHeight:
                  step === "prompt-edit" || step === "llm-edit"
                    ? "4rem"
                    : "10rem",
                background: "rgba(244, 244, 244, 0.06)",
                border: "1px solid rgba(216, 216, 216, 0.25)",
                borderRadius: "0.375rem",
                color: C.bgLight,
                fontFamily: TERMINAL_FONT,
                fontSize: "0.8125rem",
                lineHeight: 1.7,
                padding: "0.5rem 0.75rem",
                resize: "vertical",
                outline: "none",
                transition: "border-color 0.2s ease, box-shadow 0.2s ease",
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = C.purpleLight;
                e.currentTarget.style.boxShadow =
                  "0 0 8px rgba(165,80,255,0.25), inset 0 0 20px rgba(92,16,244,0.05)";
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = "rgba(216, 216, 216, 0.25)";
                e.currentTarget.style.boxShadow = "none";
              }}
            />
          </div>
        );

      default:
        return null;
    }
  };

  // ──────────────── Component render ────────────────

  return (
    <Stack
      className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0]"
      bg="white"
    >
      <Title size="h2" c={C.dark} mb="0.125rem">
        Knowledge Extraction Settings
      </Title>
      <Text c={C.muted} size="lg" mb="1rem">
        Control how Cognee extracts knowledge from your documents
      </Text>

      {/* (#5) Animated gradient border glow */}
      <Box
        style={{
          background: C.dark,
          borderRadius: "0.5rem",
          overflow: "hidden",
          position: "relative",
          animation: "borderGlow 3s ease-in-out infinite",
        }}
      >
        {/* (#1) macOS-style title bar */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            padding: "0.5rem 0.75rem",
            background: "rgba(0, 0, 0, 0.35)",
            borderBottom: "1px solid rgba(165, 80, 255, 0.15)",
            gap: "0.5rem",
            position: "relative",
            userSelect: "none",
          }}
        >
          <div style={{ display: "flex", gap: "0.4rem" }}>
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                background: "#ff5f57",
                boxShadow: "0 0 4px rgba(255,95,87,0.4)",
              }}
            />
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                background: C.green,
                boxShadow: `0 0 4px rgba(13,255,0,0.4)`,
              }}
            />
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                background: C.purpleLight,
                boxShadow: "0 0 4px rgba(165,80,255,0.4)",
              }}
            />
          </div>
          <span
            style={{
              position: "absolute",
              left: "50%",
              transform: "translateX(-50%)",
              fontFamily: TERMINAL_FONT,
              fontSize: "0.6875rem",
              color: C.muted,
              letterSpacing: "0.05em",
            }}
          >
            cognee-settings &mdash; zsh
          </span>
        </div>

        {/* (#6) CRT scanline overlay */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            zIndex: 5,
            background:
              "repeating-linear-gradient(to bottom, transparent, transparent 2px, rgba(0,0,0,0.045) 2px, rgba(0,0,0,0.045) 4px)",
            mixBlendMode: "multiply",
          }}
        />

        {/* (#8) Save flash overlay */}
        {flashActive && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              background:
                "radial-gradient(ellipse at center, rgba(13,255,0,0.18) 0%, transparent 70%)",
              animation: "saveFlash 0.6s ease-out forwards",
              pointerEvents: "none",
              zIndex: 10,
            }}
          />
        )}

        {/* Terminal content area */}
        <div
          ref={terminalRef}
          style={{
            padding: "0.75rem 1rem",
            minHeight: "8rem",
            maxHeight: "24rem",
            overflowY: "auto",
            fontFamily: TERMINAL_FONT,
            fontSize: "0.8125rem",
            lineHeight: 1.7,
            position: "relative",
            zIndex: 1,
          }}
        >
          {/* (#9) Smooth step transition wrapper */}
          <div key={step} style={{ animation: "termFadeIn 0.25s ease-out" }}>
            {lines.map((line, i) => renderLine(line, i))}
          </div>

          {/* Coming soon message */}
          {comingSoon && (
            <div
              style={{
                color: C.purpleLight,
                textShadow: "0 0 6px rgba(165,80,255,0.5)",
                fontFamily: TERMINAL_FONT,
                fontSize: "0.8125rem",
                lineHeight: 1.8,
                marginTop: "0.25rem",
                animation: "termFadeIn 0.25s ease-out",
              }}
            >
              &gt; Unknown command. Type &quot;help&quot; to see available commands.
            </div>
          )}

          {/* Input line */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              marginTop: "0.375rem",
              color: C.green,
              textShadow: "0 0 6px rgba(13,255,0,0.5)",
            }}
          >
            <span style={{ marginRight: "0.5rem" }}>&gt;</span>
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleInputKey}
              placeholder="type a command..."
              style={{
                flex: 1,
                background: "transparent",
                border: "none",
                outline: "none",
                color: C.green,
                fontFamily: TERMINAL_FONT,
                fontSize: "0.8125rem",
                lineHeight: 1.7,
                caretColor: C.green,
                textShadow: "0 0 6px rgba(13,255,0,0.5)",
              }}
            />
            <span
              style={{
                display: "inline-block",
                width: "0.5rem",
                height: "1em",
                background: C.purpleLight,
                boxShadow: "0 0 6px rgba(165,80,255,0.5)",
                marginLeft: "0.25rem",
                verticalAlign: "text-bottom",
                animation: "cursorBlink 1s step-end infinite",
              }}
            />
          </div>
        </div>

        {/* (#2) Bottom status bar */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "0.3rem 0.75rem",
            background: "rgba(0, 0, 0, 0.35)",
            borderTop: "1px solid rgba(165, 80, 255, 0.15)",
            fontFamily: TERMINAL_FONT,
            fontSize: "0.6875rem",
            userSelect: "none",
          }}
        >
          <span style={{ color: C.muted }}>{STEP_BREADCRUMBS[step]}</span>
          <span
            style={{
              color: configuredCount > 0 ? C.green : C.muted,
              textShadow:
                configuredCount > 0
                  ? "0 0 4px rgba(13,255,0,0.4)"
                  : "none",
            }}
          >
            {configuredCount}/2 configured
          </span>
        </div>
      </Box>

      <Button
        mt="1rem"
        color="primary2.6"
        disabled={cognifyDisabled || !settingsChanged}
        onClick={handleCognifyClick}
      >
        Cognify
      </Button>

      {/* ── Keyframe animations ── */}
      <style>{`
        @keyframes cursorBlink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        @keyframes termFadeIn {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes borderGlow {
          0%, 100% {
            box-shadow:
              0 0 8px  rgba(92, 16, 244, 0.3),
              0 0 20px rgba(92, 16, 244, 0.08);
          }
          50% {
            box-shadow:
              0 0 14px rgba(165, 80, 255, 0.45),
              0 0 32px rgba(165, 80, 255, 0.12);
          }
        }
        @keyframes saveFlash {
          0%   { opacity: 1; }
          100% { opacity: 0; }
        }
      `}</style>
    </Stack>
  );
}
