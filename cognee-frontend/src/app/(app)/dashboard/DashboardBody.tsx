"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import useDatasets from "@/modules/ingestion/useDatasets";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { Stack, Flex, Text, Title, CloseButton, Button } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import DatasetSearchWidget from "@/ui/elements/Widgets/DatasetSearchWidget";
import DatasetAddWidget from "@/ui/elements/Widgets/DatasetAddWidget";
import VisualizationWidget from "@/ui/elements/Widgets/VisualizationWidget";
import { CogneeInstance } from "@/modules/instances/types";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import CognifyActivityPanel from "@/ui/elements/Widgets/CognifyActivityPanel";
import FrostedLoadingOverlay from "./FrostedLoadingOverlay";
import { LoadingProvider, useLoading } from "./LoadingContext";
import DashboardControlPanel from "./DashboardControlPanel";
import { tokens } from "@/ui/theme/tokens";
import { loadPrompts } from "@/modules/prompts/storage";
import type { Prompt } from "@/modules/prompts/storage";

function BusyScreen({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <Stack
      className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem]"
      bg="white"
      style={{
        minHeight: 400,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        animation: "cognifyFadeIn 0.4s ease-out",
      }}
    >
      <div style={{ textAlign: "center" }}>
        <div style={{ position: "relative", width: 80, height: 80, margin: "0 auto 2rem" }}>
          <div
            style={{
              position: "absolute",
              inset: 0,
              borderRadius: "50%",
              border: "3px solid rgba(101, 16, 244, 0.1)",
            }}
          />
          <div
            style={{
              position: "absolute",
              inset: 0,
              borderRadius: "50%",
              border: "3px solid transparent",
              borderTopColor: tokens.purple,
              animation: "cognifySpin 1s cubic-bezier(0.4, 0, 0.2, 1) infinite",
            }}
          />
          <div
            style={{
              position: "absolute",
              inset: 12,
              borderRadius: "50%",
              border: "3px solid transparent",
              borderTopColor: tokens.green,
              animation: "cognifySpin 1.4s cubic-bezier(0.4, 0, 0.2, 1) infinite reverse",
            }}
          />
          <div
            style={{
              position: "absolute",
              inset: 24,
              borderRadius: "50%",
              border: "3px solid transparent",
              borderTopColor: tokens.purple,
              opacity: 0.5,
              animation: "cognifySpin 1.8s cubic-bezier(0.4, 0, 0.2, 1) infinite",
            }}
          />
        </div>
        <p
          style={{
            margin: 0,
            fontSize: "1.125rem",
            fontWeight: 500,
            color: tokens.textDark,
            letterSpacing: "0.01em",
            animation: "cognifyPulse 2.5s ease-in-out infinite",
          }}
        >
          {title}
        </p>
        <p
          style={{
            margin: "0.5rem 0 0",
            fontSize: "0.875rem",
            fontWeight: 400,
            color: tokens.textSecondary,
            animation: "cognifyPulse 2.5s ease-in-out infinite 0.3s",
          }}
        >
          {subtitle}
        </p>
      </div>
      <style>{`
        @keyframes cognifySpin { to { transform: rotate(360deg); } }
        @keyframes cognifyPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
        @keyframes cognifyFadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </Stack>
  );
}

type BusyState = "idle" | "cognifying" | "loading-dataset";

interface DashboardContentProps {
  instance: CogneeInstance;
  busyState: BusyState;
  dataVersion: number;
  onCognifyStart: () => void;
  onCognifyComplete: () => void;
  onWidgetReady: () => void;
  onDatasetChangeStart: () => void;
}

function DashboardContent({
  instance,
  busyState,
  dataVersion,
  onCognifyStart,
  onCognifyComplete,
  onWidgetReady,
  onDatasetChangeStart,
}: DashboardContentProps) {
  const { stopLoading } = useLoading();

  // Onboarding banner state
  const [showOnboarding, setShowOnboarding] = useState(() => {
    if (typeof window === "undefined") return true;
    return !localStorage.getItem("cognee-onboarding-dismissed");
  });

  const dismissOnboarding = useCallback(() => {
    localStorage.setItem("cognee-onboarding-dismissed", "1");
    setShowOnboarding(false);
  }, []);

  // Post-cognify "What's next?" prompt
  const [showWhatsNext, setShowWhatsNext] = useState(false);
  const prevBusyState = useRef<BusyState>(busyState);
  const whatsNextTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (busyState === "idle" && prevBusyState.current === "cognifying") {
      setShowWhatsNext(true);
      whatsNextTimer.current = setTimeout(() => setShowWhatsNext(false), 30000);
    }
    prevBusyState.current = busyState;
    return () => {
      if (whatsNextTimer.current) clearTimeout(whatsNextTimer.current);
    };
  }, [busyState]);

  const dismissWhatsNext = useCallback(() => {
    setShowWhatsNext(false);
    if (whatsNextTimer.current) clearTimeout(whatsNextTimer.current);
  }, []);

  const onDatasetsReady = useCallback(() => {
    stopLoading();
  }, [stopLoading]);

  const {
    datasets,
    addDataset,
    refreshDatasets,
    searchDataset,
    visualizeDataset,
    getDatasetData,
  } = useDatasets(instance, "", onDatasetsReady);

  const [llmModel, setLlmModel] = useState("gpt-4o");

  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [activePromptId, setActivePromptId] = useState<string | null>(null);

  useEffect(() => {
    setPrompts(loadPrompts());
  }, []);
  const [selectedDatasetId, setSelectedDatasetIdRaw] = useState<string | null>(null);
  const restoredRef = useRef(false);
  const prevDatasetRef = useRef<string | null>(null);

  const setSelectedDatasetId = useCallback((id: string | null) => {
    const prev = prevDatasetRef.current;
    if (id === prev) return;
    prevDatasetRef.current = id;
    setSelectedDatasetIdRaw(id);
    if (id) {
      localStorage.setItem("selectedDatasetId", id);
    } else {
      localStorage.removeItem("selectedDatasetId");
    }
    if (restoredRef.current && id) {
      onDatasetChangeStart();
    }
  }, [onDatasetChangeStart]);

  useEffect(() => {
    if (datasets.length > 0 && !restoredRef.current) {
      restoredRef.current = true;
      const stored = localStorage.getItem("selectedDatasetId");
      if (stored && datasets.some((d) => d.id === stored)) {
        setSelectedDatasetId(stored);
      } else {
        setSelectedDatasetId(datasets[0].id);
      }
    }
  }, [datasets, setSelectedDatasetId]);

  const isBusy = busyState !== "idle";

  const resolvedCustomPrompt = useMemo(() => {
    if (!activePromptId) return undefined;
    return prompts.find((p) => p.id === activePromptId)?.content;
  }, [activePromptId, prompts]);

  const handleRecognify = useCallback(() => {
    if (!selectedDatasetId || isBusy) return;
    const dataset = datasets.find((d) => d.id === selectedDatasetId);
    if (!dataset) return;
    onCognifyStart();
    cognifyDataset(dataset, instance, {
      customPrompt: resolvedCustomPrompt,
      llmModel,
    })
      .then(() => onCognifyComplete())
      .catch((err) => {
        onCognifyComplete();
        notifications.show({
          title: "Cognify failed",
          message: err?.detail ?? err?.message ?? (typeof err === "string" ? err : "Something went wrong"),
          color: "red",
        });
      });
  }, [selectedDatasetId, isBusy, datasets, instance, resolvedCustomPrompt, llmModel, onCognifyStart, onCognifyComplete]);

  const busyTitle = busyState === "cognifying"
    ? "We are cognifying your data"
    : "Loading dataset";
  const busySubtitle = busyState === "cognifying"
    ? "Building knowledge graph and processing your documents..."
    : "Fetching visualization and example questions...";

  return (
    <Stack className="h-full overflow-auto" gap="0.625rem">
      <DashboardControlPanel
        datasets={datasets}
        selectedDatasetId={selectedDatasetId}
        onDatasetChange={setSelectedDatasetId}
        addDataset={addDataset}
        refreshDatasets={refreshDatasets}
        prompts={prompts}
        activePromptId={activePromptId}
        onPromptChange={setActivePromptId}
        llmModel={llmModel}
        onLlmModelChange={setLlmModel}
      />
      {showOnboarding && (
        <div
          className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem]"
          style={{ background: "#ffffff", position: "relative" }}
        >
          <CloseButton
            onClick={dismissOnboarding}
            style={{ position: "absolute", top: "0.75rem", right: "0.75rem" }}
            aria-label="Dismiss onboarding"
          />
          <Title size="h3" mb="1rem">Getting started with Cognee</Title>
          <Flex gap="2rem" wrap="wrap">
            <div style={{ flex: 1, minWidth: 160 }}>
              <Text fw={600} size="sm" c={tokens.purple} mb="0.25rem">1. Upload documents</Text>
              <Text size="sm" c={tokens.textSecondary}>Add files or paste text into a dataset</Text>
            </div>
            <div style={{ flex: 1, minWidth: 160 }}>
              <Text fw={600} size="sm" c={tokens.purple} mb="0.25rem">2. Cognify</Text>
              <Text size="sm" c={tokens.textSecondary}>Your data is transformed into a knowledge graph</Text>
            </div>
            <div style={{ flex: 1, minWidth: 160 }}>
              <Text fw={600} size="sm" c={tokens.purple} mb="0.25rem">3. Search & explore</Text>
              <Text size="sm" c={tokens.textSecondary}>Ask questions or explore the graph visualization</Text>
            </div>
          </Flex>
        </div>
      )}
      <div
        style={{
          opacity: isBusy ? 0.4 : 1,
          pointerEvents: isBusy ? "none" : "auto",
          transition: "opacity 0.3s ease",
        }}
      >
        <DatasetSearchWidget
          selectedDatasetId={selectedDatasetId}
          searchDataset={searchDataset}
          getDatasetData={getDatasetData}
          dataVersion={dataVersion}
          onReady={onWidgetReady}
        />
      </div>
      <CognifyActivityPanel isActive={busyState === "cognifying"} />
      {showWhatsNext && (
        <div
          className="rounded-[0.5rem] px-[2rem] pt-[1rem] pb-[1rem]"
          style={{
            background: "#ffffff",
            borderLeft: `4px solid ${tokens.green}`,
            position: "relative",
            animation: "answerSlideIn 0.4s ease-out forwards",
          }}
        >
          <CloseButton
            onClick={dismissWhatsNext}
            size="sm"
            style={{ position: "absolute", top: "0.5rem", right: "0.5rem" }}
            aria-label="Dismiss"
          />
          <Text fw={600} size="sm" c={tokens.textDark}>
            Knowledge graph ready!
          </Text>
          <Text size="sm" c={tokens.textSecondary}>
            Try asking a question above, or expand the graph visualization below to explore your data.
          </Text>
          <style>{`
            @keyframes answerSlideIn {
              from { opacity: 0; transform: translateY(12px) scale(0.98); }
              to { opacity: 1; transform: translateY(0) scale(1); }
            }
          `}</style>
        </div>
      )}
      <div
        style={{
          opacity: isBusy ? 0.4 : 1,
          pointerEvents: isBusy ? "none" : "auto",
          transition: "opacity 0.3s ease",
        }}
      >
        <DatasetAddWidget
          selectedDatasetId={selectedDatasetId}
          refreshDatasets={refreshDatasets}
          instance={instance}
          onCognifyStart={onCognifyStart}
          onCognifyComplete={onCognifyComplete}
          customPrompt={resolvedCustomPrompt}
          llmModel={llmModel}
          activePromptName={prompts.find((p) => p.id === activePromptId)?.name}
        />
      </div>
      <div style={{ position: "relative" }}>
        {isBusy && (
          <div style={{ position: "absolute", inset: 0, zIndex: 1 }}>
            <BusyScreen title={busyTitle} subtitle={busySubtitle} />
          </div>
        )}
        <VisualizationWidget
          selectedDatasetId={selectedDatasetId}
          visualizeDataset={visualizeDataset}
          dataVersion={dataVersion}
          onReady={onWidgetReady}
          hidden={isBusy}
        />
      </div>
    </Stack>
  );
}

function DashboardInner() {
  const { cogniInstance, statusMessage } = useCogniInstance();
  const { isLoading, stopLoading } = useLoading();

  // Safety net: force-dismiss the loading overlay after 15 s in case the
  // datasets API hangs and stopLoading() is never called normally.
  useEffect(() => {
    if (!isLoading) return;
    const t = setTimeout(stopLoading, 15000);
    return () => clearTimeout(t);
  }, [isLoading, stopLoading]);

  const instance = useMemo(() => {
    if (!cogniInstance) return null;
    return cogniInstance;
  }, [cogniInstance]);

  const [busyState, setBusyState] = useState<BusyState>("idle");
  const [dataVersion, setDataVersion] = useState(0);
  const pendingRef = useRef(0);

  const handleCognifyStart = useCallback(() => {
    setBusyState("cognifying");
  }, []);

  const handleCognifyComplete = useCallback(() => {
    pendingRef.current = 2;
    setDataVersion((v) => v + 1);
  }, []);

  const handleDatasetChangeStart = useCallback(() => {
    pendingRef.current = 2;
    setBusyState("loading-dataset");
  }, []);

  const handleWidgetReady = useCallback(() => {
    if (pendingRef.current <= 0) return;
    pendingRef.current -= 1;
    if (pendingRef.current === 0) {
      setBusyState("idle");
    }
  }, []);

  const isBusy = busyState !== "idle";

  // Show overlay while tenant is still initializing (instance not yet available)
  // OR while datasets are loading (isLoading) — but not during busy states like cognify
  const showOverlay = (!instance || isLoading) && !isBusy;

  return (
    <>
      <FrostedLoadingOverlay
        visible={showOverlay}
        title={statusMessage?.title}
        subtitle={statusMessage?.subtitle}
      />
      {instance ? (
        <DashboardContent
          instance={instance}
          busyState={busyState}
          dataVersion={dataVersion}
          onCognifyStart={handleCognifyStart}
          onCognifyComplete={handleCognifyComplete}
          onWidgetReady={handleWidgetReady}
          onDatasetChangeStart={handleDatasetChangeStart}
        />
      ) : null}
    </>
  );
}

export default function DashboardBody() {
  return (
    <LoadingProvider initialCount={1}>
      <DashboardInner />
    </LoadingProvider>
  );
}
