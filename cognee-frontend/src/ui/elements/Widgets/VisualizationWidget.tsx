"use client";

import { Center, Flex, Portal, Stack, Title, UnstyledButton } from "@mantine/core";
import { useCallback, useEffect, useRef, useState } from "react";
import { tokens } from "@/ui/theme/tokens";

const FIT_SCRIPT = `
<script>
(function() {
  function resizeAndFit() {
    var containers = document.querySelectorAll('[id*="network"], [id*="mynetwork"], [id*="graph"]');
    if (containers.length === 0) {
      var canvas = document.querySelector('canvas');
      if (canvas) containers = [canvas.parentElement];
    }
    for (var i = 0; i < containers.length; i++) {
      if (containers[i]) {
        containers[i].style.width = '100%';
        containers[i].style.height = '100vh';
      }
    }
    document.documentElement.style.margin = '0';
    document.documentElement.style.padding = '0';
    document.documentElement.style.height = '100%';
    document.documentElement.style.overflow = 'hidden';
    document.body.style.margin = '0';
    document.body.style.padding = '0';
    document.body.style.height = '100%';
    document.body.style.overflow = 'hidden';

    if (typeof network !== 'undefined' && network) {
      network.setSize('100%', '100%');
      network.redraw();
      network.fit({ animation: { duration: 200, easingFunction: 'easeInOutQuad' } });
    }
  }

  var attempts = 0;
  var waitForNetwork = setInterval(function() {
    attempts++;
    if (typeof network !== 'undefined' && network && network.fit) {
      clearInterval(waitForNetwork);
      resizeAndFit();
      network.once('stabilized', function() { resizeAndFit(); });
    }
    if (attempts > 100) clearInterval(waitForNetwork);
  }, 100);

  window.addEventListener('resize', function() { resizeAndFit(); });
  window.addEventListener('message', function(e) {
    if (e.data === 'fit') resizeAndFit();
  });
})();
</script>`;

function injectFitScript(html: string): string {
  if (html.includes("</body>")) {
    return html.replace("</body>", FIT_SCRIPT + "</body>");
  }
  return html + FIT_SCRIPT;
}

interface VisualizationWidgetProps {
  selectedDatasetId: string | null;
  visualizeDataset: (datasetId: string) => Promise<Response>;
  dataVersion?: number;
  onReady?: () => void;
  hidden?: boolean;
}

export default function VisualizationWidget({
  selectedDatasetId,
  visualizeDataset,
  dataVersion = 0,
  onReady,
  hidden = false,
}: VisualizationWidgetProps) {
  const [visualization, setVisualization] = useState<string | null>(null);
  const prevDatasetId = useRef<string | null>(null);
  const prevVersion = useRef(dataVersion);
  const containerRef = useRef<HTMLDivElement>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [containerHeight, setContainerHeight] = useState(750);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [manualHeight, setManualHeight] = useState<number | null>(null);
  const isDragging = useRef(false);

  useEffect(() => {
    const datasetChanged = selectedDatasetId !== prevDatasetId.current;
    const versionChanged = dataVersion !== prevVersion.current;

    if (selectedDatasetId && (datasetChanged || versionChanged)) {
      prevDatasetId.current = selectedDatasetId;
      prevVersion.current = dataVersion;
      visualizeDataset(selectedDatasetId)
        .then((resp) => resp.text())
        .then((html) => {
          setVisualization(injectFitScript(html));
          onReady?.();
        })
        .catch(() => {
          setVisualization(null);
          onReady?.();
        });
    } else if (!selectedDatasetId) {
      prevDatasetId.current = null;
      prevVersion.current = dataVersion;
      setVisualization("");
    }
  }, [selectedDatasetId, visualizeDataset, dataVersion, onReady]);

  const sendFitMessage = useCallback(() => {
    if (iframeRef.current?.contentWindow) {
      iframeRef.current.contentWindow.postMessage("fit", "*");
    }
  }, []);

  useEffect(() => {
    const onResize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const available = window.innerHeight - rect.top - 40;
        setContainerHeight(Math.max(400, available));
      }
      sendFitMessage();
    };

    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [sendFitMessage]);

  const wasHidden = useRef(hidden);
  useEffect(() => {
    if (wasHidden.current && !hidden) {
      requestAnimationFrame(() => {
        if (containerRef.current) {
          const rect = containerRef.current.getBoundingClientRect();
          const available = window.innerHeight - rect.top - 40;
          setContainerHeight(Math.max(400, available));
        }
        sendFitMessage();
      });
    }
    wasHidden.current = hidden;
  }, [hidden, sendFitMessage]);

  // Escape key to exit fullscreen
  useEffect(() => {
    if (!isFullscreen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsFullscreen(false);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [isFullscreen]);

  // Re-fit graph when toggling fullscreen
  useEffect(() => {
    requestAnimationFrame(() => sendFitMessage());
  }, [isFullscreen, sendFitMessage]);

  const handleToggleFullscreen = useCallback(() => {
    setIsFullscreen((v) => !v);
  }, []);

  const handleResizeStart = useCallback(
    (startY: number) => {
      isDragging.current = true;
      document.body.style.userSelect = "none";
      document.body.style.cursor = "row-resize";

      const onMove = (clientY: number) => {
        if (!isDragging.current || !containerRef.current) return;
        const top = containerRef.current.getBoundingClientRect().top;
        const newHeight = Math.min(
          Math.max(200, clientY - top),
          window.innerHeight - 100,
        );
        setManualHeight(newHeight);
      };

      const onEnd = () => {
        isDragging.current = false;
        document.body.style.userSelect = "";
        document.body.style.cursor = "";
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", onEnd);
        document.removeEventListener("touchmove", handleTouchMove);
        document.removeEventListener("touchend", onEnd);
      };

      const handleMouseMove = (e: MouseEvent) => onMove(e.clientY);
      const handleTouchMove = (e: TouchEvent) => onMove(e.touches[0].clientY);

      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", onEnd);
      document.addEventListener("touchmove", handleTouchMove);
      document.addEventListener("touchend", onEnd);
    },
    [],
  );

  const resolvedHeight = manualHeight !== null ? manualHeight : containerHeight;

  const toggleButton = (
    <UnstyledButton
      onClick={handleToggleFullscreen}
      title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
      style={{
        padding: "4px 8px",
        borderRadius: 4,
        color: tokens.textSecondary,
        fontSize: "1.125rem",
        lineHeight: 1,
      }}
    >
      {isFullscreen ? (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="4 14 10 14 10 20" />
          <polyline points="20 10 14 10 14 4" />
          <line x1="14" y1="10" x2="21" y2="3" />
          <line x1="3" y1="21" x2="10" y2="14" />
        </svg>
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="15 3 21 3 21 9" />
          <polyline points="9 21 3 21 3 15" />
          <line x1="21" y1="3" x2="14" y2="10" />
          <line x1="3" y1="21" x2="10" y2="14" />
        </svg>
      )}
    </UnstyledButton>
  );

  const emptyState = (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: "0.75rem",
        color: tokens.textPlaceholder,
      }}
    >
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="2" />
        <circle cx="5" cy="7" r="2" />
        <circle cx="19" cy="7" r="2" />
        <circle cx="5" cy="17" r="2" />
        <circle cx="19" cy="17" r="2" />
        <line x1="7" y1="7" x2="10" y2="11" />
        <line x1="17" y1="7" x2="14" y2="11" />
        <line x1="7" y1="17" x2="10" y2="13" />
        <line x1="17" y1="17" x2="14" y2="13" />
      </svg>
      <p style={{ margin: 0, fontSize: "1rem", fontWeight: 500 }}>
        No visualization available
      </p>
      <p style={{ margin: 0, fontSize: "0.875rem" }}>
        Add data and cognify to generate a knowledge graph
      </p>
    </div>
  );

  return (
    <>
      {/* Normal (non-fullscreen) widget */}
      <Stack
        ref={containerRef}
        className="rounded-[0.5rem] px-[2rem] pt-[1.5rem] pb-[1.75rem] !gap-[0]"
        bg="white"
      >
        <Flex justify="space-between" align="center" mb="1.625rem">
          <Title size="h2">Visualization</Title>
          {toggleButton}
        </Flex>
        <Center w="100%">
          {visualization ? (
            <iframe
              ref={isFullscreen ? undefined : iframeRef}
              srcDoc={visualization}
              style={{
                width: "100%",
                height: `${resolvedHeight}px`,
                border: "none",
                display: "block",
                visibility: isFullscreen ? "hidden" : "visible",
              }}
            />
          ) : (
            <div style={{ width: "100%", height: `${resolvedHeight}px` }}>
              {emptyState}
            </div>
          )}
        </Center>
        <div
          onMouseDown={(e) => handleResizeStart(e.clientY)}
          onTouchStart={(e) => handleResizeStart(e.touches[0].clientY)}
          onDoubleClick={() => setManualHeight(null)}
          style={{
            height: 6,
            width: "100%",
            cursor: "row-resize",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginTop: 4,
            borderRadius: 3,
            background: "#f3f4f6",
            transition: "background 0.15s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#e5e7eb")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "#f3f4f6")}
        >
          <div
            style={{
              width: 32,
              height: 3,
              borderRadius: 2,
              background: "#d1d5db",
            }}
          />
        </div>
      </Stack>

      {/* Fullscreen overlay — rendered at body level via Portal to escape any ancestor clipping */}
      {isFullscreen && (
        <Portal>
          <div
            style={{
              position: "fixed",
              inset: 0,
              zIndex: 300,
              background: "white",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <Flex
              justify="space-between"
              align="center"
              px="2rem"
              pt="1.5rem"
              pb="1rem"
            >
              <Title size="h2">Visualization</Title>
              {toggleButton}
            </Flex>
            <div style={{ flex: 1, overflow: "hidden" }}>
              {visualization ? (
                <iframe
                  ref={iframeRef}
                  srcDoc={visualization}
                  style={{
                    width: "100%",
                    height: "100%",
                    border: "none",
                    display: "block",
                  }}
                />
              ) : (
                emptyState
              )}
            </div>
          </div>
        </Portal>
      )}
    </>
  );
}
