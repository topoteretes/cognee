"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import {
  DEFAULT_PIPELINE_SETTINGS,
  getPipelineSettingsFromStorage,
  loadPipelineSettings,
  savePipelineSettings,
  storePipelineSettingsLocally,
  type PipelineSettings,
} from "@/modules/configuration/pipelineSettings";

const CHUNK_SIZE_OPTIONS = [128, 256, 512, 1024, 2048, 4096, 8192];
const CHUNKS_PER_BATCH_OPTIONS = [1, 5, 10, 20, 50, 100, 5000];
const TOP_K_OPTIONS = [5, 10, 20, 30, 50, 100, 200];

interface Props {
  onClose: () => void;
}

function SelectField({
  label,
  description,
  value,
  options,
  onChange,
}: {
  label: string;
  description: string;
  value: number;
  options: number[];
  onChange: (v: number) => void;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <label style={{ fontSize: 13, fontWeight: 500, color: "#EDECEA" }}>{label}</label>
      <p style={{ fontSize: 12, color: "rgba(237,236,234,0.55)", margin: 0 }}>{description}</p>
      <select
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{
          marginTop: 4,
          height: 36,
          borderRadius: 8,
          border: "1px solid rgba(255,255,255,0.12)",
          padding: "0 10px",
          fontSize: 13,
          color: "#EDECEA",
          background: "rgba(255,255,255,0.06)",
          cursor: "pointer",
          outline: "none",
          appearance: "auto",
        }}
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}

function ToggleField({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: 14 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 2, flex: 1, minWidth: 0 }}>
        <label style={{ fontSize: 13, fontWeight: 500, color: "#EDECEA" }}>{label}</label>
        <p style={{ fontSize: 12, color: "rgba(237,236,234,0.55)", margin: 0 }}>{description}</p>
      </div>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className="cursor-pointer"
        style={{
          flexShrink: 0, marginTop: 2,
          width: 36, height: 20, borderRadius: 999,
          padding: 2, border: "none",
          background: checked ? "#BC9BFF" : "rgba(255,255,255,0.15)",
          transition: "background 150ms ease",
          display: "flex", alignItems: "center",
        }}
      >
        <span style={{
          width: 16, height: 16, borderRadius: "50%",
          background: "#fff",
          transform: checked ? "translateX(16px)" : "translateX(0)",
          transition: "transform 150ms ease",
          boxShadow: "0 1px 2px rgba(0,0,0,0.3)",
        }} />
      </button>
    </div>
  );
}

export default function ExtractionSettingsModal({ onClose }: Props) {
  const { cogniInstance } = useCogniInstance();
  const [values, setValues] = useState<PipelineSettings>(getPipelineSettingsFromStorage);
  const [saved, setSaved] = useState<PipelineSettings>(getPipelineSettingsFromStorage);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Portal target — null on first render (SSR) to skip mounting until the
  // client takes over, then set to document.body so the fixed overlay
  // escapes any transformed/filtered ancestor that would otherwise turn
  // `position: fixed` into containing-block-relative (the bug that put
  // this modal inside the header instead of over the viewport).
  const [portalTarget, setPortalTarget] = useState<HTMLElement | null>(null);
  useEffect(() => { setPortalTarget(document.body); }, []);

  useEffect(() => {
    if (!cogniInstance) return;
    loadPipelineSettings(cogniInstance).then((fromBackend) => {
      if (fromBackend) {
        setValues(fromBackend);
        setSaved(fromBackend);
        storePipelineSettingsLocally(fromBackend);
      }
    });
  }, [cogniInstance]);

  const isDirty =
    values.chunkSize !== saved.chunkSize ||
    values.chunksPerBatch !== saved.chunksPerBatch ||
    values.topK !== saved.topK ||
    values.includeReferences !== saved.includeReferences;

  const handleSave = async () => {
    if (!cogniInstance) return;
    setSaving(true);
    setError(null);
    try {
      await savePipelineSettings(cogniInstance, values);
      setSaved(values);
      onClose();
    } catch {
      setError("Failed to save settings. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  if (!portalTarget) return null;
  return createPortal(
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.35)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)",
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: "rgba(15,15,15,0.92)",
          backdropFilter: "blur(16px)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 12,
          padding: 24,
          width: 400,
          boxShadow: "0 20px 60px rgba(0,0,0,0.6)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ marginBottom: 20 }}>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: "#EDECEA", margin: 0 }}>
            Extraction Settings
          </h2>
          <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: "4px 0 0" }}>
            Default parameters for knowledge extraction and search
          </p>
        </div>

        {/* Divider */}
        <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "0 0 20px" }} />

        {/* Fields */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <SelectField
            label="Chunk Size"
            description="Characters per document chunk during ingestion"
            value={values.chunkSize}
            options={CHUNK_SIZE_OPTIONS}
            onChange={(v) => setValues((s) => ({ ...s, chunkSize: v }))}
          />
          <SelectField
            label="Chunks Per Batch"
            description="Chunks processed in parallel during cognification"
            value={values.chunksPerBatch}
            options={CHUNKS_PER_BATCH_OPTIONS}
            onChange={(v) => setValues((s) => ({ ...s, chunksPerBatch: v }))}
          />
          <SelectField
            label="Top-K Results"
            description="Maximum results returned per search query"
            value={values.topK}
            options={TOP_K_OPTIONS}
            onChange={(v) => setValues((s) => ({ ...s, topK: v }))}
          />
          <ToggleField
            label="Source references"
            description="Attach citations and provenance links to recall answers"
            checked={values.includeReferences}
            onChange={(v) => setValues((s) => ({ ...s, includeReferences: v }))}
          />
        </div>

        {/* Error */}
        {error && (
          <p style={{ fontSize: 12, color: "#EF4444", margin: "12px 0 0" }}>{error}</p>
        )}

        {/* Divider */}
        <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "20px 0" }} />

        {/* Footer */}
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button
            onClick={() => setValues({ ...DEFAULT_PIPELINE_SETTINGS })}
            disabled={saving}
            style={{
              height: 36,
              padding: "0 16px",
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.15)",
              background: "transparent",
              fontSize: 13,
              fontWeight: 500,
              color: "rgba(237,236,234,0.8)",
              cursor: "pointer",
              fontFamily: "inherit",
              marginRight: "auto",
            }}
          >
            Reset
          </button>
          <button
            onClick={handleSave}
            disabled={!isDirty || !cogniInstance || saving}
            style={{
              height: 36,
              padding: "0 16px",
              borderRadius: 8,
              border: "none",
              background: !isDirty || !cogniInstance || saving ? "rgba(255,255,255,0.08)" : "#6510F4",
              fontSize: 13,
              fontWeight: 500,
              color: !isDirty || !cogniInstance || saving ? "rgba(237,236,234,0.35)" : "#fff",
              cursor: !isDirty || !cogniInstance || saving ? "default" : "pointer",
              fontFamily: "inherit",
              transition: "background 0.15s",
            }}
          >
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    </div>,
    portalTarget,
  );
}
