"use client";

import type { ReactElement } from "react";
import { Menu, Tooltip } from "@mantine/core";
import type { GraphModel } from "@/modules/graphModels/types";
import type { CustomPromptsMap } from "@/modules/configuration/userConfiguration";
import type { OntologyMeta } from "@/modules/ontologies/ontologyApi";

function InfoIcon(): ReactElement {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}>
      <circle cx="8" cy="8" r="7" stroke="#A1A1AA" strokeWidth="1.5" />
      <text x="8" y="12" textAnchor="middle" fontSize="10" fontWeight="700" fill="#A1A1AA">i</text>
    </svg>
  );
}

const triggerStyle = { background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", color: "rgba(237,236,234,0.7)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6, whiteSpace: "nowrap" } as const;
const dropdownStyle = { background: "#1a1a1a", border: "1px solid rgba(255,255,255,0.08)", backdropFilter: "blur(16px)", borderRadius: 8, boxShadow: "0 8px 24px rgba(0,0,0,0.4)" } as const;
const labelStyle = { fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: 0.2, display: "flex", alignItems: "center", gap: 4 } as const;
const itemStyle = { fontSize: 13, color: "#EDECEA" } as const;
const chevron = <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M2.5 4L5 6.5L7.5 4" stroke="rgba(237,236,234,0.55)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>;
const checkOrBlank = (checked: boolean): ReactElement => <span style={{ width: 16, textAlign: "center", fontSize: 13, color: "#BC9BFF", flexShrink: 0 }}>{checked ? "✓" : ""}</span>;

const editIcon = <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" /></svg>;
const trashIcon = <svg width="14" height="14" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}><path d="M3 4h10M6 4V3h4v1M5 4v8.5a.5.5 0 00.5.5h5a.5.5 0 00.5-.5V4" stroke="#EF4444" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>;

export default function MemoryCustomizationBar({
  graphModels,
  selectedModelId,
  onSelectModel,
  onEditModel,
  onCreateModel,
  customPrompts,
  selectedPromptName,
  onSelectPrompt,
  onEditPrompt,
  onCreatePrompt,
  ontologies,
  selectedOntologyKey,
  onSelectOntology,
  onDeleteOntology,
  onUploadOntology,
}: {
  graphModels: GraphModel[];
  selectedModelId: string | null;
  onSelectModel: (id: string | null) => void;
  onEditModel: (id: string) => void;
  onCreateModel: () => void;
  customPrompts: CustomPromptsMap;
  selectedPromptName: string | null;
  onSelectPrompt: (name: string | null) => void;
  onEditPrompt: (name: string, text: string) => void;
  onCreatePrompt: () => void;
  ontologies: Record<string, OntologyMeta>;
  selectedOntologyKey: string | null;
  onSelectOntology: (key: string | null) => void;
  onDeleteOntology: (key: string) => void;
  onUploadOntology: () => void;
}): ReactElement {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <span style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.35)", letterSpacing: 0.3, textTransform: "uppercase" }}>Memory customization</span>
      <div style={{ display: "flex", gap: 16 }}>
        {/* Graph model dropdown */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={labelStyle}>Graph Model <Tooltip label="Define entity types and relationships to control how Cognee structures your knowledge graph." withArrow multiline w={240} position="top"><span style={{ display: "inline-flex" }}><InfoIcon /></span></Tooltip></span>
          <Menu shadow="md" width={220} position="bottom-start" withinPortal>
            <Menu.Target>
              <button className="cursor-pointer hover:bg-white/10" style={triggerStyle}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="6" r="3" /><circle cx="18" cy="6" r="3" /><circle cx="12" cy="18" r="3" /><line x1="8.5" y1="7.5" x2="10.5" y2="16" /><line x1="15.5" y1="7.5" x2="13.5" y2="16" /></svg>
                {selectedModelId ? (graphModels.find((m) => m.id === selectedModelId)?.name ?? "Automatic") : "Automatic"}
                {chevron}
              </button>
            </Menu.Target>
            <Menu.Dropdown style={dropdownStyle}>
              <Menu.Label style={labelStyle}>Graph Model</Menu.Label>
              <Menu.Item style={itemStyle} onClick={() => onSelectModel(null)} leftSection={checkOrBlank(selectedModelId === null)} rightSection={<span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)" }}>Default</span>}>
                Automatic
              </Menu.Item>
              {graphModels.length > 0 && <Menu.Divider style={{ borderColor: "rgba(255,255,255,0.08)" }} />}
              {graphModels.map((model) => (
                <Menu.Item
                  key={model.id}
                  style={itemStyle}
                  onClick={() => onSelectModel(model.id)}
                  leftSection={checkOrBlank(selectedModelId === model.id)}
                  rightSection={
                    <button onClick={(e) => { e.stopPropagation(); onEditModel(model.id); }} className="cursor-pointer hover:opacity-100" style={{ background: "none", border: "none", padding: 2, opacity: 0.4, transition: "opacity 150ms" }} title="Edit model">{editIcon}</button>
                  }
                >
                  {model.name}
                </Menu.Item>
              ))}
              <Menu.Divider style={{ borderColor: "rgba(255,255,255,0.08)" }} />
              <Menu.Item style={{ ...itemStyle, color: "#6510F4", fontWeight: 500 }} onClick={onCreateModel} leftSection={<span style={{ width: 16, textAlign: "center" }}>+</span>}>
                Create new
              </Menu.Item>
            </Menu.Dropdown>
          </Menu>
        </div>

        {/* Custom prompt dropdown */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={labelStyle}>Prompt <Tooltip label="Custom instructions that guide how Cognee extracts entities and relationships from your data." withArrow multiline w={240} position="top"><span style={{ display: "inline-flex" }}><InfoIcon /></span></Tooltip></span>
          <Menu shadow="md" width={220} position="bottom-start" withinPortal>
            <Menu.Target>
              <button className="cursor-pointer hover:bg-white/10" style={triggerStyle}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" /></svg>
                {selectedPromptName ?? "Automatic"}
                {chevron}
              </button>
            </Menu.Target>
            <Menu.Dropdown style={dropdownStyle}>
              <Menu.Label style={labelStyle}>Custom Prompt</Menu.Label>
              <Menu.Item style={itemStyle} onClick={() => onSelectPrompt(null)} leftSection={checkOrBlank(selectedPromptName === null)} rightSection={<span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)" }}>Default</span>}>
                Automatic
              </Menu.Item>
              {Object.keys(customPrompts).length > 0 && <Menu.Divider style={{ borderColor: "rgba(255,255,255,0.08)" }} />}
              {Object.entries(customPrompts).map(([name, text]) => (
                <Menu.Item
                  key={name}
                  style={itemStyle}
                  onClick={() => onSelectPrompt(name)}
                  leftSection={checkOrBlank(selectedPromptName === name)}
                  rightSection={
                    <button onClick={(e) => { e.stopPropagation(); onEditPrompt(name, text); }} className="cursor-pointer hover:opacity-100" style={{ background: "none", border: "none", padding: 2, opacity: 0.4, transition: "opacity 150ms" }} title="Edit prompt">{editIcon}</button>
                  }
                >
                  {name}
                </Menu.Item>
              ))}
              <Menu.Divider style={{ borderColor: "rgba(255,255,255,0.08)" }} />
              <Menu.Item style={{ ...itemStyle, color: "#6510F4", fontWeight: 500 }} onClick={onCreatePrompt} leftSection={<span style={{ width: 16, textAlign: "center" }}>+</span>}>
                Create new
              </Menu.Item>
            </Menu.Dropdown>
          </Menu>
        </div>

        {/* Ontology dropdown */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={labelStyle}>Ontology <Tooltip label="Upload a formal ontology (OWL/RDF) to enforce domain-specific vocabulary and relationships in your knowledge graph." withArrow multiline w={240} position="top"><span style={{ display: "inline-flex" }}><InfoIcon /></span></Tooltip></span>
          <Menu shadow="md" width={260} position="bottom-start" withinPortal>
            <Menu.Target>
              <button className="cursor-pointer hover:bg-white/10" style={triggerStyle}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5A2.5 2.5 0 016.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z" /></svg>
                {selectedOntologyKey ?? "Automatic"}
                {chevron}
              </button>
            </Menu.Target>
            <Menu.Dropdown style={dropdownStyle}>
              <Menu.Label style={labelStyle}>Ontology</Menu.Label>
              <Menu.Item style={itemStyle} onClick={() => onSelectOntology(null)} leftSection={checkOrBlank(selectedOntologyKey === null)}>
                Automatic
              </Menu.Item>
              {Object.keys(ontologies).length > 0 && <Menu.Divider style={{ borderColor: "rgba(255,255,255,0.08)" }} />}
              {Object.entries(ontologies).map(([key, meta]) => (
                <Menu.Item
                  key={key}
                  style={itemStyle}
                  onClick={() => onSelectOntology(key)}
                  leftSection={checkOrBlank(selectedOntologyKey === key)}
                  rightSection={
                    <button onClick={(e) => { e.stopPropagation(); onDeleteOntology(key); }} className="cursor-pointer hover:opacity-100" style={{ background: "none", border: "none", padding: 4, opacity: 0.5, transition: "opacity 150ms", minWidth: 20, minHeight: 20, display: "flex", alignItems: "center", justifyContent: "center" }} title="Delete ontology">{trashIcon}</button>
                  }
                >
                  <span title={meta.filename} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "block" }}>{meta.filename}</span>
                </Menu.Item>
              ))}
              <Menu.Divider style={{ borderColor: "rgba(255,255,255,0.08)" }} />
              <Menu.Item style={{ ...itemStyle, color: "#6510F4", fontWeight: 500 }} onClick={onUploadOntology} leftSection={<span style={{ width: 16, textAlign: "center" }}>+</span>}>
                Upload new
              </Menu.Item>
            </Menu.Dropdown>
          </Menu>
        </div>
      </div>
    </div>
  );
}
