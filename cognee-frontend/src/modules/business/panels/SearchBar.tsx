"use client";

import { useState } from "react";
import type { CogneeInstance } from "@/modules/instances/types";
import searchDataset from "@/modules/datasets/searchDataset";
import type { BusinessEntity } from "../sceneTypes";
import { truncate } from "../textUtils";
import { captureException } from "@/utils/monitoring";

const MAX_MATCHES = 5;

interface SearchBarProps {
  cogniInstance: CogneeInstance;
  activeDatasetId: string | null;
  onAnswer: (question: string, answer: string) => void;
  entities: BusinessEntity[] | null;
  onPickEntity: (entity: BusinessEntity) => void;
}

// One bar for both "find a record on the map" and "ask the AI a question" —
// two separate inputs read as clutter, and the two intents disambiguate
// themselves: a short entity-ish query substring-matches record names (rows
// to jump to), a full question matches nothing (Enter falls through to a
// real /v1/search ask). The dropdown's last row always offers the ask
// explicitly, for the rare query that's both. /v1/search returns no
// node_ids (verified against the running backend), so an asked answer shows
// text only — the spotlight arrives later via the live-events replay.
export default function SearchBar({ cogniInstance, activeDatasetId, onAnswer, entities, onPickEntity }: SearchBarProps) {
  const [query, setQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const needle = query.trim().toLowerCase();
  // Prefix hits outrank substring hits, importance breaks ties — otherwise
  // "tom" surfaces five incidental "…custom…" records while tom becker,
  // the record anyone typing that means, sits past the cutoff.
  const matches = needle
    ? (entities ?? [])
        .filter((e) => e.name && !e.is_unnamed && String(e.name).toLowerCase().includes(needle))
        .sort((a, b) => {
          const ap = String(a.name).toLowerCase().startsWith(needle) ? 1 : 0;
          const bp = String(b.name).toLowerCase().startsWith(needle) ? 1 : 0;
          if (ap !== bp) return bp - ap;
          return (b.importance || 0) - (a.importance || 0);
        })
        .slice(0, MAX_MATCHES)
    : [];

  const pickEntity = (entity: BusinessEntity): void => {
    onPickEntity(entity);
    setQuery("");
  };

  const ask = async (): Promise<void> => {
    const trimmed = query.trim();
    if (!trimmed || !activeDatasetId || isSearching) return;
    setIsSearching(true);
    try {
      const results = await searchDataset(cogniInstance, {
        query: trimmed,
        datasetIds: [activeDatasetId],
        searchType: "GRAPH_COMPLETION",
      });
      const answer = results[0]?.search_result?.join("\n\n") || "no answer found.";
      onAnswer(trimmed, answer);
      setQuery("");
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      console.error("Business search failed:", error);
      captureException(error, { context: "BusinessView search" });
      onAnswer(trimmed, "search failed — try again.");
    } finally {
      setIsSearching(false);
    }
  };

  return (
    // top-[29px] lines the input up with the BrainSwitcher chip and the
    // operators' workspace card: both sit under a 10px-offset label with a
    // leading-[15px] line box and a 4px gap (10+15+4 — see BusinessView's
    // brain label comment).
    <div className="absolute left-1/2 top-[29px] z-10 w-[360px] -translate-x-1/2">
      <div className="flex items-center gap-1.5 rounded-[10px] border border-[#2A3652] bg-[#1A2438] px-2.5 py-1.5">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              if (matches[0]) pickEntity(matches[0]);
              else void ask();
            }
            if (e.key === "Escape") setQuery("");
          }}
          placeholder="find a record or ask anything…"
          disabled={isSearching || !activeDatasetId}
          className="flex-1 bg-transparent text-[#E9EEF6] placeholder:text-[#7E8CA6] focus:outline-none"
        />
        <button
          type="button"
          onClick={() => void ask()}
          disabled={isSearching || !query.trim() || !activeDatasetId}
          className="text-[#43D9E8] disabled:text-[#7E8CA6]"
        >
          {isSearching ? "…" : "ask"}
        </button>
      </div>
      {needle && isSearching && (
        // The button's "…" alone was easy to miss — especially the common
        // path where Enter (not a click) triggered the ask, so the user's
        // attention was never on that small a target to begin with. This
        // sits where the results dropdown would be, the one place the user
        // is actually looking right after submitting.
        <div className="mt-1 overflow-hidden rounded-[10px] border border-[#2A3652] bg-[#1A2438] px-2.5 py-1.5 text-[11px] text-[#7E8CA6]">
          searching…
        </div>
      )}
      {needle && !isSearching && (
        <div className="mt-1 overflow-hidden rounded-[10px] border border-[#2A3652] bg-[#1A2438]">
          {matches.map((e) => (
            <button
              key={e.id}
              type="button"
              onClick={() => pickEntity(e)}
              className="block w-full truncate px-2.5 py-1.5 text-left text-[#E9EEF6] hover:bg-[#141D33]"
            >
              ⌖ {String(e.name)}
            </button>
          ))}
          <button
            type="button"
            onClick={() => void ask()}
            className="block w-full truncate px-2.5 py-1.5 text-left text-[#43D9E8] hover:bg-[#141D33]"
          >
            ⌕ ask: &ldquo;{truncate(query.trim(), 40)}&rdquo;
          </button>
        </div>
      )}
    </div>
  );
}
