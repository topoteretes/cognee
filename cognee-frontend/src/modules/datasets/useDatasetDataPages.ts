"use client";

import { type SetStateAction, useCallback, useEffect, useRef, useState } from "react";
import type { CogneeInstance } from "../instances/types";
import getDatasetData, { DEFAULT_DATASET_DATA_LIMIT, getDatasetDataCount } from "./getDatasetData";

/** Load bounded pages; discard responses superseded by a selection, refresh, or mutation. */
export default function useDatasetDataPages<T extends { id: string }>(
  instance: CogneeInstance | null,
  maxRows = Infinity,
  includeTotal = false,
) {
  const [data, updateData] = useState<T[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const generation = useRef(0);
  const active = useRef(false);
  const failedAppend = useRef(true);
  const more = useRef(false);
  const nextOffset = useRef(0);
  const dataset = useRef<string | null>(null);
  const rows = useRef<T[]>([]);
  const controller = useRef<AbortController | null>(null);

  const invalidate = useCallback(() => {
    generation.current++;
    controller.current?.abort();
    active.current = false;
  }, []);
  const reset = useCallback(() => {
    invalidate();
    dataset.current = null;
    rows.current = [];
    nextOffset.current = 0;
    more.current = false;
    updateData([]);
    setTotal(null);
    setLoading(false);
    setError(false);
    setHasMore(false);
  }, [invalidate]);
  useEffect(() => {
    reset();
    return invalidate;
  }, [instance, maxRows, invalidate, reset]);

  // Local deletion shifts the server's offset, but duplicate rows in a page
  // do not. Keep the consumed offset separate from the rendered row count.
  const setData = useCallback((action: SetStateAction<T[]>) => {
    const previous = rows.current;
    const next = (typeof action === "function" ? action(previous) : action).slice(0, maxRows);
    const keptIds = new Set(next.map(row => row.id));
    const removed = previous.filter(row => !keptIds.has(row.id)).length;
    invalidate();
    nextOffset.current = Math.max(0, nextOffset.current - removed);
    rows.current = next;
    updateData(next);
    setTotal(value => value === null ? null : Math.max(0, value - removed));
    setLoading(false);
  }, [invalidate, maxRows]);

  const load = useCallback(async (id: string, append = false): Promise<void> => {
    if (!instance || (append && (active.current || !more.current || rows.current.length >= maxRows))) return;
    if (!append) invalidate();
    if (!append || !controller.current || controller.current.signal.aborted) {
      controller.current = new AbortController();
    }
    const token = generation.current;
    const offset = append ? nextOffset.current : 0;
    const previous = append ? rows.current : [];
    const limit = Math.min(DEFAULT_DATASET_DATA_LIMIT, maxRows - previous.length);
    const changedDataset = dataset.current !== id;
    dataset.current = id;
    active.current = true;
    setLoading(true);
    setError(false);
    if (!append && changedDataset) {
      rows.current = [];
      updateData([]);
      nextOffset.current = 0;
      more.current = false;
      setHasMore(false);
      setTotal(null);
    }
    // A count failure must not prevent reading documents. Paging uses the
    // response length, so a stale count can never cause an endless retry.
    if (!append && (includeTotal || Number.isFinite(maxRows))) {
      void getDatasetDataCount(id, instance, controller.current?.signal)
        .then(count => { if (generation.current === token) setTotal(count); })
        .catch(() => { if (generation.current === token) setTotal(null); });
    }
    try {
      const page: T[] = await getDatasetData(id, instance, { offset, limit, signal: controller.current?.signal });
      if (generation.current !== token) return;
      const seen = new Set(previous.map(row => row.id));
      const next = [...previous];
      for (const item of page) {
        if (next.length >= maxRows) break;
        if (!seen.has(item.id)) {
          seen.add(item.id);
          next.push(item);
        }
      }
      rows.current = next;
      nextOffset.current = offset + page.length;
      more.current = page.length >= limit && next.length > previous.length;
      updateData(next);
      setHasMore(more.current);
    } catch {
      if (generation.current === token) {
        failedAppend.current = append;
        setError(true);
      }
    } finally {
      if (generation.current === token) {
        active.current = false;
        setLoading(false);
      }
    }
  }, [instance, maxRows, includeTotal, invalidate]);

  const loadMore = useCallback(() => {
    if (dataset.current) void load(dataset.current, error ? failedAppend.current : true);
  }, [load, error]);

  return {
    data, setData, total, loading, error, hasMore, load, reset,
    loadMore,
  };
}
