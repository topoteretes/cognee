"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { CogneeInstance } from "../instances/types";
import getDatasetData, { DEFAULT_DATASET_DATA_LIMIT } from "./getDatasetData";

/** Load document pages on demand; discard responses superseded by a selection or refresh. */
export default function useDatasetDataPages<T extends { id: string }>(instance: CogneeInstance | null) {
  const [data, setData] = useState<T[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const request = useRef(0);
  const active = useRef(false);
  const dataset = useRef<string | null>(null);
  const rows = useRef(data);
  rows.current = data;

  const invalidate = useCallback(() => {
    request.current++;
    active.current = false;
  }, []);
  const reset = useCallback(() => {
    invalidate();
    dataset.current = null;
    rows.current = [];
    setData([]);
    setLoading(false);
    setError(false);
    setHasMore(false);
  }, [invalidate]);
  useEffect(() => {
    reset();
    return invalidate;
  }, [instance, invalidate, reset]);

  const load = useCallback(async (id: string, append = false): Promise<void> => {
    if (!instance || (append && active.current)) return;
    const token = ++request.current;
    const offset = append ? rows.current.length : 0;
    const previous = append ? rows.current : [];
    const changedDataset = dataset.current !== id;
    dataset.current = id;
    active.current = true;
    setLoading(true);
    setError(false);
    if (!append && changedDataset) {
      setData([]);
      setHasMore(false);
    }
    try {
      const page: T[] = await getDatasetData(id, instance, { offset });
      if (request.current !== token) return;
      const seen = new Set(previous.map(row => row.id));
      setData([...previous, ...page.filter(item => !seen.has(item.id))]);
      setHasMore(page.length === DEFAULT_DATASET_DATA_LIMIT);
    } catch {
      if (request.current === token) setError(true);
    } finally {
      if (request.current === token) {
        active.current = false;
        setLoading(false);
      }
    }
  }, [instance]);

  return {
    data, setData, loading, error, hasMore, load, reset,
    loadMore: () => { if (dataset.current) void load(dataset.current, true); },
  };
}
