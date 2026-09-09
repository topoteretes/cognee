"use client";

import { useCallback, useState } from "react";
import type { BusinessEntity, SemanticLink } from "./sceneTypes";
import { useShortestPath } from "./useShortestPath";

export interface EntitySelection {
  selectedEntity: BusinessEntity | null;
  pathIds: Set<string>;
  pathEdgeKeys: Set<string>;
  pathTargetName: string | null;
  pathHops: number | null;
  selectEntity: (entity: BusinessEntity) => void;
  // The canvas's click handler — a plain click selects (toggling off on a
  // second click on the same record); shift-click, only meaningful with an
  // existing selection, traces the shortest path to the new record instead.
  handleCanvasSelect: (entity: BusinessEntity, shiftKey: boolean) => void;
  clearSelection: () => void;
  clearPath: () => void;
}

// Bundles click-to-select with its shift-click extension (trace the
// shortest path to a second record — see useShortestPath, both new in this
// port with no source equivalent) since a plain click always resets both:
// keeping them in one hook means BusinessView never has to re-derive
// "is this a path click or a fresh selection" on its own.
export function useEntitySelection(
  semanticLinks: SemanticLink[] | undefined,
  entityById: Record<string, BusinessEntity> | undefined,
): EntitySelection {
  const [selectedEntity, setSelectedEntity] = useState<BusinessEntity | null>(null);
  const [pathTargetId, setPathTargetId] = useState<string | null>(null);
  const { pathIds, pathEdgeKeys } = useShortestPath(semanticLinks, selectedEntity?.id ?? null, pathTargetId);

  const clearPath = useCallback(() => setPathTargetId(null), []);
  const clearSelection = useCallback(() => {
    setSelectedEntity(null);
    setPathTargetId(null);
  }, []);

  const selectEntity = useCallback((entity: BusinessEntity) => {
    setSelectedEntity(entity);
    setPathTargetId(null);
  }, []);

  const handleCanvasSelect = useCallback(
    (entity: BusinessEntity, shiftKey: boolean) => {
      if (shiftKey && selectedEntity && selectedEntity.id !== entity.id) {
        setPathTargetId(entity.id);
        return;
      }
      setSelectedEntity((current) => (current?.id === entity.id ? null : entity));
      setPathTargetId(null);
    },
    [selectedEntity],
  );

  const pathTargetName = pathTargetId ? String(entityById?.[pathTargetId]?.name || "") || null : null;
  const pathHops = pathTargetName && pathIds.size ? pathIds.size - 1 : null;

  return {
    selectedEntity, pathIds, pathEdgeKeys, pathTargetName, pathHops,
    selectEntity, handleCanvasSelect, clearSelection, clearPath,
  };
}
