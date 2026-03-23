/**
 * Shared helpers for document status display and formatting.
 * Used by DatasetsAccordion and DocumentDetailModal.
 */

export function getDocStatus(
  pipeline_status?: Record<string, Record<string, string>>
): "completed" | "processing" | "pending" {
  if (!pipeline_status || Object.keys(pipeline_status).length === 0)
    return "pending";
  const values = Object.values(pipeline_status).flatMap((v) =>
    Object.values(v)
  );
  if (values.every((s) => s === "DATA_ITEM_PROCESSING_COMPLETED"))
    return "completed";
  return "processing";
}

export function formatSize(bytes?: number): string {
  if (!bytes || bytes <= 0) return "";
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

export function formatTokens(count?: number): string {
  if (count === undefined || count === null || count < 0) return "";
  return count.toLocaleString() + " tok";
}
