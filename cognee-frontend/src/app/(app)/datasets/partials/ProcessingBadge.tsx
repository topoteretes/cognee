export default function ProcessingBadge({ completed }: { completed?: boolean }) {
  if (completed === undefined) return null;
  return <span style={{ fontSize: 11, whiteSpace: "nowrap", color: completed ? "#86EFAC" : "#FCD34D" }} title={completed ? "Processed and ready to search" : "Not yet processed successfully; may be waiting, processing, or affected by a failed run"}>
    {completed ? "Ready" : "Not ready"}
  </span>;
}
