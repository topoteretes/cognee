import { tokens } from "@/ui/theme/tokens";

export default function StatusIndicator({ status }: { status: "DATASET_PROCESSING_COMPLETED" | string }) {
  const statusColor = {
    DATASET_PROCESSING_STARTED: tokens.statusProcessing,
    DATASET_PROCESSING_INITIATED: tokens.statusProcessing,
    DATASET_PROCESSING_COMPLETED: tokens.statusSuccess,
    DATASET_PROCESSING_ERRORED: tokens.statusError,
  };

  const isSuccess = status === "DATASET_PROCESSING_COMPLETED";

  return (
    <div
      style={{
        width: "16px",
        height: "16px",
        borderRadius: "4px",
        background: statusColor[status as keyof typeof statusColor],
      }}
      title={isSuccess ? "Dataset cognified" : "Cognify data in order to explore it"}
    />
  );
}
