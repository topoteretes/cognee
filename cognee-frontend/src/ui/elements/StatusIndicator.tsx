export default function StatusIndicator({ status }: { status: "DATASET_PROCESSING_COMPLETED" | string }) {
  const statusColor = {
    DATASET_PROCESSING_STARTED: "#ffd500",
    DATASET_PROCESSING_INITIATED: "#ffd500",
    DATASET_PROCESSING_COMPLETED: "#53ff24",
    DATASET_PROCESSING_ERRORED: "#ff5024",
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
