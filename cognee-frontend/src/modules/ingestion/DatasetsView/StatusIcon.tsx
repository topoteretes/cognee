export default function StatusIcon({ status }: { status: 'DATASET_PROCESSING_COMPLETED' | string }) {
  const isSuccess = status === 'DATASET_PROCESSING_COMPLETED';

  return (
    <div
      style={{
        width: '16px',
        height: '16px',
        borderRadius: '4px',
        background: isSuccess ? '#53ff24' : '#ff5024',
      }}
      title={isSuccess ? 'Dataset cognified' : 'Cognify data in order to explore it'}
    />
  );
}