import { IFrameView } from '@/ui';

interface RawDataPreviewProps {
  fileName: string;
  rawData: ArrayBuffer;
  onClose: () => void;
}

const file_header = ';headers=filename%3D';

export default function RawDataPreview({ fileName, rawData, onClose }: RawDataPreviewProps) {
  const src = `data:application/pdf;base64,${arrayBufferToBase64(rawData)}`.replace(';', file_header + encodeURIComponent(fileName) + ';');
  
  return (
    <IFrameView
      src={src}
      title={fileName}
      onClose={onClose}
    />
  );
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;

  for (var i = 0; i < len; i++) {
      binary += String.fromCharCode( bytes[ i ] );
  }

  return window.btoa(binary);
}

