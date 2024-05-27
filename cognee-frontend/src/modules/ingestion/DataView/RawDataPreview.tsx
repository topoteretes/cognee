import { IFrameView } from '@/ui/Partials';
import { CloseIcon, GhostButton, Modal, Spacer, Stack, Text } from 'ohmy-ui';
import styles from './RawDataPreview.module.css';

interface RawDataPreviewProps {
  fileName: string;
  rawData: ArrayBuffer;
  onClose: () => void;
}

const file_header = ';headers=filename%3D';

export default function RawDataPreview({ fileName, rawData, onClose }: RawDataPreviewProps) {
  const src = `data:application/pdf;base64,${arrayBufferToBase64(rawData)}`.replace(';', file_header + encodeURIComponent(fileName) + ';');
  
  return (
    <Modal isOpen onClose={onClose} className={styles.dataPreviewModal}>
      <Spacer horizontal="2" vertical="3" wrap>
        <Text>{fileName}</Text>
      </Spacer> 
      <IFrameView src={src} />
    </Modal>
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

