import { CloseIcon, GhostButton, Spacer, Stack, Text } from 'ohmy-ui';
import styles from './IFrameView.module.css';

interface IFrameViewProps {
  src: string;
  title: string;
  onClose: () => void;
}

export default function IFrameView({ title, src, onClose }: IFrameViewProps) {
  return (
    <div className={styles.iFrameViewContainer}>
      <Stack gap="between" align="center/" orientation="horizontal">
        <Spacer horizontal="2">
          <Text>{title}</Text>
        </Spacer>
        <GhostButton onClick={onClose}>
          <CloseIcon />
        </GhostButton>
      </Stack>
      <iframe
        src={src}
        width="100%"
        height="100%"
      />
    </div>
  );
}
