import { useCallback, useEffect, useState } from 'react';
import classNames from 'classnames';
import { Spacer, Stack } from 'ohmy-ui';
import { getExplorationGraphUrl } from '@/modules/exploration';
import { IFrameView, SearchView } from '@/ui/Partials';
import { LoadingIndicator } from '@/ui/App';
import styles from './Explorer.module.css';

interface ExplorerProps {
  dataset: { id: string };
  className?: string;
  style?: React.CSSProperties;
}

export default function Explorer({ dataset, className, style }: ExplorerProps) {
  const [graphUrl, setGraphUrl] = useState<string | null>(null);

  const exploreData = useCallback(() => {
    getExplorationGraphUrl(dataset)
      .then((graphUrl) => {
        setGraphUrl(graphUrl);
      });
  }, [dataset]);
  
  useEffect(() => {
    exploreData();
  }, [exploreData]);

  return (
    <Stack
      gap="6"
      style={style}
      orientation="horizontal"
      className={classNames(styles.explorerContent, className)}
    >
      <div className={styles.graphExplorer}>
        {!graphUrl ? (
          <Spacer horizontal="2" wrap>
            <LoadingIndicator />
          </Spacer>
        ) : (
          <IFrameView src={graphUrl} />
        )}
      </div>
      <div className={styles.chat}>
        <SearchView />
      </div>
    </Stack>
  )
}
