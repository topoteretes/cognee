import classNames from 'classnames';
import { useCallback, useEffect, useState } from 'react';
import { Spacer, Stack, Text } from 'ohmy-ui';
import { LoadingIndicator } from '@/ui/App';
import { IFrameView, SearchView } from '@/ui/Partials';
import { getExplorationGraphUrl } from '@/modules/exploration';
import styles from './Explorer.module.css';

interface ExplorerProps {
  dataset: { name: string };
  className?: string;
  style?: React.CSSProperties;
}

export default function Explorer({ dataset, className, style }: ExplorerProps) {
  const [error, setError] = useState<Error | null>(null);
  const [graphHtml, setGraphHtml] = useState<string | null>(null);

  const exploreData = useCallback(() => {
    getExplorationGraphUrl(dataset)
      .then((graphHtml) => {
        setError(null);
        setGraphHtml(graphHtml);
      })
      .catch((error) => {
        setError(error);
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
        {error ? (
          <Text color="red">{error.message}</Text>
        ) : (
          <>
            {!graphHtml ? (
              <Spacer horizontal="2" wrap>
                <LoadingIndicator />
              </Spacer>
            ) : (
              <IFrameView src="http://127.0.0.1:8000/api/v1/visualize" />
            )}
          </>
        )}
      </div>
      <div className={styles.chat}>
        <SearchView />
      </div>
    </Stack>
  )
}
