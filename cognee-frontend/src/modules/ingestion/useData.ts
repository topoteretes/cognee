import { v4 } from 'uuid';
import { useCallback, useState } from 'react';

export interface DataFile {
  id: string;
  name: string;
  file: File;
  datasetId: string;
}

const useData = () => {
  const [data, setNewData] = useState<DataFile[]>([]);

  const addData = useCallback((files: File[]) => {
    setNewData(
      files.map((file) => ({
        id: v4(),
        name: file.name,
        file,
        datasetId: "",
      }))
    );
  }, []);

  const removeData = useCallback((dataToRemove: DataFile) => {
    setNewData((data) =>
      data ? data.filter((data) => data.file !== dataToRemove.file) : []
    );
  }, []);

  return { data, addData, removeData };
};

export default useData;
