"use client";

import { ChangeEvent, useCallback, useEffect, useState } from "react";
import { useBoolean } from "@/utils";
import { Accordion, CTAButton, GhostButton, IconButton, Input, Modal, PopupMenu } from "@/ui/elements";
import { AccordionProps } from "@/ui/elements/Accordion";
import { CloseIcon, DatasetIcon, MinusIcon, PlusIcon } from "@/ui/Icons";
import useDatasets, { Dataset } from "@/modules/ingestion/useDatasets";
import addData from "@/modules/ingestion/addData";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import { DataFile } from "@/modules/ingestion/useData";
import { LoadingIndicator } from "@/ui/App";

interface DatasetsChangePayload {
  datasets: Dataset[]
  refreshDatasets: () => void;
}

export interface DatasetsAccordionProps extends Omit<AccordionProps, "isOpen" | "openAccordion" | "closeAccordion" | "children"> {
  onDatasetsChange?: (payload: DatasetsChangePayload) => void;
  useCloud?: boolean;
}

export default function DatasetsAccordion({
  title,
  tools,
  switchCaretPosition = false,
  className,
  contentClassName,
  onDatasetsChange,
  useCloud = false,
}: DatasetsAccordionProps) {
  const {
    value: isDatasetsPanelOpen,
    setTrue: openDatasetsPanel,
    setFalse: closeDatasetsPanel,
  } = useBoolean(true);

  const {
    datasets,
    refreshDatasets,
    addDataset,
    removeDataset,
    getDatasetData,
    removeDatasetData,
  } = useDatasets(useCloud);

  useEffect(() => {
    if (datasets.length === 0) {
      refreshDatasets();
    }
  }, [datasets.length, refreshDatasets]);

  const [openDatasets, openDataset] = useState<Set<string>>(new Set());

  const toggleDataset = (id: string) => {
    openDataset((prev) => {
      const newState = new Set(prev);

      if (newState.has(id)) {
        newState.delete(id)
      } else {
        getDatasetData(id)
          .then(() => {
            newState.add(id);
          });
      }

      return newState;
    });
  };

  const refreshOpenDatasetsData = useCallback(() => {
    return Promise.all(
      openDatasets.values().map(
        (datasetId) => getDatasetData(datasetId)
      )
    );
  }, [getDatasetData, openDatasets]);

  const refreshDatasetsAndData = useCallback(() => {
    refreshDatasets()
     .then(refreshOpenDatasetsData);
  }, [refreshDatasets, refreshOpenDatasetsData]);

  useEffect(() => {
    onDatasetsChange?.({
      datasets,
      refreshDatasets: refreshDatasetsAndData,
    });
  }, [datasets, onDatasetsChange, refreshDatasets, refreshDatasetsAndData]);

  const {
    value: isNewDatasetModalOpen,
    setTrue: openNewDatasetModal,
    setFalse: closeNewDatasetModal,
  } = useBoolean(false);

  const handleDatasetAdd = () => {
    openNewDatasetModal();
  };

  const [newDatasetError, setNewDatasetError] = useState("");

  const handleNewDatasetSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setNewDatasetError("");

    const formElements = event.currentTarget;

    const datasetName = formElements.datasetName.value;

    if (datasetName.trim().length === 0) {
      setNewDatasetError("Dataset name cannot be empty.");
      return;
    }

    if (datasetName.includes(" ") || datasetName.includes(".")) {
      setNewDatasetError("Dataset name cannot contain spaces or periods.");
      return;
    }

    addDataset(datasetName)
      .then(() => {
        closeNewDatasetModal();
        refreshDatasetsAndData();
      });
  };

  const {
    value: isRemoveDatasetModalOpen,
    setTrue: openRemoveDatasetModal,
    setFalse: closeRemoveDatasetModal,
  } = useBoolean(false);

  const [datasetToRemove, setDatasetToRemove] = useState<Dataset | null>(null);

  const handleDatasetRemove = (dataset: Dataset) => {
    setDatasetToRemove(dataset);
    openRemoveDatasetModal();
  };

  const handleDatasetRemoveCancel = () => {
    setDatasetToRemove(null);
    closeRemoveDatasetModal();
  };

  const handleRemoveDatasetConfirm = (event: React.FormEvent<HTMLButtonElement>) => {
    event.preventDefault();

    if (datasetToRemove) {
      removeDataset(datasetToRemove.id)
        .then(() => {
          closeRemoveDatasetModal();
          setDatasetToRemove(null);
          refreshDatasetsAndData();
        });
    }
  };

  const [datasetInProcessing, setProcessingDataset] = useState<Dataset | null>(null);

  const handleAddFiles = (dataset: Dataset, event: ChangeEvent<HTMLInputElement>) => {
    event.stopPropagation();

    if (datasetInProcessing) {
      return;
    }

    setProcessingDataset(dataset);

    if (!event.target.files) {
      return;
    }

    const files: File[] = Array.from(event.target.files);

    if (!files.length) {
      return;
    }

    return addData(dataset, files, useCloud)
      .then(async () => {
        await getDatasetData(dataset.id);

        return cognifyDataset(dataset, useCloud)
          .finally(() => {
            setProcessingDataset(null);
          });
      });
  };

  const [dataToRemove, setDataToRemove] = useState<DataFile | null>(null);
  const {
    value: isRemoveDataModalOpen,
    setTrue: openRemoveDataModal,
    setFalse: closeRemoveDataModal,
  } = useBoolean(false);

  const handleDataRemove = (data: DataFile) => {
    setDataToRemove(data);

    openRemoveDataModal();
  };
  const handleDataRemoveCancel = () => {
    setDataToRemove(null);
    closeRemoveDataModal();
  };
  const handleDataRemoveConfirm = (event: React.FormEvent<HTMLButtonElement>) => {
    event.preventDefault();

    if (dataToRemove) {
      removeDatasetData(dataToRemove.datasetId, dataToRemove.id)
         .then(() => {
          closeRemoveDataModal();
          setDataToRemove(null);
          refreshDatasetsAndData();
        });
    }
  }

  return (
    <>
      <Accordion
        title={title || <span>Datasets</span>}
        isOpen={isDatasetsPanelOpen}
        openAccordion={openDatasetsPanel}
        closeAccordion={closeDatasetsPanel}
        tools={(
          <div className="flex flex-row gap-4 items-center">
            {tools}
            <IconButton onClick={handleDatasetAdd}><PlusIcon /></IconButton>
          </div>
        )}
        switchCaretPosition={switchCaretPosition}
        className={className}
        contentClassName={contentClassName}
      >
        <div className="flex flex-col">
          {datasets.length === 0 && (
            <div className="flex flex-row items-baseline-last text-sm text-gray-400 mt-2 px-2">
              <span>No datasets here, add one by clicking +</span>
            </div>
          )}
          {datasets.map((dataset) => {
            return (
              <Accordion
                key={dataset.id}
                title={(
                  <div className="flex flex-row gap-2 items-center py-1.5 cursor-pointer">
                    {datasetInProcessing?.id == dataset.id ? <LoadingIndicator /> : <DatasetIcon />}
                    <span className="text-xs">{dataset.name}</span>
                  </div>
                )}
                isOpen={openDatasets.has(dataset.id)}
                openAccordion={() => toggleDataset(dataset.id)}
                closeAccordion={() => toggleDataset(dataset.id)}
                tools={(
                  <IconButton className="relative">
                    <PopupMenu>
                      <div className="flex flex-col gap-0.5">
                        <div className="hover:bg-gray-100 w-full text-left px-2 cursor-pointer relative">
                          <input tabIndex={-1} type="file" multiple onChange={handleAddFiles.bind(null, dataset)} className="absolute w-full h-full cursor-pointer opacity-0" />
                          <span>add data</span>
                        </div>
                      </div>
                      <div className="flex flex-col gap-0.5 items-start">
                        <div onClick={() => handleDatasetRemove(dataset)} className="hover:bg-gray-100 w-full text-left px-2 cursor-pointer">delete</div>
                      </div>
                    </PopupMenu>
                  </IconButton>
                )}
                className="first:pt-1.5"
                switchCaretPosition={true}
              >
                <>
                  {dataset.data?.length === 0 && (
                    <div className="flex flex-row items-baseline-last text-sm text-gray-400 mt-2 px-2">
                      <span>No data in a dataset, add by clicking &quot;add data&quot; in a dropdown menu</span>
                    </div>
                  )}
                  {dataset.data?.map((data) => (
                    <div key={data.id} className="flex flex-row gap-2 items-center justify-between py-1.5 pl-6 last:pb-2.5">
                      <span className="text-xs">{data.name}</span>
                      <div>
                        <IconButton onClick={() => handleDataRemove(data)}><MinusIcon /></IconButton>
                      </div>
                    </div>
                  ))}
                </>
              </Accordion>
            );
          })}
        </div>
      </Accordion>

      <Modal isOpen={isNewDatasetModalOpen}>
        <div className="w-full max-w-2xl">
          <div className="flex flex-row items-center justify-between">
            <span className="text-2xl">Create a new dataset?</span>
            <IconButton onClick={closeNewDatasetModal}><CloseIcon /></IconButton>
          </div>
          <div className="mt-8 mb-6">Please provide a name for the dataset being created.</div>
          <form onSubmit={handleNewDatasetSubmit}>
            <div className="max-w-md">
              <Input name="datasetName" type="text" placeholder="Dataset name" required />
              {newDatasetError && <span className="text-sm pl-4 text-gray-400">{newDatasetError}</span>}
            </div>
            <div className="flex flex-row gap-4 mt-4 justify-end">
              <GhostButton type="button" onClick={() => closeNewDatasetModal()}>cancel</GhostButton>
              <CTAButton type="submit">create</CTAButton>
            </div>
          </form>
        </div>
      </Modal>

      <Modal isOpen={isRemoveDatasetModalOpen}>
        <div className="w-full max-w-2xl">
          <div className="flex flex-row items-center justify-between">
            <span className="text-2xl">Delete <span className="text-indigo-600">{datasetToRemove?.name}</span> dataset?</span>
            <IconButton onClick={handleDatasetRemoveCancel}><CloseIcon /></IconButton>
          </div>
          <div className="mt-8 mb-6">Are you sure you want to delete <span className="text-indigo-600">{datasetToRemove?.name}</span>? This action cannot be undone.</div>
          <div className="flex flex-row gap-4 mt-4 justify-end">
            <GhostButton type="button" onClick={handleDatasetRemoveCancel}>cancel</GhostButton>
            <CTAButton onClick={handleRemoveDatasetConfirm} type="submit">delete</CTAButton>
          </div>
        </div>
      </Modal>

      <Modal isOpen={isRemoveDataModalOpen}>
        <div className="w-full max-w-2xl">
          <div className="flex flex-row items-center justify-between">
            <span className="text-2xl">Delete <span className="text-indigo-600">{dataToRemove?.name}</span> data?</span>
            <IconButton onClick={handleDataRemoveCancel}><CloseIcon /></IconButton>
          </div>
          <div className="mt-8 mb-6">Are you sure you want to delete <span className="text-indigo-600">{dataToRemove?.name}</span>? This action cannot be undone.</div>
          <div className="flex flex-row gap-4 mt-4 justify-end">
            <GhostButton type="button" onClick={handleDataRemoveCancel}>cancel</GhostButton>
            <CTAButton onClick={handleDataRemoveConfirm} type="submit">delete</CTAButton>
          </div>
        </div>
      </Modal>
    </>
  );
}
