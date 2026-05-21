"use client";

import { ChangeEvent, useCallback, useEffect, useState } from "react";
import { useBoolean } from "@/utils";
import PopupMenu from "@/ui/elements/PopupMenu";
import {
  Accordion,
  CTAButton,
  GhostButton,
  IconButton,
  Modal,
} from "@/ui/elements";
import { AccordionProps } from "@/ui/elements/Accordion";
import { CloseIcon, DatasetIcon, MinusIcon, PlusIcon } from "@/ui/icons";
import useDatasets, { Dataset } from "@/modules/ingestion/useDatasets";
import addData from "@/modules/ingestion/addData";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import { DataFile } from "@/modules/ingestion/useData";
import { LoadingIndicator } from "@/ui/app";
import { CogneeInstance } from "@/modules/instances/types";
import { useModal } from "@/ui/elements/Modal";
import CreateNewDatasetModal from "./elements/CreateNewDatasetAccordion";

interface DatasetsChangePayload {
  datasets: Dataset[];
  refreshDatasets: () => void;
}

export interface DatasetsAccordionProps extends Omit<
  AccordionProps,
  "isOpen" | "openAccordion" | "closeAccordion" | "children"
> {
  onDatasetsChange?: (payload: DatasetsChangePayload) => void;
  instance: CogneeInstance;
  searchValue: string;
}

export default function DatasetsAccordion({
  title,
  tools,
  searchValue,
  switchCaretPosition = false,
  className,
  contentClassName,
  onDatasetsChange,
  instance,
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
  } = useDatasets(instance, searchValue);

  const [openDatasets, openDataset] = useState<Set<string>>(new Set());

  const toggleDataset = (id: string) => {
    openDataset((prev) => {
      const newState = new Set(prev);

      if (newState.has(id)) {
        newState.delete(id);
      } else {
        newState.add(id);
        setDatasetToProcessing({ id } as Dataset);
        getDatasetData(id).finally(() => {
          setDatasetToProcessing({ id } as Dataset, false);
        });
      }

      return newState;
    });
  };

  const refreshOpenDatasetsData = useCallback(() => {
    return Promise.all(
      openDatasets.values().map((datasetId) => getDatasetData(datasetId)),
    );
  }, [getDatasetData, openDatasets]);

  const refreshDatasetsAndData = useCallback(() => {
    refreshDatasets().then(refreshOpenDatasetsData);
  }, [refreshDatasets, refreshOpenDatasetsData]);

  useEffect(() => {
    onDatasetsChange?.({
      datasets,
      refreshDatasets: refreshDatasetsAndData,
    });
  }, [datasets, onDatasetsChange, refreshDatasets, refreshDatasetsAndData]);

  const [newDatasetError, setNewDatasetError] = useState("");

  const handleNewDatasetSubmit = useCallback(
    (_: Dataset, event?: React.FormEvent<HTMLFormElement>) => {
      event?.preventDefault();
      setNewDatasetError("");

      const formElements = event?.currentTarget;

      const datasetName = formElements?.datasetName?.value;

      if (datasetName.trim().length === 0) {
        setNewDatasetError("Dataset name cannot be empty.");
        throw new Error("Dataset name cannot be empty.");
      }

      if (datasetName.includes(" ") || datasetName.includes(".")) {
        setNewDatasetError("Dataset name cannot contain spaces or periods.");
        throw new Error("Dataset name cannot contain spaces or periods.");
      }

      return addDataset(datasetName).then(() => {
        refreshDatasetsAndData();
      });
    },
    [addDataset, refreshDatasetsAndData],
  );

  const handleRemoveDatasetConfirm = useCallback(
    (dataset: Dataset, event?: React.FormEvent<HTMLFormElement>) => {
      event?.preventDefault();

      return removeDataset(dataset.id).then(() => {
        refreshDatasetsAndData();
      });
    },
    [refreshDatasetsAndData, removeDataset],
  );

  const [datasetsInProcessing, setProcessingDatasets] = useState<{
    [key: string]: boolean;
  }>({});

  const setDatasetToProcessing = useCallback(
    (dataset: Dataset, flag: boolean = true) => {
      setProcessingDatasets((datasets) => ({
        ...datasets,
        [dataset.id]: flag,
      }));
    },
    [],
  );

  const handleAddFiles = useCallback(
    (dataset: Dataset, event: ChangeEvent<HTMLInputElement>) => {
      event.stopPropagation();

      if (datasetsInProcessing[dataset.id]) {
        return;
      }

      setDatasetToProcessing(dataset);

      if (!event.target.files) {
        return;
      }

      const files: File[] = Array.from(event.target.files);

      if (!files.length) {
        return;
      }

      return addData(dataset, files, instance).then(async () => {
        await getDatasetData(dataset.id);

        return cognifyDataset(dataset, instance).finally(() => {
          setDatasetToProcessing(dataset, false);
        });
      });
    },
    [datasetsInProcessing, getDatasetData, instance, setDatasetToProcessing],
  );

  const handleDataRemoveConfirm = useCallback(
    (data: DataFile, event?: React.FormEvent<HTMLFormElement>) => {
      event?.preventDefault();

      setDatasetToProcessing({ id: data.datasetId } as Dataset);

      return removeDatasetData(data.datasetId, data.id).then(() => {
        refreshDatasetsAndData();
        setDatasetToProcessing({ id: data.datasetId } as Dataset, false);
      });
    },
    [refreshDatasetsAndData, removeDatasetData, setDatasetToProcessing],
  );

  const {
    isModalOpen: isNewDatasetModalOpen,
    openModal: openNewDatasetModal,
    closeModal: closeNewDatasetModal,
    confirmAction: handleNewDatasetSubmitConfirm,
    isActionLoading: isNewDatasetPending,
  } = useModal<Dataset>(false, handleNewDatasetSubmit);

  const {
    modalState: datasetToRemove,
    isModalOpen: isRemoveDatasetModalOpen,
    openModal: openRemoveDatasetModal,
    closeModal: closeRemoveDatasetModal,
    confirmAction: handleRemoveDatasetSubmitConfirm,
    isActionLoading: isRemoveDatasetPending,
  } = useModal<Dataset>(false, handleRemoveDatasetConfirm);

  const {
    modalState: dataToRemove,
    isModalOpen: isRemoveDataModalOpen,
    openModal: openRemoveDataModal,
    closeModal: closeRemoveDataModal,
    confirmAction: handleRemoveDataSubmitConfirm,
    isActionLoading: isRemoveDataPending,
  } = useModal<DataFile>(false, handleDataRemoveConfirm);

  return (
    <>
      <Accordion
        title={title || <span>Datasets</span>}
        isOpen={isDatasetsPanelOpen}
        openAccordion={openDatasetsPanel}
        closeAccordion={closeDatasetsPanel}
        tools={
          <div className="flex flex-row gap-4 items-center">
            {tools}
            <IconButton onClick={() => openNewDatasetModal()}>
              <PlusIcon />
            </IconButton>
          </div>
        }
        switchCaretPosition={switchCaretPosition}
        className={className}
        contentClassName={contentClassName}
      >
        <div className="flex flex-col">
          {datasets.length === 0 && !searchValue && (
            <div className="flex flex-row items-baseline-last text-sm text-gray-400 mt-2 px-2">
              <span>No datasets here, add one by clicking +</span>
            </div>
          )}
          {datasets.length === 0 && searchValue && (
            <div className="flex flex-row items-baseline-last text-sm text-gray-400 mt-2 px-2">
              <span>No datasets found, please adjust your search term</span>
            </div>
          )}
          {datasets.map((dataset) => {
            return (
              <Accordion
                key={dataset.id}
                title={
                  <div className="flex flex-row gap-2 items-center py-1.5 cursor-pointer">
                    {datasetsInProcessing?.[dataset.id] ? (
                      <LoadingIndicator />
                    ) : (
                      <DatasetIcon />
                    )}
                    <span className="text-xs">{dataset.name}</span>
                  </div>
                }
                isOpen={openDatasets.has(dataset.id)}
                openAccordion={() => toggleDataset(dataset.id)}
                closeAccordion={() => toggleDataset(dataset.id)}
                tools={
                  <IconButton className="relative">
                    <PopupMenu>
                      <div className="flex flex-col gap-0.5">
                        <div className="hover:bg-gray-100 w-full text-left px-2 cursor-pointer relative">
                          <input
                            tabIndex={-1}
                            type="file"
                            multiple
                            onChange={handleAddFiles.bind(null, dataset)}
                            className="absolute w-full h-full cursor-pointer opacity-0"
                          />
                          <span>add data</span>
                        </div>
                      </div>
                      <div className="flex flex-col gap-0.5 items-start">
                        <div
                          onClick={() => openRemoveDatasetModal(dataset)}
                          className="hover:bg-gray-100 w-full text-left px-2 cursor-pointer"
                        >
                          delete
                        </div>
                      </div>
                    </PopupMenu>
                  </IconButton>
                }
                className="first:pt-1.5"
                switchCaretPosition={true}
              >
                <>
                  {dataset.data?.length === 0 && (
                    <div className="flex flex-row items-baseline-last text-sm text-gray-400 mt-2 px-2">
                      <span>
                        No data in a dataset, add by clicking &quot;add
                        data&quot; in a dropdown menu
                      </span>
                    </div>
                  )}
                  {dataset.data?.map((data) => (
                    <div
                      key={data.id}
                      className="flex flex-row gap-2 items-center justify-between py-1.5 pl-6 last:pb-2.5"
                    >
                      <span className="text-xs">{data.name}</span>
                      <div>
                        <IconButton onClick={() => openRemoveDataModal(data)}>
                          <MinusIcon />
                        </IconButton>
                      </div>
                    </div>
                  ))}
                </>
              </Accordion>
            );
          })}
        </div>
      </Accordion>

      <CreateNewDatasetModal
        isNewDatasetPending={isNewDatasetPending}
        closeNewDatasetModal={closeNewDatasetModal}
        handleNewDatasetSubmitConfirm={handleNewDatasetSubmitConfirm}
        newDatasetError={newDatasetError}
        isOpen={isNewDatasetModalOpen}
      />

      {/* <Modal isOpen={isNewDatasetModalOpen}>
        <div className="w-full max-w-2xl">
          <div className="flex flex-row items-center justify-between">
            <span className="text-2xl">Create a new dataset?</span>
            <IconButton
              disabled={isNewDatasetPending}
              onClick={closeNewDatasetModal}
            >
              <CloseIcon />
            </IconButton>
          </div>
          <div className="mt-8 mb-6">
            Please provide a name for the dataset being created.
          </div>
          <form onSubmit={handleNewDatasetSubmitConfirm}>
            <div className="max-w-md">
              <Input
                name="datasetName"
                type="text"
                placeholder="Dataset name"
                required
              />
              {newDatasetError && (
                <span className="text-sm pl-4 text-gray-400">
                  {newDatasetError}
                </span>
              )}
            </div>
            <div className="flex flex-row gap-4 mt-4 justify-end">
              <GhostButton
                disabled={isNewDatasetPending}
                type="button"
                onClick={() => closeNewDatasetModal()}
              >
                cancel
              </GhostButton>
              <CTAButton disabled={isNewDatasetPending} type="submit">
                {isNewDatasetPending && <LoadingIndicator color="white" />}
                create
              </CTAButton>
            </div>
          </form>
        </div>
      </Modal> */}

      <Modal isOpen={isRemoveDatasetModalOpen}>
        <div className="w-full max-w-2xl">
          <div className="flex flex-row items-center justify-between">
            <span className="text-2xl">
              Delete{" "}
              <span className="text-cognee-purple">{datasetToRemove?.name}</span>{" "}
              dataset?
            </span>
            <IconButton
              disabled={isRemoveDatasetPending}
              onClick={closeRemoveDatasetModal}
            >
              <CloseIcon />
            </IconButton>
          </div>
          <div className="mt-8 mb-6">
            Are you sure you want to delete{" "}
            <span className="text-cognee-purple">{datasetToRemove?.name}</span>?
            This action cannot be undone.
          </div>
          <form
            onSubmit={handleRemoveDatasetSubmitConfirm}
            className="flex flex-row gap-4 mt-4 justify-end"
          >
            <GhostButton
              disabled={isRemoveDatasetPending}
              type="button"
              onClick={closeRemoveDatasetModal}
            >
              cancel
            </GhostButton>
            <CTAButton disabled={isRemoveDatasetPending} type="submit">
              delete
              {isRemoveDatasetPending && <LoadingIndicator color="white" />}
            </CTAButton>
          </form>
        </div>
      </Modal>

      <Modal isOpen={isRemoveDataModalOpen}>
        <div className="w-full max-w-2xl">
          <div className="flex flex-row items-center justify-between">
            <span className="text-2xl">
              Delete{" "}
              <span className="text-cognee-purple">{dataToRemove?.name}</span>{" "}
              data?
            </span>
            <IconButton
              disabled={isRemoveDataPending}
              onClick={closeRemoveDataModal}
            >
              <CloseIcon />
            </IconButton>
          </div>
          <div className="mt-8 mb-6">
            Are you sure you want to delete{" "}
            <span className="text-cognee-purple">{dataToRemove?.name}</span>? This
            action cannot be undone.
          </div>
          <form
            onSubmit={handleRemoveDataSubmitConfirm}
            className="flex flex-row gap-4 mt-4 justify-end"
          >
            <GhostButton
              disabled={isRemoveDataPending}
              type="button"
              onClick={closeRemoveDataModal}
            >
              cancel
            </GhostButton>
            <CTAButton disabled={isRemoveDataPending} type="submit">
              delete
              {isRemoveDataPending && <LoadingIndicator color="white" />}
            </CTAButton>
          </form>
        </div>
      </Modal>
    </>
  );
}
