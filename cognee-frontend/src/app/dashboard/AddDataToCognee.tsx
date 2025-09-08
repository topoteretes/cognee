import { FormEvent, useCallback, useState } from "react";
import { CloseIcon, PlusIcon } from "@/ui/Icons";
import { useModal } from "@/ui/elements/Modal";
import { CTAButton, GhostButton, IconButton, Modal, Select } from "@/ui/elements";

import addData from "@/modules/ingestion/addData";
import { Dataset } from "@/modules/ingestion/useDatasets";

interface AddDataToCogneeProps {
  datasets: Dataset[];
  refreshDatasets: () => void;
}

export default function AddDataToCognee({ datasets, refreshDatasets }: AddDataToCogneeProps) {
  const [filesForUpload, setFilesForUpload] = useState<FileList | null>(null);

  const prepareFiles = useCallback((event: FormEvent<HTMLInputElement>) => {
    const formElements = event.currentTarget;
    const files = formElements.files;

    setFilesForUpload(files);
  }, []);

  const processDataWithCognee = useCallback((state: object, event?: FormEvent<HTMLFormElement>) => {
    event!.preventDefault();

    if (!filesForUpload) {
      return;
    }

    const formElements = event!.currentTarget;
    const datasetId = formElements.datasetName.value;

    return addData(
      datasetId ? {
        id: datasetId,
      } : {
        name: "main_dataset",
      },
      Array.from(filesForUpload)
    )
      .then(() => {
        refreshDatasets();
        setFilesForUpload(null);
      });
  }, [filesForUpload, refreshDatasets]);

  const {
    isModalOpen: isAddDataModalOpen,
    openModal: openAddDataModal,
    closeModal: closeAddDataModal,
    isActionLoading: isProcessingDataWithCognee,
    confirmAction: submitDataToCognee,
  } = useModal(false, processDataWithCognee);

  return (
    <>
      <GhostButton onClick={openAddDataModal} className="mb-5 py-1.5 !px-2 text-sm w-full items-center justify-start">
        <PlusIcon />
        Add data to cognee
      </GhostButton>

      <Modal isOpen={isAddDataModalOpen}>
        <div className="w-full max-w-2xl">
          <div className="flex flex-row items-center justify-between">
            <span className="text-2xl">Add new data to a dataset?</span>
            <IconButton onClick={closeAddDataModal}><CloseIcon /></IconButton>
          </div>
          <div className="mt-8 mb-6">Please select a dataset to add data in.<br/> If you don&apos;t have any, don&apos;t worry, we will create one for you.</div>
          <form onSubmit={submitDataToCognee}>
            <div className="max-w-md flex flex-col gap-4">
              <Select name="datasetName">
                <option value="">select a dataset</option>
                {datasets.map((dataset: Dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}
              </Select>

              <GhostButton className="w-full relative justify-start pl-4">
                <input onChange={prepareFiles} required name="files" tabIndex={-1} type="file" multiple className="absolute w-full h-full cursor-pointer opacity-0" />
                <span>select files</span>
              </GhostButton>

              {filesForUpload?.length && (
                <div className="pt-4 mt-4 border-t-1 border-t-gray-100">
                  <div className="mb-1.5">selected files:</div>
                  {Array.from(filesForUpload || []).map((file) => (
                    <div key={file.name} className="py-1.5 pl-2">
                      <span className="text-sm">{file.name}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="flex flex-row gap-4 mt-4 justify-end">
              <GhostButton type="button" onClick={() => closeAddDataModal()}>cancel</GhostButton>
              <CTAButton disabled={isProcessingDataWithCognee} type="submit">
                {isProcessingDataWithCognee ? "processing..." : "add"}
              </CTAButton>
            </div>
          </form>
        </div>
      </Modal>
    </>
  );
}
