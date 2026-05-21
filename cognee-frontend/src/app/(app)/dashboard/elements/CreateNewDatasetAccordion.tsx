import { LoadingIndicator } from "@/ui/app";
import CTAButton from "@/ui/elements/CTAButton";
import GhostButton from "@/ui/elements/GhostButton";
import IconButton from "@/ui/elements/IconButton";
import Input from "@/ui/elements/Input";
import { Modal } from "@/ui/elements/Modal";
import CloseIcon from "@/ui/icons/CloseIcon";
import { FormEvent } from "react";

interface CreateNewDatasetModalProps {
  isOpen: boolean;
  isNewDatasetPending: boolean;
  closeNewDatasetModal: VoidFunction;
  handleNewDatasetSubmitConfirm: (
    event?: FormEvent<HTMLFormElement> | undefined,
  ) => void | Promise<void>;
  newDatasetError: string;
}

export default function CreateNewDatasetModal({
  isOpen,
  isNewDatasetPending,
  closeNewDatasetModal,
  handleNewDatasetSubmitConfirm,
  newDatasetError,
}: CreateNewDatasetModalProps) {
  return (
    <Modal isOpen={isOpen}>
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
    </Modal>
  );
}
