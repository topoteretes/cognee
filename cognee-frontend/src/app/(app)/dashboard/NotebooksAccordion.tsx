"use client";

import { FormEvent, useCallback } from "react";
import { useBoolean } from "@/utils";
import { Accordion, CTAButton, GhostButton, IconButton, Input, Modal } from "@/ui/elements";
import { CloseIcon, MinusIcon, NotebookIcon, PlusIcon } from "@/ui/icons";
import { Notebook } from "@/ui/elements/Notebook/types";
import { useModal } from "@/ui/elements/Modal";
import { LoadingIndicator } from "@/ui/app";

interface NotebooksAccordionProps {
  notebooks: Notebook[];
  addNotebook: (name: string) => Promise<Notebook>;
  removeNotebook: (id: string) => Promise<void>;
  openNotebook: (id: string) => void;
}

export default function NotebooksAccordion({
  notebooks,
  addNotebook,
  removeNotebook,
  openNotebook,
}: NotebooksAccordionProps) {
  const {
    value: isNotebookPanelOpen,
    setTrue: openNotebookPanel,
    setFalse: closeNotebookPanel,
  } = useBoolean(true);

  const handleNotebookRemove = useCallback((notebook: Notebook, formEvent?: FormEvent<HTMLFormElement>) => {
    if (!formEvent) {
      return;
    }
    formEvent.preventDefault();

    return removeNotebook(notebook.id);
  }, [removeNotebook]);

  const handleNotebookAdd = useCallback((_: Notebook, formEvent?: FormEvent<HTMLFormElement>) => {
    if (!formEvent) {
      return;
    }
    formEvent.preventDefault();

    const formElements = formEvent.currentTarget;
    const notebookName = formElements.notebookName.value.trim();

    return addNotebook(notebookName)
      .then(() => {});
  }, [addNotebook]);

  const {
    modalState: notebookToRemove,
    isModalOpen: isRemoveNotebookModalOpen,
    openModal: openRemoveNotebookModal,
    closeModal: closeRemoveNotebookModal,
    confirmAction: handleNotebookRemoveConfirm,
    isActionLoading: isRemovNotebookPending,
  } = useModal<Notebook>(false, handleNotebookRemove);

  const {
    isModalOpen: isNewNotebookModalOpen,
    openModal: openNewNotebookModal,
    closeModal: closeNewNotebookModal,
    confirmAction: handleNewNotebookSubmit,
    isActionLoading: isNewNotebookPending,
  } = useModal<Notebook>(false, handleNotebookAdd);

  return (
    <>
      <Accordion
        title={<span>Notebooks</span>}
        isOpen={isNotebookPanelOpen}
        openAccordion={openNotebookPanel}
        closeAccordion={closeNotebookPanel}
        tools={isNewNotebookPending ? (
          <LoadingIndicator />
        ) : (
          <IconButton onClick={() => openNewNotebookModal()}>
            <PlusIcon />
          </IconButton>
        )}
      >
        {notebooks.length === 0 && (
          <div className="flex flex-row items-baseline-last text-sm text-gray-400 mt-2 px-2">
            <span>No notebooks here, add one by clicking +</span>
          </div>
        )}
        {notebooks.map((notebook: Notebook) => (
          <div key={notebook.id} className="flex flex-row gap-2.5 items-center justify-between py-1.5 first:pt-3">
            <button onClick={() => openNotebook(notebook.id)} className="flex flex-row gap-2 items-center cursor-pointer">
              {isNewNotebookPending ? <LoadingIndicator /> : <NotebookIcon />}
              <span className="text-xs">{notebook.name}</span>
            </button>
            <div>
              {notebook.deletable && <IconButton onClick={() => openRemoveNotebookModal(notebook)}><MinusIcon /></IconButton>}
            </div>
          </div>
        ))}
      </Accordion>

      <Modal isOpen={isNewNotebookModalOpen}>
        <div className="w-full max-w-2xl">
          <div className="flex flex-row items-center justify-between">
            <span className="text-2xl">Create a new notebook?</span>
            <IconButton onClick={closeNewNotebookModal}><CloseIcon /></IconButton>
          </div>
          <div className="mt-8 mb-6">Please provide a name for the notebook being created.</div>
          <form onSubmit={handleNewNotebookSubmit}>
            <div className="max-w-md">
              <Input name="notebookName" type="text" placeholder="Notebook name" required />
              {/* {newDatasetError && <span className="text-sm pl-4 text-gray-400">{newDatasetError}</span>} */}
            </div>
            <div className="flex flex-row gap-4 mt-4 justify-end">
              <GhostButton disabled={isNewNotebookPending} type="button" onClick={() => closeNewNotebookModal()}>cancel</GhostButton>
              <CTAButton disabled={isNewNotebookPending} type="submit">
                {isNewNotebookPending && <LoadingIndicator color="white" />}
                create
              </CTAButton>
            </div>
          </form>
        </div>
      </Modal>

      <Modal isOpen={isRemoveNotebookModalOpen}>
        <div className="w-full max-w-2xl">
          <div className="flex flex-row items-center justify-between">
            <span className="text-2xl">Delete <span className="text-cognee-purple">{notebookToRemove?.name}</span> notebook?</span>
            <IconButton onClick={closeRemoveNotebookModal}><CloseIcon /></IconButton>
          </div>
          <div className="mt-8 mb-6">Are you sure you want to delete <span className="text-cognee-purple">{notebookToRemove?.name}</span>? This action cannot be undone.</div>
          <form onSubmit={handleNotebookRemoveConfirm} className="flex flex-row gap-4 mt-4 justify-end">
            <GhostButton disabled={isRemovNotebookPending} type="button" onClick={closeRemoveNotebookModal}>cancel</GhostButton>
            <CTAButton disabled={isRemovNotebookPending} type="submit">
              {isRemovNotebookPending && <LoadingIndicator color="white" />}
              delete
            </CTAButton>
          </form>
        </div>
      </Modal>
    </>
  );
}
