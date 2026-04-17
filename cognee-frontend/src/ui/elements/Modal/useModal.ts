import { FormEvent, useCallback, useState } from "react";
import { useBoolean } from "@/utils";

export default function useModal<ModalState extends object, ConfirmActionEvent = FormEvent<HTMLFormElement>>(initiallyOpen?: boolean, confirmCallback?: (state: ModalState, event?: ConfirmActionEvent) => Promise<void> | void) {
  const [modalState, setModalState] = useState<ModalState>();
  const [isActionLoading, setLoading] = useState(false);

  const {
    value: isModalOpen,
    setTrue: openModalInternal,
    setFalse: closeModalInternal,
  } = useBoolean(initiallyOpen || false);

  const openModal = useCallback((state?: ModalState) => {
    if (state) {
      setModalState(state);
    }
    openModalInternal();
  }, [openModalInternal]);

  const closeModal = useCallback(() => {
    closeModalInternal();
    setModalState({} as ModalState);
  }, [closeModalInternal]);

  const confirmAction = useCallback((event?: ConfirmActionEvent) => {
    if (confirmCallback && !isActionLoading) {
      setLoading(true);

      try {
        const maybePromise = confirmCallback(modalState as ModalState, event);

        if (maybePromise instanceof Promise) {
          return maybePromise
            .finally(closeModal)
            .finally(() => {
              setTimeout(() => {
                setLoading(false)
              }, 100);
            });
        } else {
          closeModal();
          setLoading(false);
          return maybePromise; // Not a promise.
        }
      } catch {
        setLoading(false);
      }
    }
  }, [closeModal, confirmCallback, isActionLoading, modalState]);

  return {
    modalState,
    isModalOpen,
    openModal,
    closeModal,
    confirmAction,
    isActionLoading,
  };
}
