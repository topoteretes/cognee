import { FormEvent, useCallback, useState } from "react";
import { useBoolean } from "@/utils";

export default function useModal<ConfirmActionReturnType = void>(initiallyOpen?: boolean, confirmCallback?: (state: object, event?: FormEvent<HTMLFormElement>) => Promise<ConfirmActionReturnType> | ConfirmActionReturnType) {
  const [modalState, setModalState] = useState<object>({});
  const [isActionLoading, setLoading] = useState(false);

  const {
    value: isModalOpen,
    setTrue: openModalInternal,
    setFalse: closeModalInternal,
  } = useBoolean(initiallyOpen || false);

  const openModal = useCallback((state?: object) => {
    if (state) {
      setModalState(state);
    }
    openModalInternal();
  }, [openModalInternal]);

  const closeModal = useCallback(() => {
    closeModalInternal();
    setModalState({});
  }, [closeModalInternal]);

  const confirmAction = useCallback((event?: FormEvent<HTMLFormElement>) => {
    if (confirmCallback) {
      setLoading(true);

      const maybePromise = confirmCallback(modalState, event);

      if (maybePromise instanceof Promise) {
        return maybePromise
          .finally(closeModal)
          .finally(() => setLoading(false));
      } else {
        return maybePromise; // Not a promise.
      }
    }
  }, [closeModal, confirmCallback, modalState]);

  return {
    isModalOpen,
    openModal,
    closeModal,
    confirmAction,
    isActionLoading,
  };
}
