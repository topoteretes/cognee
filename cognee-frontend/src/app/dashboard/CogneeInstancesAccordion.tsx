"use client";

import { useCallback, useEffect } from "react";
import { fetch, useBoolean } from "@/utils";
import { Accordion, CTAButton, GhostButton, IconButton, Input, Modal } from "@/ui/elements";
import { CloseIcon, CloudIcon, LocalCogneeIcon } from "@/ui/Icons";
import { checkCloudConnection } from "@/modules/cloud";

export default function CogneeInstancesAccordion() {
  const {
    value: isInstancesPanelOpen,
    setTrue: openInstancesPanel,
    setFalse: closeInstancesPanel,
  } = useBoolean(true);

  const {
    value: isLocalCogneeConnected,
    setTrue: setLocalCogneeConnected,
  } = useBoolean(false);

  const {
    value: isCloudCogneeConnected,
    setTrue: setCloudCogneeConnected,
  } = useBoolean(false);

  const checkConnectionToCloudCognee = useCallback((apiKey: string) => {
      // checkCloudConnection("d8fa00e7fe326d4b2a32975a88bba27d6b8049d363e22b26")
      return checkCloudConnection(apiKey)
        .then(setCloudCogneeConnected)
    }, [setCloudCogneeConnected]);

  useEffect(() => {
    const checkConnectionToLocalCognee = () => {
      fetch.checkHealth()
        .then(setLocalCogneeConnected)
    };

    checkConnectionToLocalCognee();

    checkConnectionToCloudCognee("");
  }, [checkConnectionToCloudCognee, setCloudCogneeConnected, setLocalCogneeConnected]);

  const {
    value: isCloudConnectedModalOpen,
    setTrue: openCloudConnectionModal,
    setFalse: closeCloudConnectionModal,
  } = useBoolean(false);

  const handleCloudConnectionConfirm = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const apiKeyValue = event.currentTarget.apiKey.value;

    checkConnectionToCloudCognee(apiKeyValue)
      .then(() => {
        closeCloudConnectionModal();
      });
  };

  return (
    <>
      <Accordion
          title={<span>Cognee Instances</span>}
          isOpen={isInstancesPanelOpen}
          openAccordion={openInstancesPanel}
          closeAccordion={closeInstancesPanel}
        >
          <div className="flex flex-row items-center justify-between px-0.5 py-0.5">
            <div className="flex flex-row items-center gap-2.5">
              <LocalCogneeIcon className="text-indigo-700" />
              <span className="text-xs">local cognee</span>
            </div>
            {isLocalCogneeConnected ? <span className="text-xs text-indigo-600">Connected</span> : <span className="text-xs text-gray-400">Not connected</span>}
          </div>
          <button className="w-full flex flex-row items-center justify-between px-0.5 py-0.5" onClick={!isCloudCogneeConnected ? openCloudConnectionModal : () => {}}>
            <div className="flex flex-row items-center gap-2.5">
              <CloudIcon color="#000000" />
              <span className="text-xs">cloud cognee</span>
            </div>
            {isCloudCogneeConnected ? <span className="text-xs text-indigo-600">Connected</span> : <span className="text-xs text-gray-400">Not connected</span>}
          </button>
        </Accordion>

        <Modal isOpen={isCloudConnectedModalOpen}>
          <div className="w-full max-w-2xl">
            <div className="flex flex-row items-center justify-between">
              <span className="text-2xl">Connect to cloud?</span>
              <IconButton onClick={closeCloudConnectionModal}><CloseIcon /></IconButton>
            </div>
            <div className="mt-8 mb-6">Please provide your API key. You can find it on <a className="!text-indigo-600" href="https://platform.cognee.ai">our platform.</a></div>
            <form onSubmit={handleCloudConnectionConfirm}>
              <div className="max-w-md">
                <Input name="apiKey" type="text" placeholder="cloud API key" required />
              </div>
              <div className="flex flex-row gap-4 mt-4 justify-end">
                <GhostButton type="button" onClick={() => closeCloudConnectionModal()}>cancel</GhostButton>
                <CTAButton type="submit">connect</CTAButton>
              </div>
            </form>
          </div>
        </Modal>
      </>
  );
}
