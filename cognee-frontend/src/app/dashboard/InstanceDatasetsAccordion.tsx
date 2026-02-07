import classNames from "classnames";
import { useCallback, useEffect } from "react";

import { fetch, isCloudEnvironment, useBoolean } from "@/utils";
import { checkCloudConnection } from "@/modules/cloud";
import { setApiKey } from "@/modules/instances/cloudFetch";
import { CaretIcon, CloseIcon, CloudIcon, LocalCogneeIcon } from "@/ui/Icons";
import { CTAButton, GhostButton, IconButton, Input, Modal } from "@/ui/elements";

import DatasetsAccordion, { DatasetsAccordionProps } from "./DatasetsAccordion";

type InstanceDatasetsAccordionProps = Omit<DatasetsAccordionProps, "title">;

export default function InstanceDatasetsAccordion({ onDatasetsChange }: InstanceDatasetsAccordionProps) {
  const {
    value: isLocalCogneeConnected,
    setTrue: setLocalCogneeConnected,
  } = useBoolean(false);

  const {
    value: isCloudCogneeConnected,
    setTrue: setCloudCogneeConnected,
  } = useBoolean(isCloudEnvironment());

  const checkConnectionToCloudCognee = useCallback((apiKey?: string) => {
      if (apiKey) {
        fetch.setApiKey(apiKey);
        setApiKey(apiKey);
      }
      return checkCloudConnection()
        .then(setCloudCogneeConnected)
    }, [setCloudCogneeConnected]);

  useEffect(() => {
    const checkConnectionToLocalCognee = () => {
      fetch.checkHealth()
        .then(setLocalCogneeConnected)
    };

    checkConnectionToLocalCognee();
    checkConnectionToCloudCognee();
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

  const isCloudEnv = isCloudEnvironment();

  return (
    <div className={classNames("flex flex-col", {
      "flex-col-reverse": isCloudEnv,
    })}>
      <DatasetsAccordion
        title={(
          <div className="flex flex-row items-center justify-between">
            <div className="flex flex-row items-center gap-2">
              <LocalCogneeIcon className="text-indigo-700" />
              <span className="text-xs">local cognee</span>
            </div>
          </div>
        )}
        tools={isLocalCogneeConnected ? <span className="text-xs text-indigo-600">Connected</span> : <span className="text-xs text-gray-400">Not connected</span>}
        switchCaretPosition={true}
        className="pt-3 pb-1.5"
        contentClassName="pl-4"
        onDatasetsChange={!isCloudEnv ? onDatasetsChange : () => {}}
      />

      {isCloudCogneeConnected ? (
        <DatasetsAccordion
          title={(
            <div className="flex flex-row items-center justify-between">
              <div className="flex flex-row items-center gap-2">
                <LocalCogneeIcon className="text-indigo-700" />
                <span className="text-xs">cloud cognee</span>
              </div>
            </div>
          )}
          tools={<span className="text-xs text-indigo-600">Connected</span>}
          switchCaretPosition={true}
          className="pt-3 pb-1.5"
          contentClassName="pl-4"
          onDatasetsChange={isCloudEnv ? onDatasetsChange : () => {}}
          useCloud={true}
        />
      ) : (
        <button className="w-full flex flex-row items-center justify-between py-1.5 cursor-pointer pt-3" onClick={!isCloudCogneeConnected ? openCloudConnectionModal : () => {}}>
          <div className="flex flex-row items-center gap-1.5">
            <CaretIcon className="rotate-[-90deg]" />
            <div className="flex flex-row items-center gap-2">
              <CloudIcon color="#000000" />
              <span className="text-xs">cloud cognee</span>
            </div>
          </div>
          <span className="text-xs text-gray-400">Not connected</span>
        </button>
      )}

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
    </div>
  );
}
