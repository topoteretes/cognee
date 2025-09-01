"use client";

import Link from "next/link";
import Image from "next/image";
import { useBoolean } from "@/utils";

import { CloseIcon, CloudIcon, CogneeIcon } from "../Icons";
import { CTAButton, GhostButton, IconButton, Modal } from "../elements";
import syncData from "@/modules/cloud/syncData";

export default function Header() {
  const {
    value: isSyncModalOpen,
    setTrue: openSyncModal,
    setFalse: closeSyncModal,
  } = useBoolean(false);

  const handleDataSyncConfirm = () => {
    syncData("2664f3a6-c9cc-514f-aa2e-15411a3a760c")
      .then(() => {
        closeSyncModal();
      });
  };

  return (
    <>
      <header className="relative bg-[rgba(244,244,244,0.5)] flex flex-row h-14 min-h-14 px-5 items-center justify-between w-full max-w-[1920px] mx-auto">
        <div className="flex flex-row gap-4 items-center">
          <CogneeIcon />
          <div className="text-lg">Cognee Graph Interface</div>
        </div>

        <div className="flex flex-row items-center gap-2.5">
          <GhostButton onClick={openSyncModal} className="text-indigo-700 gap-3 pl-4 pr-4">
            <CloudIcon />
            <div>Sync</div>
          </GhostButton>
          <a href="/plan">
            <GhostButton className="text-indigo-700 pl-4 pr-4">Premium</GhostButton>
          </a>
          {/* <div className="px-2 py-2 mr-3">
            <SettingsIcon />
          </div> */}
          <Link href="/account" className="bg-indigo-600 w-8 h-8 rounded-full overflow-hidden">
            <Image width="32" height="32" alt="Name of the user" src="/images/cognee-logo-with-text.png" />
          </Link>
        </div>
      </header>

      <Modal isOpen={isSyncModalOpen}>
        <div className="w-full max-w-2xl">
          <div className="flex flex-row items-center justify-between">
            <span className="text-2xl">Sync local datasets with cloud datasets?</span>
            <IconButton onClick={closeSyncModal}><CloseIcon /></IconButton>
          </div>
          <div className="mt-8 mb-6">Are you sure you want to sync local datasets to cloud?</div>
          <div className="flex flex-row gap-4 mt-4 justify-end">
            <GhostButton type="button" onClick={closeSyncModal}>cancel</GhostButton>
            <CTAButton onClick={handleDataSyncConfirm} type="submit">confirm</CTAButton>
          </div>
        </div>
      </Modal>
    </>
  );
}
