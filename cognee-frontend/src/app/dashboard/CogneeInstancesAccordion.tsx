"use client";

import { useBoolean } from "@/utils";
import { Accordion } from "@/ui/elements";

interface CogneeInstancesAccordionProps {
  children: React.ReactNode;
}

export default function CogneeInstancesAccordion({
  children,
}: CogneeInstancesAccordionProps) {
  const {
    value: isInstancesPanelOpen,
    setTrue: openInstancesPanel,
    setFalse: closeInstancesPanel,
  } = useBoolean(true);

  return (
    <>
      <Accordion
        title={<span>Cognee Instances</span>}
        isOpen={isInstancesPanelOpen}
        openAccordion={openInstancesPanel}
        closeAccordion={closeInstancesPanel}
      >
        {children}
      </Accordion>
    </>
  );
}
