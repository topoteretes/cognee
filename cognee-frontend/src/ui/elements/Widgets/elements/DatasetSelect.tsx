import { Dataset } from "@/modules/ingestion/useDatasets";
import { LoadingIndicator } from "@/ui/app";
import {
  Center,
  Flex,
  Select,
  Text,
} from "@mantine/core";
import { useCallback, useState } from "react";
import { useModal } from "../../Modal";
import CreateNewDatasetModal from "@/app/(app)/dashboard/elements/CreateNewDatasetAccordion";
import { notifications } from "@mantine/notifications";
import { trackEvent } from "@/modules/analytics";
import { tokens } from "@/ui/theme/tokens";

interface DatasetSelectProps {
  datasets: Dataset[];
  addDataset: (name: string) => Promise<void>;
  refreshDatasets: () => Promise<Dataset[]>;
  selectedValue: string | null;
  onChange: (value: string | null) => void;
  label?: string;
  hasAdd?: boolean;
}

export default function DatasetSelect({
  datasets,
  addDataset,
  refreshDatasets,
  selectedValue,
  onChange,
  label,
  hasAdd = false,
}: DatasetSelectProps) {
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
        trackEvent({ pageName: "Dashboard", eventName: "dataset_created", additionalProperties: { dataset_name: datasetName } });
        notifications.show({
          title: "New dataset added!",
          message: "",
          color: "green",
        });
        refreshDatasets().then((updated) => {
          const created = updated.find((d) => d.name === datasetName);
          if (created) onChange(created.id);
        });
      });
    },
    [addDataset, refreshDatasets, onChange],
  );

  const {
    isModalOpen: isNewDatasetModalOpen,
    openModal: openNewDatasetModal,
    closeModal: closeNewDatasetModal,
    confirmAction: handleNewDatasetSubmitConfirm,
    isActionLoading: isNewDatasetPending,
  } = useModal<Dataset>(false, handleNewDatasetSubmit);

  return (
    <Flex align="center" gap="0.5rem">
      {label && (
        <Text size="sm" c="dimmed" className="whitespace-nowrap">
          {label}
        </Text>
      )}
      {datasets.length !== 0 || hasAdd ? (
        <Select
          placeholder="Select dataset"
          allowDeselect={false}
          data={[
            ...Array.from(new Map(datasets.map((d) => [d.id, { value: d.id, label: d.name }])).values()),
            ...(hasAdd ? [{ value: "__new__", label: "+ Create New" }] : []),
          ]}
          value={selectedValue}
          size="sm"
          w={200}
          styles={{
            input: {
              borderColor: "#d9d9d9",
              borderRadius: "0.5rem",
              "&:focus": { borderColor: tokens.purple },
            },
            dropdown: {
              borderRadius: "0.5rem",
              boxShadow: "0 4px 12px rgba(0, 0, 0, 0.08)",
            },
          }}
          renderOption={({ option }) =>
            option.value === "__new__" ? (
              <span style={{ color: tokens.purple, fontWeight: 500 }}>{option.label}</span>
            ) : (
              <span>{option.label}</span>
            )
          }
          onChange={(value) => {
            if (value === "__new__") {
              openNewDatasetModal();
              return;
            }
            onChange(value);
          }}
        />
      ) : (
        <Center w={200}>
          <LoadingIndicator />
        </Center>
      )}
      {hasAdd && (
        <CreateNewDatasetModal
          isNewDatasetPending={isNewDatasetPending}
          closeNewDatasetModal={closeNewDatasetModal}
          handleNewDatasetSubmitConfirm={handleNewDatasetSubmitConfirm}
          newDatasetError={newDatasetError}
          isOpen={isNewDatasetModalOpen}
        />
      )}
    </Flex>
  );
}
