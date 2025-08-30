import { useBoolean } from "@/utils";
import { Accordion, IconButton } from "@/ui/elements";
import { NotebookIcon, PlusIcon } from "@/ui/Icons";
import { Notebook } from "@/ui/elements/Notebook/types";

interface NotebooksAccordionProps {
  notebooks: Notebook[];
  addNotebook: (name: string) => void;
  removeNotebook: (id: string) => void;
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

  return (
    <Accordion
      title={<span>Notebooks</span>}
      isOpen={isNotebookPanelOpen}
      openAccordion={openNotebookPanel}
      closeAccordion={closeNotebookPanel}
      tools={<IconButton onClick={() => addNotebook("Default Notebook Name")}><PlusIcon /></IconButton>}
    >
      {notebooks.map((notebook: Notebook) => (
        <button key={notebook.id} onClick={() => openNotebook(notebook.id)} className="flex flex-row gap-2.5 items-center px-0.5 py-1.5">
          <NotebookIcon />
          <span className="text-xs">{notebook.name}</span>
        </button>
      ))}
    </Accordion>
  );
}
